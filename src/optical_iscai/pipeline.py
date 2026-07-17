"""End-to-end range-Doppler detection, clustering, and tracking pipeline.

This module joins the processing stages that operate after frame simulation:
CA-CFAR detection, connected-component clustering, and multi-frame target
tracking.  The pipeline is stateful only through its tracker; each call processes
one range-Doppler map and advances tracks by the supplied frame interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from optical_iscai.cfar import (
    CFARConfiguration,
    CFARResult,
    Detection,
    detect_range_doppler,
)
from optical_iscai.clustering import (
    ClusterConfiguration,
    DetectionCluster,
    cluster_cfar_detections,
)
from optical_iscai.frame import FrameResult
from optical_iscai.range_doppler import RangeDopplerMap
from optical_iscai.tracking import (
    MultiTargetTracker,
    TrackEstimate,
    TrackerConfiguration,
)


@dataclass(frozen=True, slots=True)
class PipelineConfiguration:
    """Configuration of all post-FFT sensing stages."""

    cfar: CFARConfiguration = field(default_factory=CFARConfiguration)
    clustering: ClusterConfiguration = field(default_factory=ClusterConfiguration)
    tracking: TrackerConfiguration = field(default_factory=TrackerConfiguration)

    def validate(self) -> None:
        self.cfar.validate()
        self.clustering.validate()
        self.tracking.validate()


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Outputs from one complete post-FFT processing update."""

    range_doppler: RangeDopplerMap
    cfar: CFARResult
    cell_detections: tuple[Detection, ...]
    clusters: tuple[DetectionCluster, ...]
    tracks: tuple[TrackEstimate, ...]

    @property
    def confirmed_tracks(self) -> tuple[TrackEstimate, ...]:
        """Return only tracks that have reached the confirmation threshold."""
        return tuple(track for track in self.tracks if track.confirmed)


class SensingPipeline:
    """Stateful CFAR-to-tracker processing chain for successive frames."""

    def __init__(self, config: PipelineConfiguration | None = None) -> None:
        self.config = PipelineConfiguration() if config is None else config
        self.config.validate()
        self.tracker = MultiTargetTracker(self.config.tracking)

    def reset(self) -> None:
        """Clear all tracker state while preserving the pipeline configuration."""
        self.tracker.reset()

    def process_range_doppler(
        self,
        range_doppler: RangeDopplerMap,
        *,
        frame_interval_s: float,
    ) -> PipelineResult:
        """Process one range-Doppler map and advance the multi-target tracker."""
        cfar, cell_detections = detect_range_doppler(
            range_doppler,
            self.config.cfar,
        )
        clusters = cluster_cfar_detections(
            range_doppler,
            cfar,
            self.config.clustering,
        )
        tracks = self.tracker.update(clusters, frame_interval_s)
        return PipelineResult(
            range_doppler=range_doppler,
            cfar=cfar,
            cell_detections=cell_detections,
            clusters=clusters,
            tracks=tracks,
        )

    def process_frame(
        self,
        frame: FrameResult,
        *,
        frame_interval_s: float,
    ) -> PipelineResult:
        """Process the range-Doppler output contained in a simulated frame."""
        if not isinstance(frame, FrameResult):
            raise TypeError("frame must be a FrameResult")
        return self.process_range_doppler(
            frame.range_doppler,
            frame_interval_s=frame_interval_s,
        )
