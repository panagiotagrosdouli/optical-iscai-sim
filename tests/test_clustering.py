import numpy as np
import pytest

from optical_iscai.cfar import CFARResult
from optical_iscai.clustering import ClusterConfiguration, cluster_cfar_detections
from optical_iscai.range_doppler import RangeDopplerMap


def make_result(power: np.ndarray, mask: np.ndarray) -> tuple[RangeDopplerMap, CFARResult]:
    velocity = np.arange(power.shape[0], dtype=np.float64) - 2.0
    ranges = np.arange(power.shape[1], dtype=np.float64) * 10.0
    rd = RangeDopplerMap(range_m=ranges, velocity_m_s=velocity, power=power)
    cfar = CFARResult(
        threshold=np.ones_like(power),
        noise_power=np.ones_like(power),
        detections=mask,
        training_cell_count=8,
        scale_factor=2.0,
    )
    return rd, cfar


def test_adjacent_cells_form_one_power_weighted_cluster() -> None:
    power = np.zeros((5, 6), dtype=np.float64)
    mask = np.zeros_like(power, dtype=bool)
    power[2, 2] = 2.0
    power[2, 3] = 6.0
    power[3, 3] = 2.0
    mask[2, 2] = mask[2, 3] = mask[3, 3] = True
    rd, cfar = make_result(power, mask)

    clusters = cluster_cfar_detections(rd, cfar)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.cell_count == 3
    assert cluster.total_power == pytest.approx(10.0)
    assert cluster.range_m == pytest.approx(28.0)
    assert cluster.velocity_m_s == pytest.approx(0.2)
    assert cluster.peak_range_index == 3
    assert cluster.peak_velocity_index == 2
    assert cluster.peak_power == pytest.approx(6.0)


def test_separated_components_are_sorted_by_total_power() -> None:
    power = np.zeros((5, 6), dtype=np.float64)
    mask = np.zeros_like(power, dtype=bool)
    power[1, 1] = 3.0
    power[3, 4] = 5.0
    mask[1, 1] = True
    mask[3, 4] = True
    rd, cfar = make_result(power, mask)

    clusters = cluster_cfar_detections(rd, cfar)

    assert len(clusters) == 2
    assert clusters[0].peak_range_index == 4
    assert clusters[1].peak_range_index == 1


def test_connectivity_controls_diagonal_merging() -> None:
    power = np.ones((4, 4), dtype=np.float64)
    mask = np.zeros_like(power, dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    rd, cfar = make_result(power, mask)

    assert len(cluster_cfar_detections(rd, cfar, ClusterConfiguration(connectivity=8))) == 1
    assert len(cluster_cfar_detections(rd, cfar, ClusterConfiguration(connectivity=4))) == 2


def test_cluster_filters_small_or_weak_components() -> None:
    power = np.zeros((5, 6), dtype=np.float64)
    mask = np.zeros_like(power, dtype=bool)
    power[1, 1] = 1.0
    power[3, 3] = 2.0
    power[3, 4] = 2.0
    mask[1, 1] = mask[3, 3] = mask[3, 4] = True
    rd, cfar = make_result(power, mask)

    clusters = cluster_cfar_detections(
        rd,
        cfar,
        ClusterConfiguration(minimum_cells=2, minimum_total_power=3.0),
    )

    assert len(clusters) == 1
    assert clusters[0].cell_count == 2


def test_clustering_rejects_invalid_configuration_and_shapes() -> None:
    power = np.ones((4, 4), dtype=np.float64)
    mask = np.zeros_like(power, dtype=bool)
    rd, cfar = make_result(power, mask)

    with pytest.raises(ValueError, match="connectivity"):
        cluster_cfar_detections(rd, cfar, ClusterConfiguration(connectivity=6))
    with pytest.raises(ValueError, match="positive integer"):
        cluster_cfar_detections(rd, cfar, ClusterConfiguration(minimum_cells=0))

    bad_cfar = CFARResult(
        threshold=np.ones((3, 3)),
        noise_power=np.ones((3, 3)),
        detections=np.zeros((3, 3), dtype=bool),
        training_cell_count=8,
        scale_factor=2.0,
    )
    with pytest.raises(ValueError, match="mask shape"):
        cluster_cfar_detections(rd, bad_cfar)
