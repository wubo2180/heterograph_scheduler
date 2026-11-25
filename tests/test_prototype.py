import unittest

from heterograph_scheduler.backends import CPUBackend
from heterograph_scheduler.communication import EdgeCodec
from heterograph_scheduler.features import extract_task_features
from heterograph_scheduler.models import DeviceKind, GraphTask, ResourceProfile, TaskKind
from heterograph_scheduler.scheduler import HeuristicScheduler
from heterograph_scheduler.simulation import simulate, synthetic_resources


def small_task() -> GraphTask:
    return GraphTask(
        task_id="small",
        kind=TaskKind.NEIGHBOR_AGGREGATION,
        num_nodes=3,
        num_edges=2,
        feature_dim=2,
        edges=((0, 1), (2, 1)),
        node_features=((1.0, 2.0), (0.0, 0.0), (3.0, 4.0)),
    )


class GraphModelTests(unittest.TestCase):
    def test_features_use_topology(self) -> None:
        features = extract_task_features(small_task())
        self.assertAlmostEqual(features.average_degree, 2 / 3)
        self.assertGreaterEqual(features.degree_skew, 1.0)
        self.assertEqual(len(features.as_vector()), 7)

    def test_invalid_edge_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GraphTask(
                "bad",
                TaskKind.NEIGHBOR_AGGREGATION,
                2,
                1,
                1,
                edges=((0, 2),),
            )

    def test_cpu_aggregation_executes(self) -> None:
        result = CPUBackend().execute(small_task())
        self.assertEqual(result.output, [[0.0, 0.0], [4.0, 6.0], [0.0, 0.0]])
        self.assertGreaterEqual(result.elapsed_ms, 0)


class SchedulingTests(unittest.TestCase):
    def test_unavailable_or_undersized_resource_is_not_selected(self) -> None:
        resources = [
            ResourceProfile(
                DeviceKind.GPU,
                1000,
                100,
                1,
                available=True,
                name="small-gpu",
            ),
            ResourceProfile(
                DeviceKind.CPU,
                10,
                10,
                1_000_000,
                available=True,
                name="cpu",
            ),
        ]
        self.assertEqual(
            HeuristicScheduler().schedule(small_task(), resources).device,
            DeviceKind.CPU,
        )

    def test_simulator_reports_only_estimates(self) -> None:
        result = simulate([small_task()], synthetic_resources())
        self.assertEqual(result["mode"], "synthetic_simulation")
        self.assertEqual(result["task_count"], 1)
        self.assertIn("estimated_total_latency_ms", result)


class CommunicationTests(unittest.TestCase):
    def test_edge_codec_is_lossless(self) -> None:
        edges = ((0, 1), (4, 8), (2, 2))
        payload, stats = EdgeCodec.encode(edges)
        self.assertEqual(EdgeCodec.decode(payload), edges)
        self.assertGreater(stats.original_bytes, 0)
        self.assertGreater(stats.ratio, 0)


if __name__ == "__main__":
    unittest.main()
