"""Focused validation for configurable TVM timing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.eval.benchmark import measure_latency_ms
from src.eval.method import MethodRunConfig, _run_level1_search, _run_level2_search
from src.eval.suite import SuiteRunConfig, run_benchmark_suite


class _FakeDevice:
    def __init__(self) -> None:
        self.sync_count = 0

    def sync(self) -> None:
        self.sync_count += 1


class _FakeLib:
    def __init__(self, results: list[float]) -> None:
        self.results = results
        self.main_calls = 0
        self.evaluator_call = None

    def __getitem__(self, name: str):
        if name != "main":
            raise KeyError(name)

        def main(*_args) -> None:
            self.main_calls += 1

        return main

    def time_evaluator(self, *args, **kwargs):
        self.evaluator_call = (args, kwargs)

        def evaluator(*_args):
            return type("TimingResult", (), {"results": self.results})()

        return evaluator


class StableTimingTest(unittest.TestCase):
    def test_passes_invocation_count_and_minimum_repeat_duration(self) -> None:
        device = _FakeDevice()
        lib = _FakeLib([0.001, 0.003])

        stats = measure_latency_ms(
            lib=lib,
            device=device,
            a_tvm=object(),
            b_tvm=object(),
            c_tvm=object(),
            num_warmup=2,
            num_trials=2,
            benchmark_invocations=7,
            min_repeat_ms=25,
        )

        self.assertEqual(
            lib.evaluator_call,
            (("main", device), {"number": 7, "repeat": 2, "min_repeat_ms": 25}),
        )
        self.assertEqual(lib.main_calls, 2)
        self.assertEqual(device.sync_count, 1)
        self.assertEqual(stats.samples_ms, [1.0, 3.0])
        self.assertEqual(stats.mean, 2.0)
        self.assertEqual(stats.std, 1.0)

    def test_omits_optional_minimum_to_preserve_tvm_default(self) -> None:
        device = _FakeDevice()
        lib = _FakeLib([0.002])

        measure_latency_ms(
            lib=lib,
            device=device,
            a_tvm=object(),
            b_tvm=object(),
            c_tvm=object(),
            num_warmup=0,
            num_trials=1,
        )

        self.assertEqual(
            lib.evaluator_call,
            (("main", device), {"number": 1, "repeat": 1}),
        )

    def test_suite_manifest_records_shared_timing_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite_dir = root / "suite"

            run_benchmark_suite(
                tasks=[],
                methods=[],
                config=SuiteRunConfig(
                    output_dir=root / "results",
                    suite_dir=suite_dir,
                    experiment_id="stable_timing_test",
                    suite_name="stable_timing",
                    num_warmup=3,
                    num_trials=10,
                    benchmark_invocations=100,
                    min_repeat_ms=10,
                ),
            )

            manifest = json.loads((suite_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["benchmark_invocations"], 100)
            self.assertEqual(manifest["min_repeat_ms"], 10)

    def test_search_timing_overrides_reach_both_evolution_loops(self) -> None:
        config = MethodRunConfig(
            M=16,
            N=16,
            K=16,
            target_name="llvm",
            output_dir=Path("unused"),
            benchmark_invocations=100,
            min_repeat_ms=10,
            search_benchmark_invocations=7,
            search_min_repeat_ms=3,
        )
        history = [{"candidate": {"path": "candidate.py"}}]

        for runner_name, runner, target in (
            ("level1", _run_level1_search, "src.eval.method.run_schedule_evolution"),
            ("level2", _run_level2_search, "src.eval.method.run_search_space_evolution"),
        ):
            with self.subTest(runner=runner_name):
                with patch(target, return_value=history) as run_evolution:
                    with patch("src.eval.method._run_evolved_best", return_value={}):
                        runner(config=config)

                self.assertEqual(run_evolution.call_args.kwargs["benchmark_invocations"], 7)
                self.assertEqual(run_evolution.call_args.kwargs["min_repeat_ms"], 3)


if __name__ == "__main__":
    unittest.main()
