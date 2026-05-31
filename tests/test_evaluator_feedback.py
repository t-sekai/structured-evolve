"""Focused validation for prompt-facing evaluator feedback."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evolution.loop import run_schedule_evolution
from src.evolution.space_loop import run_search_space_evolution


class EvaluatorFeedbackTest(unittest.TestCase):
    def test_level1_persisted_prompt_includes_ranked_metrics_and_concise_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_path = root / "seed.py"
            seed_path.write_text("import tvm\n", encoding="utf-8")
            row = _evaluated_row(
                candidate_path=seed_path,
                result={
                    "compile_passed": True,
                    "correctness_passed": False,
                    "latency_ms_mean": 1.25,
                    "latency_ms_std": 0.125,
                    "error_type": "RuntimeError",
                    "error_message": "bad\n schedule " + "x" * 400,
                },
            )

            with patch("src.evolution.loop._evaluate_candidate", return_value=row):
                run_schedule_evolution(
                    seed_candidate_path=seed_path,
                    run_dir=root / "run",
                    output_dir=root / "results",
                    generations=1,
                    population_size=2,
                    survivors=1,
                    target_name="llvm",
                    M=16,
                    N=16,
                    K=16,
                    num_warmup=3,
                    num_trials=10,
                    benchmark_invocations=100,
                    min_repeat_ms=10,
                    bedrock_client=None,
                    dry_run=True,
                )

            prompt = (root / "run/gen_001/candidate_001.prompt.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("Parent evaluator feedback:", prompt)
            self.assertIn("- candidate_id: g000_c000_seed", prompt)
            self.assertIn("- generation_rank: 1", prompt)
            self.assertIn("- compile_passed: True", prompt)
            self.assertIn("- correctness_passed: False", prompt)
            self.assertIn("- latency_ms_mean: 1.25", prompt)
            self.assertIn("- latency_ms_std: 0.125", prompt)
            self.assertIn("- fitness_score: 0.8", prompt)
            self.assertIn("- error: RuntimeError: bad schedule ", prompt)
            self.assertIn("...", prompt)
            self.assertNotIn("bad\n schedule", prompt)

    def test_level2_persisted_prompt_includes_artifacts_and_bounded_schedule_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_path = root / "seed.py"
            seed_path.write_text("import tvm\n", encoding="utf-8")
            work_dir = root / "work"
            scheduled_module_path = work_dir / "scheduled_module.txt"
            scheduled_module_json_path = work_dir / "scheduled_module.json"
            scheduled_module_path.parent.mkdir(parents=True)
            scheduled_module_path.write_text(
                "\n".join(
                    ["# selected schedule", *[f"line_{index}" for index in range(80)]]
                )
                + "\nSHOULD_NOT_APPEAR\n",
                encoding="utf-8",
            )
            tuning_record_path = work_dir / "database_tuning_record.json"
            workload_path = work_dir / "database_workload.json"
            row = _evaluated_row(
                candidate_path=seed_path,
                result={
                    "compile_passed": True,
                    "correctness_passed": True,
                    "latency_ms_mean": 0.5,
                    "latency_ms_std": 0.05,
                    "error_type": "",
                    "error_message": "",
                    "metaschedule_work_dir": str(work_dir),
                    "metaschedule_database_tuning_record": str(tuning_record_path),
                    "metaschedule_database_workload": str(workload_path),
                    "scheduled_module_path": str(scheduled_module_path),
                    "scheduled_module_json_path": str(scheduled_module_json_path),
                    "used_fallback_schedule": False,
                },
            )

            with patch("src.evolution.space_loop._evaluate_candidate", return_value=row):
                run_search_space_evolution(
                    seed_candidate_path=seed_path,
                    run_dir=root / "run",
                    output_dir=root / "results",
                    generations=1,
                    population_size=2,
                    survivors=1,
                    target_name="llvm",
                    M=16,
                    N=16,
                    K=16,
                    num_warmup=3,
                    num_trials=10,
                    benchmark_invocations=100,
                    min_repeat_ms=10,
                    max_trials_global=2,
                    num_trials_per_iter=1,
                    cost_model="random",
                    task_scheduler="round-robin",
                    seed=0,
                    num_tuning_cores=1,
                    bedrock_client=None,
                    dry_run=True,
                )

            prompt = (root / "run/gen_001/candidate_001.prompt.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("- generation_rank: 1", prompt)
            self.assertIn(f"- metaschedule_work_dir: {work_dir}", prompt)
            self.assertIn(f"- metaschedule_database_tuning_record: {tuning_record_path}", prompt)
            self.assertIn(f"- metaschedule_database_workload: {workload_path}", prompt)
            self.assertIn(f"- scheduled_module_path: {scheduled_module_path}", prompt)
            self.assertIn(f"- scheduled_module_json_path: {scheduled_module_json_path}", prompt)
            self.assertIn("- used_fallback_schedule: False", prompt)
            self.assertIn("- selected_schedule_summary:\n  # selected schedule", prompt)
            self.assertIn("\n  ...", prompt)
            self.assertNotIn("SHOULD_NOT_APPEAR", prompt)


def _evaluated_row(*, candidate_path: Path, result: dict) -> dict:
    return {
        "candidate": {
            "candidate_id": "g000_c000_seed",
            "generation": 0,
            "path": str(candidate_path),
            "parent_id": None,
            "prompt_path": None,
            "response_path": None,
        },
        "fitness": {"score": 0.8, "reason": "test fitness"},
        "result": result,
    }


if __name__ == "__main__":
    unittest.main()
