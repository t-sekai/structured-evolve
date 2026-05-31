"""Focused validation for cheap evolution rejection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evolution.cascade import CascadeCheckResult, preflight_generated_search_space
from src.evolution.loop import run_schedule_evolution
from src.evolution.prompts import evaluator_feedback
from src.evolution.space_loop import run_search_space_evolution


class RejectionCascadeTest(unittest.TestCase):
    def test_level1_syntax_rejection_skips_full_benchmark_and_reaches_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed = root / "broken_schedule.py"
            seed.write_text("def broken(:\n", encoding="utf-8")

            with patch("src.evolution.loop.run_matmul_experiment") as expensive:
                history = run_schedule_evolution(
                    seed_candidate_path=seed,
                    run_dir=root / "run",
                    output_dir=root / "results",
                    generations=0,
                    population_size=1,
                    survivors=1,
                    target_name="llvm",
                    M=16,
                    N=16,
                    K=16,
                    num_warmup=0,
                    num_trials=1,
                    benchmark_invocations=1,
                    min_repeat_ms=None,
                    bedrock_client=None,
                    dry_run=True,
                )

            expensive.assert_not_called()
            result = history[0]["result"]
            self.assertTrue(result["cascade_rejected"])
            self.assertTrue(result["tuning_skipped"])
            self.assertFalse(result["syntax_check_passed"])
            self.assertEqual(result["rejection_stage"], "syntax")
            feedback = evaluator_feedback(history[0])
            self.assertIn("- rejection_stage: syntax", feedback)
            self.assertIn("- rejection_reason: SyntaxError:", feedback)

    def test_level2_syntax_rejection_skips_tuning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed = root / "broken_space.py"
            seed.write_text("def broken(:\n", encoding="utf-8")

            with patch("src.evolution.space_loop.run_matmul_experiment") as expensive:
                history = run_search_space_evolution(
                    seed_candidate_path=seed,
                    run_dir=root / "run",
                    output_dir=root / "results",
                    generations=0,
                    population_size=1,
                    survivors=1,
                    target_name="llvm",
                    M=16,
                    N=16,
                    K=16,
                    num_warmup=0,
                    num_trials=1,
                    benchmark_invocations=1,
                    min_repeat_ms=None,
                    max_trials_global=2,
                    num_trials_per_iter=1,
                    cost_model="random",
                    task_scheduler="round-robin",
                    seed=0,
                    num_tuning_cores=1,
                    bedrock_client=None,
                    dry_run=True,
                )

            expensive.assert_not_called()
            result = history[0]["result"]
            self.assertTrue(result["cascade_rejected"])
            self.assertTrue(result["tuning_skipped"])
            self.assertEqual(result["rejection_stage"], "syntax")

    def test_level2_correctness_preflight_rejection_skips_tuning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed = root / "incorrect_space.py"
            seed.write_text("import tvm\n", encoding="utf-8")
            rejection = CascadeCheckResult(
                passed=False,
                stage="correctness",
                reason="preflight_correctness_failed: max_abs_error=1",
                metadata={
                    "syntax_check_passed": True,
                    "preflight_compile_passed": True,
                    "preflight_correctness_passed": False,
                },
            )

            with patch(
                "src.evolution.space_loop.preflight_generated_search_space",
                return_value=rejection,
            ):
                with patch("src.evolution.space_loop.run_matmul_experiment") as expensive:
                    history = run_search_space_evolution(
                        seed_candidate_path=seed,
                        run_dir=root / "run",
                        output_dir=root / "results",
                        generations=0,
                        population_size=1,
                        survivors=1,
                        target_name="llvm",
                        M=16,
                        N=16,
                        K=16,
                        num_warmup=0,
                        num_trials=1,
                        benchmark_invocations=1,
                        min_repeat_ms=None,
                        max_trials_global=2,
                        num_trials_per_iter=1,
                        cost_model="random",
                        task_scheduler="round-robin",
                        seed=0,
                        num_tuning_cores=1,
                        bedrock_client=None,
                        dry_run=True,
                    )

            expensive.assert_not_called()
            result = history[0]["result"]
            self.assertEqual(result["rejection_stage"], "correctness")
            self.assertTrue(result["preflight_compile_passed"])
            self.assertFalse(result["preflight_correctness_passed"])
            self.assertIn("max_abs_error=1", result["rejection_reason"])

    def test_factory_search_space_skips_direct_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "factory_space.py"
            candidate.write_text(
                "def create_space_generator(target_name):\n"
                "    return object()\n",
                encoding="utf-8",
            )

            result = preflight_generated_search_space(
                candidate,
                m=16,
                n=16,
                k=16,
                target_name="llvm",
            )

            self.assertTrue(result.passed)
            self.assertEqual(result.stage, "accepted")
            self.assertTrue(result.metadata["preflight_skipped"])
            self.assertEqual(
                result.metadata["preflight_reason"],
                "factory_search_space_requires_tune_context",
            )


if __name__ == "__main__":
    unittest.main()
