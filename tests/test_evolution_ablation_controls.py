"""Focused validation for evolution ablation controls."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evolution.selection import plan_next_generation
from src.evolution.loop import run_schedule_evolution
from src.evolution.space_loop import run_search_space_evolution


class EvolutionAblationControlsTest(unittest.TestCase):
    def test_selection_can_preserve_elite_and_use_valid_diverse_inspiration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            elite = _selection_row(
                root=root,
                candidate_id="elite",
                filename="elite.py",
                code="def schedule():\n    return 1\n",
                score=10.0,
            )
            duplicate = _selection_row(
                root=root,
                candidate_id="duplicate",
                filename="duplicate.py",
                code="def schedule():\n    return 1\n",
                score=9.0,
            )
            inspiration = _selection_row(
                root=root,
                candidate_id="inspiration",
                filename="inspiration.py",
                code="def schedule():\n    return 2\n",
                score=8.0,
            )
            invalid = _selection_row(
                root=root,
                candidate_id="invalid",
                filename="invalid.py",
                code="def schedule():\n    return 3\n",
                compile_passed=False,
                correctness_passed=False,
                score=7.0,
            )

            plan = plan_next_generation(
                [elite, duplicate, inspiration, invalid],
                survivors=1,
                population_size=3,
                enable_elite_carry_forward=True,
                enable_diverse_inspiration=True,
            )

            self.assertEqual([row["candidate"]["candidate_id"] for row in plan.elites], ["elite"])
            self.assertEqual(
                [source.selection_source_role for source in plan.mutation_sources],
                ["diverse_inspiration", "elite_parent"],
            )
            self.assertEqual(
                plan.mutation_sources[0].result["candidate"]["candidate_id"],
                "inspiration",
            )

    def test_selection_defaults_mutate_survivors_without_carry_forward(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            elite = _selection_row(
                root=root,
                candidate_id="elite",
                filename="elite.py",
                code="def schedule():\n    return 1\n",
                score=10.0,
            )
            inspiration = _selection_row(
                root=root,
                candidate_id="inspiration",
                filename="inspiration.py",
                code="def schedule():\n    return 2\n",
                score=9.0,
            )

            plan = plan_next_generation(
                [elite, inspiration],
                survivors=1,
                population_size=2,
            )

            self.assertEqual(plan.elites, [])
            self.assertEqual(
                [source.selection_source_role for source in plan.mutation_sources],
                ["elite_parent", "elite_parent"],
            )
            self.assertEqual(
                [source.result["candidate"]["candidate_id"] for source in plan.mutation_sources],
                ["elite", "elite"],
            )

    def test_diverse_inspiration_does_not_replace_only_parent_mutation_slot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            elite = _selection_row(
                root=root,
                candidate_id="elite",
                filename="elite.py",
                code="def schedule():\n    return 1\n",
                score=10.0,
            )
            inspiration = _selection_row(
                root=root,
                candidate_id="inspiration",
                filename="inspiration.py",
                code="def schedule():\n    return 2\n",
                score=9.0,
            )

            plan = plan_next_generation(
                [elite, inspiration],
                survivors=1,
                population_size=1,
                enable_diverse_inspiration=True,
            )

            self.assertEqual(plan.elites, [])
            self.assertEqual(
                [source.selection_source_role for source in plan.mutation_sources],
                ["elite_parent"],
            )
            self.assertEqual(
                plan.mutation_sources[0].result["candidate"]["candidate_id"],
                "elite",
            )

    def test_level1_ablation_toggles_disable_feedback_and_cascade(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed = root / "broken_schedule.py"
            seed.write_text("def broken(:\n", encoding="utf-8")

            with patch(
                "src.evolution.loop.run_matmul_experiment",
                side_effect=_successful_experiment_result,
            ) as expensive:
                history = run_schedule_evolution(
                    seed_candidate_path=seed,
                    run_dir=root / "run",
                    output_dir=root / "results",
                    generations=1,
                    population_size=2,
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
                    include_evaluator_feedback=False,
                    enable_rejection_cascade=False,
                    enable_elite_carry_forward=True,
                    enable_diverse_inspiration=False,
                )

            self.assertEqual(expensive.call_count, 3)
            selection = json.loads((root / "run/gen_001/selection.json").read_text())
            self.assertFalse(selection["evaluator_feedback_enabled"])
            self.assertFalse(selection["rejection_cascade_enabled"])
            self.assertTrue(selection["elite_carry_forward_enabled"])
            self.assertFalse(selection["diverse_inspiration_enabled"])
            self.assertEqual(selection["elites"][0]["candidate_id"], "g000_c000_seed")
            self.assertTrue((root / "run/gen_001/candidate_000_elite.py").exists())

            prompt = (root / "run/gen_001/candidate_001.prompt.txt").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("Parent evaluator feedback:", prompt)
            self.assertTrue(
                all(not row["result"]["rejection_cascade_enabled"] for row in history)
            )
            self.assertTrue(
                all(row["result"]["elite_carry_forward_enabled"] for row in history)
            )

    def test_level2_ablation_toggles_disable_feedback_and_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed = root / "broken_space.py"
            seed.write_text("def broken(:\n", encoding="utf-8")

            with patch("src.evolution.space_loop.preflight_generated_search_space") as preflight:
                with patch(
                    "src.evolution.space_loop.run_matmul_experiment",
                    side_effect=_successful_experiment_result,
                ) as expensive:
                    history = run_search_space_evolution(
                        seed_candidate_path=seed,
                        run_dir=root / "run",
                        output_dir=root / "results",
                        generations=1,
                        population_size=2,
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
                        include_evaluator_feedback=False,
                        enable_rejection_cascade=False,
                        enable_elite_carry_forward=True,
                        enable_diverse_inspiration=False,
                    )

            preflight.assert_not_called()
            self.assertEqual(expensive.call_count, 3)
            selection = json.loads((root / "run/gen_001/selection.json").read_text())
            self.assertFalse(selection["evaluator_feedback_enabled"])
            self.assertFalse(selection["rejection_cascade_enabled"])
            self.assertTrue(selection["elite_carry_forward_enabled"])
            prompt = (root / "run/gen_001/candidate_001.prompt.txt").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("Parent evaluator feedback:", prompt)
            self.assertTrue(
                all(not row["result"]["rejection_cascade_enabled"] for row in history)
            )


def _selection_row(
    *,
    root: Path,
    candidate_id: str,
    filename: str,
    code: str,
    compile_passed: bool = True,
    correctness_passed: bool = True,
    score: float,
) -> dict:
    path = root / filename
    path.write_text(code, encoding="utf-8")
    return {
        "candidate": {
            "candidate_id": candidate_id,
            "generation": 0,
            "path": str(path),
            "parent_id": None,
            "prompt_path": None,
            "response_path": None,
        },
        "fitness": {"score": score, "reason": "test"},
        "result": {
            "compile_passed": compile_passed,
            "correctness_passed": correctness_passed,
            "latency_ms_mean": 1.0,
            "latency_ms_std": 0.0,
        },
    }


def _successful_experiment_result(**kwargs) -> dict:
    result = {
        **_jsonable_metadata(dict(kwargs.get("extra_metadata") or {})),
        "strategy": kwargs["strategy"].name,
        "level": kwargs["strategy"].level,
        "M": kwargs["M"],
        "N": kwargs["N"],
        "K": kwargs["K"],
        "target": kwargs["target_name"],
        "compile_passed": True,
        "correctness_passed": True,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "latency_ms_mean": 1.0,
        "latency_ms_std": 0.0,
        "error_type": "",
        "error_message": "",
        "fitness_score": 1.0,
        "fitness_reason": "mock success",
    }
    postprocess = kwargs.get("postprocess_result")
    if postprocess is not None:
        postprocessed = postprocess(result)
        if postprocessed:
            result.update(postprocessed)
    return result


def _jsonable_metadata(metadata: dict) -> dict:
    return {key: _jsonable_value(value) for key, value in metadata.items()}


def _jsonable_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable_value(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    unittest.main()
