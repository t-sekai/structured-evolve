"""Focused validation for explicit Level 2 final-evaluation policies."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tvm

from src.eval.method import MethodRunConfig, _run_evolved_best
from src.kernels.matmul_tir import create_matmul_ir_module
from src.strategies.base import StrategyBuildConfig
from src.strategies.saved_scheduled_module import SavedScheduledModuleStrategy


class Level2FinalEvaluationTest(unittest.TestCase):
    def test_exact_winner_uses_serialized_search_schedule_without_retuning(self) -> None:
        result = self._run_policy("exact-winner")
        strategy_config = result["strategy_config"]
        metadata = result["extra_metadata"]

        self.assertEqual(result["strategy_name"], "saved-scheduled-module")
        self.assertEqual(
            strategy_config.saved_scheduled_module_json_path,
            Path("search/scheduled_module.json"),
        )
        self.assertEqual(metadata["selection_role"], "best_of_search_exact_winner")
        self.assertEqual(metadata["final_evaluation_policy"], "exact-winner")
        self.assertEqual(
            metadata["final_evaluation_semantics"],
            "reuse_search_time_scheduled_module",
        )
        self.assertTrue(metadata["exact_schedule_reused"])
        self.assertEqual(
            metadata["search_winner_metaschedule_database_workload"],
            "search/database_workload.json",
        )
        self.assertFalse(metadata["search_winner_used_fallback_schedule"])

    def test_fresh_retune_uses_winning_search_space_generator(self) -> None:
        result = self._run_policy("fresh-retune")
        strategy_config = result["strategy_config"]
        metadata = result["extra_metadata"]

        self.assertEqual(result["strategy_name"], "generated-search-space")
        self.assertEqual(
            strategy_config.generated_search_space_path,
            Path("generated/search_spaces/cuda_matmul.py"),
        )
        self.assertEqual(metadata["selection_role"], "best_of_search_fresh_retune")
        self.assertEqual(metadata["final_evaluation_policy"], "fresh-retune")
        self.assertEqual(
            metadata["final_evaluation_semantics"],
            "retune_winning_search_space_generator",
        )
        self.assertFalse(metadata["exact_schedule_reused"])

    def test_saved_schedule_strategy_loads_lossless_ir_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "scheduled_module.json"
            scheduled_module = create_matmul_ir_module(16, 16, 16)
            json_path.write_text(tvm.ir.save_json(scheduled_module), encoding="utf-8")

            with patch("src.strategies.saved_scheduled_module.tvm.build", return_value="lib"):
                build_result = SavedScheduledModuleStrategy().build(
                    ir_module=create_matmul_ir_module(16, 16, 16),
                    target=tvm.target.Target("llvm"),
                    target_name="llvm",
                    config=StrategyBuildConfig(
                        saved_scheduled_module_json_path=json_path,
                    ),
                )

            self.assertTrue(
                tvm.ir.structural_equal(build_result.scheduled_module, scheduled_module)
            )
            self.assertEqual(build_result.metadata["scheduled_module_json_path"], str(json_path))
            self.assertTrue(build_result.metadata["exact_schedule_reused"])
            self.assertEqual(build_result.metadata["tuning_time_sec"], 0.0)

    def _run_policy(self, policy: str) -> dict:
        captured = {}
        best = {
            "candidate": {
                "candidate_id": "g000_c000_seed",
                "path": "generated/search_spaces/cuda_matmul.py",
            },
            "fitness": {"score": 1.0, "reason": "correct"},
            "result": {
                "scheduled_module_path": "search/scheduled_module.txt",
                "scheduled_module_json_path": "search/scheduled_module.json",
                "metaschedule_work_dir": "search",
                "metaschedule_database_tuning_record": "search/database_tuning_record.json",
                "metaschedule_database_workload": "search/database_workload.json",
                "used_fallback_schedule": False,
                "compile_passed": True,
                "correctness_passed": True,
            },
        }

        def fake_run_matmul_experiment(*, strategy, strategy_config, extra_metadata, **_kwargs):
            captured["strategy_name"] = strategy.name
            captured["strategy_config"] = strategy_config
            captured["extra_metadata"] = extra_metadata
            return {}

        config = MethodRunConfig(
            M=16,
            N=16,
            K=16,
            target_name="cuda",
            output_dir=Path("results"),
            generated_search_space_path=Path("generated/search_spaces/cuda_matmul.py"),
            level2_final_evaluation_policy=policy,
        )
        with patch("src.eval.method.run_matmul_experiment", side_effect=fake_run_matmul_experiment):
            _run_evolved_best(
                config=config,
                method="level2-search",
                strategy_name="generated-search-space",
                run_dir=Path("run"),
                history=[best],
                best=best,
                evolution_time_sec=1.0,
            )
        return captured


if __name__ == "__main__":
    unittest.main()
