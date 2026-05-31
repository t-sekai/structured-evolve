"""Focused validation for clear MetaSchedule no-winner reporting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tvm

from src.kernels.matmul_tir import create_matmul_ir_module
from src.strategies.base import StrategyBuildConfig
from src.strategies.metaschedule import MetaScheduleStrategy


class MetaScheduleRecoveryTest(unittest.TestCase):
    def test_no_winner_raises_clear_error_instead_of_dereferencing_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("src.strategies.metaschedule.tune_tir", return_value="db"),
                patch("src.strategies.metaschedule.compile_tir", return_value=None),
            ):
                with self.assertRaisesRegex(RuntimeError, "produced no valid schedule"):
                    MetaScheduleStrategy().build(
                        ir_module=create_matmul_ir_module(16, 16, 16),
                        target=tvm.target.Target("llvm"),
                        target_name="cuda",
                        config=StrategyBuildConfig(
                            work_dir=Path(temp_dir),
                            max_trials_global=2,
                            num_trials_per_iter=1,
                            num_tuning_cores=1,
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
