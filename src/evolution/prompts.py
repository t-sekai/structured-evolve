"""Prompts for Level-1 schedule evolution."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent, indent
from typing import Any, Mapping


MAX_ERROR_CHARS = 240
MAX_SCHEDULE_SUMMARY_CHARS = 1200
MAX_SCHEDULE_SUMMARY_LINES = 24


SYSTEM_PROMPT = dedent(
    """
    You are optimizing Apache TVM TensorIR schedule code for ML kernels.
    Return only a complete Python file. Do not include Markdown
    fences, commentary, explanations, XML tags, or reasoning. Keep the file
    concise and avoid docstrings.
    """
).strip()


def mutation_prompt(
    *,
    parent_code: str,
    target_name: str,
    M: int,
    N: int,
    K: int,
    generation: int,
    candidate_index: int,
    parent_feedback: str = "",
    workload_context: str | None = None,
    primary_block_name: str = "C",
) -> str:
    """Build a prompt asking the model to mutate one schedule candidate."""
    feedback_block = _feedback_block(parent_feedback)
    # The fallback is only for legacy matmul callers; normal runs pass the
    # workload-specific prompt_context from Workload.
    workload_context = workload_context or (
        f"matmul shape M={M}, N={N}, K={K}. Preserve numerical correctness for C = A @ B."
    )
    return dedent(
        f"""
        Mutate this TVM schedule candidate for {workload_context}
        Target: {target_name}.

        Requirements:
        - Return a complete Python file.
        - Do not include reasoning, XML tags, Markdown, prose, or docstrings.
        - Define exactly this callable:
          def apply_schedule(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
        - The function must return a tvm.IRModule.
        - Prefer simple, valid TVM schedule transformations.
        - This TVM build exposes tvm.s_tir.Schedule, not tvm.tir.Schedule.
        - Do not write `from tvm import tir` or `import tvm.tir`.
        - Use `sch = tvm.s_tir.Schedule(ir_module)` for schedule mutations.
        - The primary compute block is named "{primary_block_name}"; use
          `sch.get_sblock("{primary_block_name}", func_name="main")`.
        - Use `sch.split(loop, factors=[None, factor])`, not
          `sch.split(loop, factor=factor)`.
        - Return `sch.mod`, not `sch.mod()`.
        - For llvm, apply at most one `sch.parallel(...)` and at most one
          `sch.vectorize(...)`.
        - Do not pass a `factor=` argument to `sch.vectorize(...)`.
        - Wrap schedule transformations in `try`/`except` and return the
          original `ir_module` if a transformation fails.
        - If unsure, make a conservative mutation rather than invalid code.
        - The code may handle llvm and cuda differently.
        - For cuda, bind all output spatial loops under blockIdx/threadIdx.
          Leaving spatial loops unbound causes TVM memory verification failures.
        - For padded Conv2D, inline the data_pad block before binding conv loops.
        - Do not import project-local modules.
        - Do not read or write files.

        This is generation {generation}, candidate {candidate_index}.

        {feedback_block}

        Parent candidate:
        {parent_code}
        """
    ).strip()


def search_space_mutation_prompt(
    *,
    parent_code: str,
    target_name: str,
    M: int,
    N: int,
    K: int,
    generation: int,
    candidate_index: int,
    parent_feedback: str = "",
    workload_context: str | None = None,
    primary_block_name: str = "C",
) -> str:
    """Build a prompt asking the model to mutate one search-space candidate."""
    feedback_block = _feedback_block(parent_feedback)
    # The fallback is only for legacy matmul callers; normal runs pass the
    # workload-specific prompt_context from Workload.
    workload_context = workload_context or (
        f"matmul shape M={M}, N={N}, K={K}. Preserve numerical correctness for C = A @ B."
    )
    return dedent(
        f"""
        Mutate this TVM MetaSchedule search-space candidate for {workload_context}
        Target: {target_name}.

        Requirements:
        - Return a complete Python file.
        - Do not include reasoning, XML tags, Markdown, prose, or docstrings.
        - Define exactly this callable:
          def generate_design_space(sch: tvm.s_tir.Schedule):
        - The function must return a list of tvm.s_tir.Schedule objects.
        - This TVM build exposes tvm.s_tir.Schedule, not tvm.tir.Schedule.
        - Do not write `from tvm import tir` or `import tvm.tir`.
        - Use `sch.copy()` before mutating alternative schedules.
        - The primary compute block is named "{primary_block_name}"; use
          get_sblock("{primary_block_name}", func_name="main").
        - Prefer 2-4 conservative design-space variants.
        - If a transformation may fail, catch the exception and skip that variant.
        - For cuda, every variant must bind all output spatial loops under
          blockIdx/threadIdx. Leaving spatial loops unbound causes TVM memory
          verification failures.
        - For padded Conv2D, inline the data_pad block before binding conv loops.
        - Do not import project-local modules.
        - Do not read or write files.

        This is generation {generation}, candidate {candidate_index}.

        {feedback_block}

        Parent candidate:
        {parent_code}
        """
    ).strip()


def evaluator_feedback(
    parent_row: Mapping[str, Any],
    *,
    include_level2_artifacts: bool = False,
) -> str:
    """Return a compact prompt-facing summary for one evaluated parent."""
    candidate = parent_row.get("candidate", {})
    result = parent_row.get("result", {})
    fitness = parent_row.get("fitness", {})
    lines = [
        f"- candidate_id: {_display(candidate.get('candidate_id'))}",
        f"- generation_rank: {_display(parent_row.get('generation_rank'))}",
        f"- compile_passed: {_display(result.get('compile_passed'))}",
        f"- correctness_passed: {_display(result.get('correctness_passed'))}",
        f"- latency_ms_mean: {_display(result.get('latency_ms_mean'))}",
        f"- latency_ms_std: {_display(result.get('latency_ms_std'))}",
        f"- fitness_score: {_display(fitness.get('score'))}",
        f"- error: {_concise_error(result)}",
    ]
    selection_source_role = parent_row.get("selection_source_role")
    if selection_source_role is not None:
        lines.insert(2, f"- selection_source_role: {_display(selection_source_role)}")
    if any(
        key in result
        for key in ("cascade_rejected", "rejection_stage", "rejection_reason")
    ):
        lines.extend(
            [
                f"- cascade_rejected: {_display(result.get('cascade_rejected'))}",
                f"- rejection_stage: {_display(result.get('rejection_stage'))}",
                f"- rejection_reason: {_display(result.get('rejection_reason'))}",
            ]
        )
    if include_level2_artifacts:
        lines.extend(
            [
                f"- metaschedule_work_dir: {_display(result.get('metaschedule_work_dir'))}",
                (
                    "- metaschedule_database_tuning_record: "
                    f"{_display(result.get('metaschedule_database_tuning_record'))}"
                ),
                (
                    "- metaschedule_database_workload: "
                    f"{_display(result.get('metaschedule_database_workload'))}"
                ),
                f"- scheduled_module_path: {_display(result.get('scheduled_module_path'))}",
                (
                    "- scheduled_module_json_path: "
                    f"{_display(result.get('scheduled_module_json_path'))}"
                ),
                f"- used_fallback_schedule: {_display(result.get('used_fallback_schedule'))}",
                "- selected_schedule_summary:",
                indent(_selected_schedule_summary(result.get("scheduled_module_path")), "  "),
            ]
        )
    return "\n".join(lines)


def _feedback_block(parent_feedback: str) -> str:
    if not parent_feedback:
        return ""
    return "Parent evaluator feedback:\n" + parent_feedback


def _selected_schedule_summary(path_value: Any) -> str:
    if not path_value:
        return "(unavailable)"
    path = Path(str(path_value))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as err:
        return f"(unavailable: {type(err).__name__})"

    summary = "\n".join(lines[:MAX_SCHEDULE_SUMMARY_LINES])
    if len(lines) > MAX_SCHEDULE_SUMMARY_LINES:
        summary += "\n..."
    return _truncate(summary, MAX_SCHEDULE_SUMMARY_CHARS)


def _concise_error(result: Mapping[str, Any]) -> str:
    error_type = str(result.get("error_type") or "").strip()
    error_message = " ".join(str(result.get("error_message") or "").split())
    if not error_type and not error_message:
        return "(none)"
    if error_type and error_message:
        return _truncate(f"{error_type}: {error_message}", MAX_ERROR_CHARS)
    return _truncate(error_type or error_message, MAX_ERROR_CHARS)


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _display(value: Any) -> str:
    if value is None or value == "":
        return "(unavailable)"
    return str(value)


def strip_code_fences(text: str) -> str:
    """Remove common wrapper text from an LLM response."""
    stripped = text.strip()
    if "</reasoning>" in stripped:
        stripped = stripped.split("</reasoning>", 1)[1].strip()
    if "<reasoning>" in stripped:
        stripped = stripped.split("<reasoning>", 1)[0].strip()

    stripped = _strip_markdown_fence(stripped)
    code_starts = [
        stripped.find("from __future__"),
        stripped.find("import tvm"),
        stripped.find("def apply_schedule"),
    ]
    code_starts = [index for index in code_starts if index >= 0]
    if code_starts:
        stripped = stripped[min(code_starts) :].strip()

    return _strip_markdown_fence(stripped)


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip() + "\n"
