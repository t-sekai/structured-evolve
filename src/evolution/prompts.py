"""Prompts for Level-1 schedule evolution."""

from __future__ import annotations

import json
import re
from pathlib import Path
from textwrap import dedent, indent
from typing import Any, Mapping


MAX_ERROR_CHARS = 240
MAX_SURVIVOR_CHARS = 460
MAX_SURVIVOR_DETAIL_CHARS = 220
MAX_SCHEDULE_SUMMARY_CHARS = 1800
MAX_SCHEDULE_SUMMARY_LINES = 36
MAX_VARIANT_GUESS_CHARS = 900
MAX_TUNING_RECORD_LINES = 128


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
    survivor_feedback: str = "",
    workload_context: str | None = None,
    primary_block_name: str = "C",
    prompt_style: str = "standard",
) -> str:
    """Build a prompt asking the model to mutate one schedule candidate."""
    survivor_block = _survivor_block(survivor_feedback)
    feedback_block = _feedback_block(parent_feedback)
    creative_block = _creative_block(prompt_style, level="schedule")
    target_requirements = indent(
        _schedule_target_requirements(target_name, primary_block_name),
        "        ",
    )
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
        - Prefer ambitious but legal TVM schedule transformations over tiny
          no-op mutations.
        - This TVM build exposes tvm.s_tir.Schedule, not tvm.tir.Schedule.
        - Do not write `from tvm import tir` or `import tvm.tir`.
        - Use `sch = tvm.s_tir.Schedule(ir_module)` for schedule mutations.
        - The primary compute block is named "{primary_block_name}"; use
          `sch.get_sblock("{primary_block_name}", func_name="main")`.
        - Use `sch.split(loop, factors=[None, factor])`, not
          `sch.split(loop, factor=factor)`.
        - Return `sch.mod`, not `sch.mod()`.
        - Do not pass a `factor=` argument to `sch.vectorize(...)`.
        - Use bounded `try`/`except` around risky transform groups and return
          the original `ir_module` only if both the ambitious path and a simpler
          fallback fail.
        - Use the survivor summary as guidance, but mutate only the parent
          candidate whose full code is provided below.
        - Do not import project-local modules.
        - Do not read or write files.

        Target-specific requirements:
{target_requirements}

        {creative_block}

        This is generation {generation}, candidate {candidate_index}.

        {survivor_block}

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
    survivor_feedback: str = "",
    workload_context: str | None = None,
    primary_block_name: str = "C",
    prompt_style: str = "standard",
) -> str:
    """Build a prompt asking the model to mutate one search-space candidate."""
    survivor_block = _survivor_block(survivor_feedback)
    feedback_block = _feedback_block(parent_feedback)
    creative_block = _creative_block(prompt_style, level="search_space")
    target_requirements = indent(
        _search_space_target_requirements(target_name, primary_block_name),
        "        ",
    )
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
        - Prefer 3-6 ambitious but legal design-space variants plus one simple
          fallback variant when useful.
        - Label each variant with a short `# Variant N: ...` comment that names
          the tile sizes and key transforms.
        - Use the survivor summary and parent feedback to preserve transforms
          that appear in selected schedules, then add alternatives that explore
          meaningful tile/cache/binding choices.
        - If a transformation may fail, catch the exception and skip that variant.
        - Do not import project-local modules.
        - Do not read or write files.

        Target-specific requirements:
{target_requirements}

        {creative_block}

        This is generation {generation}, candidate {candidate_index}.

        {survivor_block}

        {feedback_block}

        Parent candidate:
        {parent_code}
        """
    ).strip()


def _schedule_target_requirements(target_name: str, primary_block_name: str) -> str:
    """Return target-specific prompt constraints for direct schedule candidates."""
    if target_name == "cuda":
        return dedent(
            f"""
            - Write a CUDA-only candidate. Start with
              `if target_name != "cuda": return ir_module`.
            - Do not include a CPU-only `else` schedule branch.
            - Do not use CPU-oriented `sch.parallel(...)` in the CUDA path.
            - Bind all output spatial loops under `blockIdx`/`threadIdx`;
              unbound CUDA spatial loops cause TVM memory verification failures.
            - For matmul-like reductions, strongly consider the full CUDA
              hierarchy: `cache_write(..., "local")` for the output accumulator,
              `cache_read(..., "shared")` for input operands, `compute_at` /
              `reverse_compute_at`, cooperative fetch loops, vectorized shared
              loads, and reduction tiling/unrolling.
            - For padded Conv2D, inline the `data_pad` block before binding the
              `{primary_block_name}` loops.
            - Keep CUDA thread extents plausible; avoid binding a reduction loop
              directly to a huge `threadIdx` extent.
            """
        ).strip()
    if target_name == "llvm":
        return dedent(
            """
            - Write an LLVM-only candidate. Start with
              `if target_name != "llvm": return ir_module`.
            - Do not include GPU-specific branches, thread-binding primitives,
              virtual threads, or GPU memory scopes.
            - Use CPU-oriented loop tiling, `sch.parallel(...)`, `sch.vectorize(...)`,
              and modest unrolling. Apply at most one `sch.parallel(...)` and at
              most one `sch.vectorize(...)`.
            """
        ).strip()
    return "- Preserve the requested target only; do not add unrelated target branches."


def _search_space_target_requirements(target_name: str, primary_block_name: str) -> str:
    """Return target-specific prompt constraints for generated search spaces."""
    if target_name == "cuda":
        return dedent(
            f"""
            - Generate CUDA-only design-space variants. Do not include CPU-only
              variants or CPU `parallel` fallbacks.
            - Every variant must bind output spatial loops under
              `blockIdx`/`threadIdx`.
            - For matmul-like reductions, include variants with local output
              accumulation, shared input reads, cooperative fetch/vectorized
              shared loads, reduction tiling, and different block/thread tile
              shapes.
            - For padded Conv2D, inline the `data_pad` block before binding the
              `{primary_block_name}` loops.
            - Include one simple split/bind fallback variant after the ambitious
              CUDA variants if it helps robustness.
            """
        ).strip()
    if target_name == "llvm":
        return dedent(
            """
            - Generate LLVM-only design-space variants. Do not use GPU-specific
              thread-binding primitives, virtual threads, or GPU memory scopes.
            - Explore CPU loop tiling, one parallel loop, one vectorized inner
              loop, and modest unrolling.
            """
        ).strip()
    return "- Preserve the requested target only; do not add unrelated target variants."


def _creative_block(prompt_style: str, *, level: str) -> str:
    if prompt_style != "creative":
        return ""
    if level == "search_space":
        body = """
        Creative exploration mode:
        - Do not merely retune constants in the parent. Propose at least one
          substantially different legal CUDA schedule family.
        - Include variants that differ in cache hierarchy, tile shape,
          per-thread work, reduction tiling, cooperative fetch/vectorization, or
          write-back placement.
        - Keep every ambitious variant wrapped in a tight `try`/`except`, and
          include a simple fallback so invalid ideas do not poison the run.
        - Prefer compact helper functions if they make several advanced
          variants easier to express correctly.
        """
    else:
        body = """
        Creative exploration mode:
        - Do not merely retune constants in the parent. Try one substantial,
          legal schedule-structure change before falling back.
        - Consider moving cache placement, changing block/thread hierarchy,
          adding local accumulation, adding shared reads/cooperative fetch, or
          changing reduction tiling and unroll structure.
        - Use a staged implementation: ambitious path first, simpler CUDA
          fallback second, original `ir_module` only as the final fallback.
        - Stay within TVM schedule APIs already shown in the parent/seed; avoid
          invented APIs and target-mismatched code.
        """
    return "Prompt style: creative.\n" + dedent(body).strip()


def survivor_summary(
    survivor_rows: list[Mapping[str, Any]],
    *,
    include_level2_artifacts: bool = False,
) -> str:
    """Return a concise prompt-facing summary of all survivor candidates."""
    if not survivor_rows:
        return ""
    lines = [
        "Top candidates selected as this generation's survivor pool. "
        "Use these as compact design guidance; full code is shown only for the assigned parent."
    ]
    for fallback_rank, row in enumerate(survivor_rows, start=1):
        lines.append(
            _survivor_summary_line(
                row,
                fallback_rank=fallback_rank,
                include_level2_artifacts=include_level2_artifacts,
            )
        )
    return "\n".join(lines)


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
    if candidate.get("origin") is not None:
        lines.insert(2, f"- candidate_origin: {_display(candidate.get('origin'))}")
    if candidate.get("parent_id") is not None:
        lines.insert(2, f"- parent_id: {_display(candidate.get('parent_id'))}")
    if candidate.get("inspiration_id") is not None:
        lines.insert(2, f"- inspiration_id: {_display(candidate.get('inspiration_id'))}")
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
        schedule_features = _schedule_feature_info(result.get("scheduled_module_path"))
        lines.extend(
            [
                f"- preflight_variant_count: {_display(result.get('preflight_variant_count'))}",
                (
                    "- preflight_first_passing_variant_index: "
                    f"{_display(result.get('preflight_variant_index'))}"
                ),
                (
                    "- preflight_note: first passing variant only; the selected "
                    "MetaSchedule winner is summarized below"
                ),
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
                "- tuning_record_summary:",
                indent(_tuning_record_summary(result.get("metaschedule_database_tuning_record")), "  "),
                "- selected_schedule_features:",
                indent(schedule_features.summary, "  "),
                "- selected_parent_variant_guess:",
                indent(
                    _parent_variant_guess(
                        candidate.get("path"),
                        selected_features=schedule_features.features,
                    ),
                    "  ",
                ),
                "- selected_schedule_summary:",
                indent(_selected_schedule_summary(result.get("scheduled_module_path")), "  "),
            ]
        )
    return "\n".join(lines)


def _survivor_summary_line(
    row: Mapping[str, Any],
    *,
    fallback_rank: int,
    include_level2_artifacts: bool,
) -> str:
    candidate = row.get("candidate", {})
    result = row.get("result", {})
    fitness = row.get("fitness", {})
    rank = row.get("generation_rank") or fallback_rank
    status = _candidate_status(result)
    pieces = [
        f"rank {rank}",
        f"id={_display(candidate.get('candidate_id'))}",
        f"score={_compact_value(fitness.get('score'))}",
        f"latency_ms={_compact_value(result.get('latency_ms_mean'))}",
        f"status={status}",
    ]
    if candidate.get("origin"):
        pieces.append(f"origin={candidate.get('origin')}")
    if candidate.get("parent_id"):
        pieces.append(f"parent={candidate.get('parent_id')}")
    details: list[str] = []
    error = _concise_error(result)
    if error != "(none)":
        details.append(f"error={error}")
    if include_level2_artifacts:
        schedule_features = _schedule_feature_info(result.get("scheduled_module_path"))
        details.append(f"selected={_compact_schedule_features(schedule_features.features)}")
        record = _compact_tuning_record(result.get("metaschedule_database_tuning_record"))
        if record:
            details.append(record)
        variant = _compact_parent_variant_guess(
            candidate.get("path"),
            selected_features=schedule_features.features,
        )
        if variant:
            details.append(variant)
    if details:
        pieces.append("; ".join(_truncate(detail, MAX_SURVIVOR_DETAIL_CHARS) for detail in details))
    return "- " + _truncate(" | ".join(pieces), MAX_SURVIVOR_CHARS)


def _candidate_status(result: Mapping[str, Any]) -> str:
    if result.get("compile_passed") and result.get("correctness_passed"):
        return "valid"
    if result.get("compile_passed"):
        return "compile-only"
    if result.get("cascade_rejected"):
        return f"rejected:{_display(result.get('rejection_stage'))}"
    return "invalid"


def _compact_value(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return str(value)


def _compact_schedule_features(features: Mapping[str, Any]) -> str:
    if not features:
        return "unavailable"
    pieces = []
    tiles = features.get("tiles") or {}
    if tiles:
        pieces.append(f"tiles({_format_tiles(tiles)})")
    if features.get("has_fused_parallel"):
        pieces.append("fused_parallel")
    if features.get("has_parallel"):
        pieces.append("parallel")
    if features.get("has_vectorize"):
        pieces.append("vectorize")
    if features.get("has_unroll"):
        pieces.append("unroll")
    return ", ".join(pieces) if pieces else "no inferred schedule features"


def _compact_tuning_record(path_value: Any) -> str:
    summary = _tuning_record_summary(path_value)
    if summary.startswith("(unavailable"):
        return ""
    for line in summary.splitlines():
        stripped = line.removeprefix("- ").strip()
        if stripped.startswith("best_record_trace:"):
            return "best_trace=" + stripped.removeprefix("best_record_trace:").strip()
    return ""


def _compact_parent_variant_guess(
    path_value: Any,
    *,
    selected_features: Mapping[str, Any],
) -> str:
    guess = _parent_variant_guess(path_value, selected_features=selected_features)
    if guess.startswith("(unavailable"):
        return ""
    useful_lines = []
    for line in guess.splitlines():
        stripped = line.removeprefix("- ").strip()
        if stripped.startswith("best_match:"):
            useful_lines.append("variant=" + stripped.removeprefix("best_match:").strip())
        elif stripped.startswith("matched_features:"):
            useful_lines.append("matches=" + stripped.removeprefix("matched_features:").strip())
    if not useful_lines:
        return ""
    return "; ".join(useful_lines)


def _survivor_block(survivor_feedback: str) -> str:
    if not survivor_feedback:
        return ""
    return "Survivor summary:\n" + survivor_feedback


def _feedback_block(parent_feedback: str) -> str:
    if not parent_feedback:
        return ""
    return "Assigned parent evaluator feedback:\n" + parent_feedback



def _selected_schedule_summary(path_value: Any) -> str:
    text = _read_text_path(path_value)
    if text is None:
        return "(unavailable)"
    lines = text.splitlines()
    summary = "\n".join(lines[:MAX_SCHEDULE_SUMMARY_LINES])
    if len(lines) > MAX_SCHEDULE_SUMMARY_LINES:
        summary += "\n..."
    return _truncate(summary, MAX_SCHEDULE_SUMMARY_CHARS)


class _ScheduleFeatureInfo:
    def __init__(self, *, features: dict[str, Any], summary: str):
        self.features = features
        self.summary = summary


def _schedule_feature_info(path_value: Any) -> _ScheduleFeatureInfo:
    text = _read_text_path(path_value)
    if text is None:
        return _ScheduleFeatureInfo(features={}, summary="(unavailable)")
    features = _extract_schedule_features(text)
    return _ScheduleFeatureInfo(
        features=features,
        summary=_format_schedule_features(features),
    )


def _extract_schedule_features(text: str) -> dict[str, Any]:
    tiles: dict[str, int] = {}
    for axis in ("y", "x", "i", "j"):
        match = re.search(
            rf"v_{axis}\s*=\s*T\.axis\.spatial\([^,]+,\s*[^\n]*?{axis}_0\s*\*\s*(\d+)",
            text,
        )
        if match:
            tiles[axis] = int(match.group(1))

    return {
        "tiles": tiles,
        "parallel_loops": _loop_kinds(text, "parallel"),
        "vectorized_loops": _loop_kinds(text, "vectorized"),
        "unrolled_loops": _loop_kinds(text, "unroll"),
        "has_parallel": "T.parallel(" in text,
        "has_vectorize": "T.vectorized(" in text,
        "has_unroll": "T.unroll(" in text,
        "has_fused_parallel": bool(re.search(r"for\s+\w*fused\w*\s+in\s+T\.parallel", text)),
    }


def _loop_kinds(text: str, kind: str) -> list[str]:
    pattern = rf"for\s+([A-Za-z0-9_, ]+)\s+in\s+T\.{kind}\(([^)]*)\)"
    return [
        f"{' '.join(loop_names.split())} extent={extent.strip()}"
        for loop_names, extent in re.findall(pattern, text)
    ]


def _format_schedule_features(features: Mapping[str, Any]) -> str:
    if not features:
        return "(unavailable)"
    tiles = features.get("tiles") or {}
    lines = [
        f"- inferred_tiles: {_format_tiles(tiles)}",
        f"- parallel_loops: {_format_list(features.get('parallel_loops'))}",
        f"- vectorized_loops: {_format_list(features.get('vectorized_loops'))}",
        f"- unrolled_loops: {_format_list(features.get('unrolled_loops'))}",
        f"- fused_parallel: {_display(features.get('has_fused_parallel'))}",
    ]
    return "\n".join(lines)


def _parent_variant_guess(path_value: Any, *, selected_features: Mapping[str, Any]) -> str:
    text = _read_text_path(path_value)
    if text is None:
        return "(unavailable)"
    variants = _extract_variant_blocks(text)
    if not variants:
        return "(unavailable: parent candidate has no `# Variant N:` comments to match)"

    scored = [
        _score_variant_match(variant, selected_features=selected_features)
        for variant in variants
    ]
    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0]
    if best["score"] <= 0:
        return "(unavailable: no parent variant feature matched the selected schedule)"

    variant = best["variant"]
    lines = [
        f"- best_match: Variant {variant['number']}: {variant['label']}",
        f"- match_score: {best['score']}",
        f"- matched_features: {_format_list(best['matches'])}",
        f"- variant_features: {_format_variant_features(best['features'])}",
    ]
    return _truncate("\n".join(lines), MAX_VARIANT_GUESS_CHARS)


def _extract_variant_blocks(text: str) -> list[dict[str, str]]:
    matches = list(re.finditer(r"(?m)^\s*#\s*Variant\s+(\d+)\s*:\s*(.*)$", text))
    variants: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        variants.append(
            {
                "number": match.group(1),
                "label": match.group(2).strip(),
                "block": text[start:end],
            }
        )
    return variants


def _score_variant_match(
    variant: Mapping[str, str],
    *,
    selected_features: Mapping[str, Any],
) -> dict[str, Any]:
    features = _extract_variant_features(variant.get("block", ""))
    selected_tiles = selected_features.get("tiles") or {}
    matches: list[str] = []
    score = 0
    for axis in ("y", "x", "i", "j"):
        expected = selected_tiles.get(axis)
        actual = features.get(f"tile_{axis}")
        if expected is not None and actual == expected:
            score += 2
            matches.append(f"tile_{axis}={actual}")
    for name in ("has_parallel", "has_vectorize", "has_unroll", "has_fuse"):
        selected_value = bool(selected_features.get(name))
        if name == "has_fuse":
            selected_value = bool(selected_features.get("has_fused_parallel"))
        actual_value = bool(features.get(name))
        if selected_value and actual_value:
            score += 1
            matches.append(name.replace("has_", ""))
    return {
        "variant": variant,
        "features": features,
        "matches": matches,
        "score": score,
    }


def _extract_variant_features(text: str) -> dict[str, Any]:
    features: dict[str, Any] = {
        "has_parallel": ".parallel(" in text,
        "has_vectorize": ".vectorize(" in text,
        "has_unroll": ".unroll(" in text,
        "has_fuse": ".fuse(" in text,
    }
    for axis in ("y", "x", "i", "j"):
        match = re.search(
            rf"\.split\(\s*{axis}\w*\s*,\s*factors=\[None,\s*(\d+)\]",
            text,
        )
        if match:
            features[f"tile_{axis}"] = int(match.group(1))
    return features


def _format_variant_features(features: Mapping[str, Any]) -> str:
    pieces = []
    for key in ("tile_y", "tile_x", "tile_i", "tile_j"):
        if features.get(key) is not None:
            pieces.append(f"{key}={features[key]}")
    for key in ("has_parallel", "has_vectorize", "has_unroll", "has_fuse"):
        if features.get(key):
            pieces.append(key.replace("has_", ""))
    return ", ".join(pieces) if pieces else "(unavailable)"


def _tuning_record_summary(path_value: Any) -> str:
    text = _read_text_path(path_value)
    if text is None:
        return "(unavailable)"
    best: dict[str, Any] | None = None
    count = 0
    for line in text.splitlines()[:MAX_TUNING_RECORD_LINES]:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        count += 1
        latency = _record_latency(record)
        trace_summary = _record_trace_summary(record)
        if latency is None:
            continue
        if best is None or latency < best["latency"]:
            best = {"latency": latency, "trace_summary": trace_summary}
    if count == 0:
        return "(unavailable: no parseable tuning records)"
    lines = [f"- records_seen: {count}"]
    if best is not None:
        lines.extend(
            [
                f"- best_record_latency_sec: {best['latency']:.6g}",
                f"- best_record_trace: {best['trace_summary']}",
            ]
        )
    return "\n".join(lines)


def _record_latency(record: Any) -> float | None:
    try:
        values = record[1][1]
    except (IndexError, TypeError):
        return None
    if not isinstance(values, list) or not values:
        return None
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _record_trace_summary(record: Any) -> str:
    try:
        instructions = record[1][0][0]
    except (IndexError, TypeError):
        return "(unavailable)"
    if not isinstance(instructions, list):
        return "(unavailable)"
    ops: list[str] = []
    split_factors: list[str] = []
    for instruction in instructions:
        if not isinstance(instruction, list) or not instruction:
            continue
        op_name = str(instruction[0])
        ops.append(op_name)
        if op_name == "Split" and len(instruction) > 1:
            factors = instruction[1]
            if isinstance(factors, list):
                numeric = [
                    str(factor)
                    for factor in factors[1:]
                    if isinstance(factor, int)
                ]
                if numeric:
                    split_factors.append("x".join(numeric))
    pieces = []
    if split_factors:
        pieces.append(f"splits={','.join(split_factors)}")
    for op_name in ("Parallel", "Vectorize", "Unroll", "Fuse"):
        count = ops.count(op_name)
        if count:
            pieces.append(f"{op_name.lower()}={count}")
    return ", ".join(pieces) if pieces else "ops=" + "->".join(ops[:8])


def _format_tiles(tiles: Mapping[str, Any]) -> str:
    if not tiles:
        return "(none inferred)"
    return ", ".join(f"{axis}={value}" for axis, value in sorted(tiles.items()))


def _format_list(values: Any) -> str:
    if not values:
        return "(none)"
    if isinstance(values, (list, tuple)):
        return ", ".join(str(value) for value in values)
    return str(values)


def _read_text_path(path_value: Any) -> str | None:
    if not path_value:
        return None
    try:
        return Path(str(path_value)).read_text(encoding="utf-8")
    except OSError:
        return None


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
        stripped.find("def generate_design_space"),
        stripped.find("def create_space_generator"),
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
