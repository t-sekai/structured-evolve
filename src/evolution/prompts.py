"""Prompts for Level-1 schedule evolution."""

from __future__ import annotations

from textwrap import dedent


SYSTEM_PROMPT = dedent(
    """
    You are optimizing Apache TVM TensorIR schedule code for matrix
    multiplication. Return only a complete Python file. Do not include Markdown
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
) -> str:
    """Build a prompt asking the model to mutate one schedule candidate."""
    return dedent(
        f"""
        Mutate this TVM schedule candidate for matmul shape
        M={M}, N={N}, K={K}, target={target_name}.

        Requirements:
        - Return a complete Python file.
        - Do not include reasoning, XML tags, Markdown, prose, or docstrings.
        - Define exactly this callable:
          def apply_schedule(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
        - The function must return a tvm.IRModule.
        - Preserve numerical correctness for C = A @ B.
        - Prefer simple, valid TVM schedule transformations.
        - This TVM build exposes tvm.s_tir.Schedule, not tvm.tir.Schedule.
        - Do not write `from tvm import tir` or `import tvm.tir`.
        - Use `sch = tvm.s_tir.Schedule(ir_module)` for schedule mutations.
        - The matmul compute block is named "C".
        - If unsure, make a conservative mutation rather than invalid code.
        - The code may handle llvm and cuda differently.
        - Do not import project-local modules.
        - Do not read or write files.

        This is generation {generation}, candidate {candidate_index}.

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
) -> str:
    """Build a prompt asking the model to mutate one search-space candidate."""
    return dedent(
        f"""
        Mutate this TVM MetaSchedule search-space candidate for matmul shape
        M={M}, N={N}, K={K}, target={target_name}.

        Requirements:
        - Return a complete Python file.
        - Do not include reasoning, XML tags, Markdown, prose, or docstrings.
        - Define exactly this callable:
          def generate_design_space(sch: tvm.s_tir.Schedule):
        - The function must return a list of tvm.s_tir.Schedule objects.
        - Preserve numerical correctness for C = A @ B.
        - This TVM build exposes tvm.s_tir.Schedule, not tvm.tir.Schedule.
        - Do not write `from tvm import tir` or `import tvm.tir`.
        - Use `sch.copy()` before mutating alternative schedules.
        - The matmul compute block is named "C"; use get_sblock("C", func_name="main").
        - Prefer 2-4 conservative design-space variants.
        - If a transformation may fail, catch the exception and skip that variant.
        - Do not import project-local modules.
        - Do not read or write files.

        This is generation {generation}, candidate {candidate_index}.

        Parent candidate:
        {parent_code}
        """
    ).strip()


def strip_code_fences(text: str) -> str:
    """Remove common wrapper text from an LLM response."""
    stripped = text.strip()
    if "</reasoning>" in stripped:
        stripped = stripped.split("</reasoning>", 1)[1].strip()
    if "<reasoning>" in stripped:
        stripped = stripped.split("<reasoning>", 1)[0].strip()

    code_starts = [
        stripped.find("from __future__"),
        stripped.find("import tvm"),
        stripped.find("def apply_schedule"),
    ]
    code_starts = [index for index in code_starts if index >= 0]
    if code_starts:
        stripped = stripped[min(code_starts) :].strip()

    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip() + "\n"
