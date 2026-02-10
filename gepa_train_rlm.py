"""
GEPA Training (Optional)
=======================

This script is intentionally separate from `rlm_vs_react_tool_calling.py` to
keep the demo clean.

Goal: demonstrate DSPy's GEPA optimizing the *RLM prompt/instructions* so the
RLM more reliably:
- enumerates all pairs/combos
- filters "no known interaction"/"no contraindication"
- returns a consistent structured assessment

Cost note: GEPA is an optimizer that uses additional LLM calls. The defaults in
this script are set to be a small "smoke run" budget. Increase gradually.
"""

from __future__ import annotations

import argparse
from itertools import combinations

import dspy

from rlm_vs_react_tool_calling import (
    TOOL_LOG,
    check_contraindication,
    check_drug_interaction,
    get_drug_class,
    _build_python_interpreter,
)


class TrainingRiskSig(dspy.Signature):
    """You are in a Python REPL (executed in a sandbox). Write Python code only.

You MUST use the provided tools (do not rely on pharmacology knowledge):
- check_drug_interaction(drug_a, drug_b)
- check_contraindication(drug, condition)

Hard requirements:
1. Enumerate ALL unique unordered drug pairs (N choose 2) and call check_drug_interaction on each.
2. Enumerate ALL drug x condition pairs (N x M) and call check_contraindication on each.
3. Filter out "No known interaction" and "No contraindication" entries.
4. Produce a structured dict with exactly:
   - interactions: list[{drug_a, drug_b, details}]
   - contraindications: list[{drug, condition, details}]
5. Call SUBMIT(risk_assessment) where risk_assessment is that dict.

Be explicit and systematic. Do not skip pairs.
"""

    medications: str = dspy.InputField()
    conditions: str = dspy.InputField()
    risk_assessment: str = dspy.OutputField()


def build_training_rlm_agent() -> dspy.Module:
    """Training-only RLM with stronger, explicit instructions for smaller models."""
    return dspy.RLM(
        signature=TrainingRiskSig,
        tools=[check_drug_interaction, check_contraindication, get_drug_class],
        max_iterations=20,
        max_llm_calls=20,
        interpreter=_build_python_interpreter(),
    )


def build_training_examples() -> list[dspy.Example]:
    # These are intentionally small and easy to run.
    return [
        dspy.Example(
            medications="warfarin, aspirin, lisinopril",
            conditions="heart_failure",
            risk_assessment=(
                "HIGH RISK: 1 major interaction (warfarin+aspirin: increased bleeding). "
                "1 moderate interaction (aspirin+lisinopril: reduced BP control). "
                "No contraindications for heart failure."
            ),
        ).with_inputs("medications", "conditions"),
        dspy.Example(
            medications="metformin, lisinopril, simvastatin",
            conditions="chronic_kidney_disease",
            risk_assessment=(
                "MODERATE RISK: 1 minor interaction (metformin+lisinopril: improved insulin sensitivity). "
                "1 contraindication: metformin contraindicated in CKD."
            ),
        ).with_inputs("medications", "conditions"),
        dspy.Example(
            medications="warfarin, amiodarone, simvastatin, aspirin, lisinopril, metformin, potassium",
            conditions="chronic_kidney_disease, heart_failure",
            risk_assessment=(
                "CRITICAL RISK: major interactions (warfarin+aspirin, warfarin+amiodarone, "
                "simvastatin+amiodarone); moderate interactions (aspirin+lisinopril, warfarin+simvastatin); "
                "contraindications: metformin and potassium in CKD."
            ),
        ).with_inputs("medications", "conditions"),
    ]


def _normalize_list(csv: str) -> list[str]:
    return [x.strip().lower() for x in csv.split(",") if x.strip()]


def tool_coverage_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> float:
    """Deterministic metric: did the agent *call tools exhaustively* for this input?

    This is the core "RLM vs ReAct" thesis: we care about enumeration completeness,
    not whether a narrative happens to include certain tokens.
    """
    meds = _normalize_list(gold.medications)
    conds = [c.strip().lower().replace(" ", "_") for c in gold.conditions.split(",") if c.strip()]

    expected_pairs = {tuple(sorted(p)) for p in combinations(meds, 2)}
    expected_dc = {(drug, cond) for drug in meds for cond in conds}

    snapshot = getattr(pred, "_tool_log_snapshot", None)
    calls = snapshot if isinstance(snapshot, list) else TOOL_LOG.calls

    seen_pairs = set()
    seen_dc = set()
    for c in calls:
        if c["tool"] == "check_drug_interaction":
            a = str(c["args"].get("drug_a", "")).strip().lower()
            b = str(c["args"].get("drug_b", "")).strip().lower()
            if a and b:
                seen_pairs.add(tuple(sorted((a, b))))
        elif c["tool"] == "check_contraindication":
            drug = str(c["args"].get("drug", "")).strip().lower()
            cond = str(c["args"].get("condition", "")).strip().lower().replace(" ", "_")
            if drug and cond:
                seen_dc.add((drug, cond))

    pair_cov = (len(seen_pairs & expected_pairs) / len(expected_pairs)) if expected_pairs else 1.0
    dc_cov = (len(seen_dc & expected_dc) / len(expected_dc)) if expected_dc else 1.0
    return (pair_cov + dc_cov) / 2.0


def semantic_f1_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> float:
    """LLM-based metric (more expensive)."""
    semantic_f1 = dspy.evaluate.SemanticF1()
    return float(semantic_f1(gold, pred))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optional GEPA optimization for the RLM agent.")
    p.add_argument(
        "--program-model",
        default="groq/openai/gpt-oss-20b",
        help="Model used by the program being optimized (smaller is cheaper).",
    )
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=0.0, help="Program LM temperature.")
    p.add_argument(
        "--reflection-model",
        default="groq/openai/gpt-oss-120b",
        help="Model used for GEPA reflection/instruction proposals (often stronger is better).",
    )
    p.add_argument(
        "--reflection-temperature",
        type=float,
        default=1.0,
        help="Higher temperature is often better for GEPA reflection.",
    )

    # DSPy GEPA (3.1.3) requires exactly one of: auto, max_full_evals, max_metric_calls.
    # We default to a small `max_metric_calls` budget if the user doesn't pick one.
    budget = p.add_mutually_exclusive_group()
    budget.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)
    budget.add_argument("--max-full-evals", type=int, default=None, help="GEPA budget knob (mutually exclusive with --auto and --max-metric-calls).")
    budget.add_argument("--max-metric-calls", type=int, default=None, help="GEPA budget knob (mutually exclusive with --auto and --max-full-evals).")

    p.add_argument("--metric", choices=["coverage", "semantic_f1"], default="coverage")
    p.add_argument("--out", default="gepa_rlm_state.json")
    p.add_argument("--save-program", action="store_true", help="Save full compiled program (larger).")
    p.add_argument("--dry-run", action="store_true", help="Print planned budget and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    trainset = build_training_examples()
    # Keep it small: first 2 train, last 1 val (same split as the earlier inline version).
    train, val = trainset[:2], trainset[2:]

    # Default budget: small metric-call budget.
    if args.auto is None and args.max_full_evals is None and args.max_metric_calls is None:
        args.max_metric_calls = 6

    if args.dry_run:
        print("GEPA dry-run (no LLM calls).")
        print(f"  program_model: {args.program_model}")
        print(f"  reflection_model: {args.reflection_model}")
        print(f"  auto: {args.auto}")
        print(f"  max_full_evals: {args.max_full_evals}")
        print(f"  max_metric_calls: {args.max_metric_calls}")
        print(f"  metric: {args.metric}")
        print(f"  train examples: {len(train)}")
        print(f"  val examples: {len(val)}")
        print(f"  output: {args.out} (save_program={args.save_program})")
        return

    program_lm = dspy.LM(args.program_model, max_tokens=args.max_tokens, temperature=args.temperature)
    reflection_lm = dspy.LM(
        args.reflection_model,
        max_tokens=args.max_tokens,
        temperature=args.reflection_temperature,
    )
    # Default LM used by the program unless overridden with settings.context.
    dspy.configure(lm=program_lm)

    metric_fn = tool_coverage_metric if args.metric == "coverage" else semantic_f1_metric

    optimizer = dspy.GEPA(
        metric=metric_fn,
        auto=args.auto,
        max_full_evals=args.max_full_evals,
        max_metric_calls=args.max_metric_calls,
        reflection_lm=reflection_lm,
        num_threads=1,  # avoid global TOOL_LOG interference
    )

    class InstrumentedRLM(dspy.Module):
        def __init__(self, lm: dspy.LM):
            super().__init__()
            self.agent = build_training_rlm_agent()
            self.lm = lm

        def forward(self, medications: str, conditions: str):
            TOOL_LOG.reset()
            with dspy.settings.context(lm=self.lm):
                pred = self.agent(medications=medications, conditions=conditions)
            # Snapshot tool calls so metrics remain correct even if other runs happen later.
            try:
                setattr(pred, "_tool_log_snapshot", list(TOOL_LOG.calls))
            except Exception:
                pass
            return pred

    program = InstrumentedRLM(program_lm)
    # NOTE: DSPy GEPA (3.1.3) does not support a separate second program in compile().
    # If you want a stronger model to guide optimization, provide it as `--reflection-model`.
    optimized = optimizer.compile(program, trainset=train, valset=val)

    if args.out:
        optimized.save(args.out, save_program=args.save_program)
        print(f"Saved optimized agent to {args.out}")

    # Quick sanity run on the val example to show the optimized agent still executes.
    ex = val[0]
    pred = optimized(medications=ex.medications, conditions=ex.conditions)
    print("\nSample output (val example):")
    print(getattr(pred, "risk_assessment", pred))
    print("\nTool-call coverage (val example):")
    meds = _normalize_list(ex.medications)
    conds = [c.strip().lower().replace(" ", "_") for c in ex.conditions.split(",") if c.strip()]
    expected_pairs = len(list(combinations(meds, 2)))
    expected_dc = len(meds) * len(conds)
    seen_pairs = sum(1 for c in TOOL_LOG.calls if c["tool"] == "check_drug_interaction")
    seen_dc = sum(1 for c in TOOL_LOG.calls if c["tool"] == "check_contraindication")
    print(f"  check_drug_interaction calls: {seen_pairs} (expected {expected_pairs})")
    print(f"  check_contraindication calls: {seen_dc} (expected {expected_dc})")


if __name__ == "__main__":
    main()
