"""
RLM vs ReAct for Tool Calling: A Hypothetical Comparison
=========================================================

Thesis: RLM's value for tool calling is NOT about long context.
It's about PROGRAMMATIC ORCHESTRATION.

ReAct reasons one step at a time:
    think -> pick tool -> call tool -> observe -> repeat

RLM writes CODE that can:
    - Loop over tool calls systematically
    - Branch based on intermediate results
    - Aggregate and cross-reference across multiple tool outputs
    - Implement retry/fallback logic
    - Build data structures from tool results
    - Perform calculations on tool outputs in-situ

This matters most when the task requires COMPOSITIONAL tool use --
calling the same tool many times with different arguments, cross-referencing
results, or performing non-trivial logic on tool outputs.

Scenario: Drug Interaction Research Agent
------------------------------------------
A pharmacist needs to check whether a patient's medication list has
dangerous interactions. This requires:
  1. Querying a drug interaction API for every PAIR of drugs
  2. Checking severity levels
  3. Cross-referencing with patient conditions
  4. Producing a structured risk assessment

For N drugs, there are N*(N-1)/2 pairs to check.
A 10-drug regimen = 45 pairwise checks.

ReAct will do this one pair at a time, burning 45+ reasoning steps.
RLM can write a loop and do it in a handful of iterations.
"""

import argparse
import dspy
import os
import shutil
import statistics
import textwrap
import time
from itertools import combinations


# ---------------------------------------------------------------------------
# Instrumentation: tool call logger
# ---------------------------------------------------------------------------


class ToolCallLog:
    """Records every tool invocation for post-hoc comparison."""

    def __init__(self):
        self.calls: list[dict] = []
        self.start_time: float | None = None

    def reset(self):
        self.calls = []
        self.start_time = time.monotonic()

    def record(self, tool_name: str, args: dict, result: str):
        elapsed = time.monotonic() - self.start_time if self.start_time else 0
        self.calls.append({
            "seq": len(self.calls) + 1,
            "t": round(elapsed, 2),
            "tool": tool_name,
            "args": args,
            "result_preview": result[:120],
        })

    def print_summary(self):
        print(f"\n  Tool calls made: {len(self.calls)}")
        seen_pairs = set()
        for c in self.calls:
            tag = ""
            if c["tool"] == "check_drug_interaction":
                pair = tuple(sorted(c["args"].values()))
                if pair in seen_pairs:
                    tag = " [DUPLICATE]"
                seen_pairs.add(pair)
            print(
                f"    #{c['seq']:>2} | {c['t']:>6.1f}s | {c['tool']}"
                f"({', '.join(f'{k}={v!r}' for k,v in c['args'].items())}){tag}"
            )
            # Show result on a second line, indented
            preview = c["result_preview"].replace("\n", " | ")
            print(f"         -> {preview}")

    def missed_pairs(self, all_drugs: list[str]) -> list[tuple[str, str]]:
        """Return drug pairs that were never checked."""
        checked = set()
        for c in self.calls:
            if c["tool"] == "check_drug_interaction":
                pair = tuple(sorted(v.lower() for v in c["args"].values()))
                checked.add(pair)
        # Normalize to avoid false "missed" reports when callers use mixed case.
        all_drugs_norm = [d.strip().lower() for d in all_drugs]
        all_pairs = {tuple(sorted(p)) for p in combinations(all_drugs_norm, 2)}
        return sorted(all_pairs - checked)


TOOL_LOG = ToolCallLog()


def logged_tool(fn):
    """Decorator that logs every call to TOOL_LOG."""
    import functools
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Build a clean args dict from positional + keyword args
        import inspect
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        args_dict = dict(bound.arguments)

        result = fn(*args, **kwargs)
        TOOL_LOG.record(fn.__name__, args_dict, str(result))
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Shared tools -- both approaches use the same tools
# ---------------------------------------------------------------------------

# Simulated drug interaction database
# IMPORTANT: Keys MUST be alphabetically sorted tuples, because the lookup
# function normalizes with tuple(sorted(...)).
INTERACTION_DB = {
    ("aspirin", "warfarin"): {
        "severity": "major",
        "effect": "Increased bleeding risk. Combined anticoagulant/antiplatelet effect.",
        "recommendation": "Avoid combination unless clinically necessary. Monitor INR closely.",
    },
    ("amiodarone", "warfarin"): {
        "severity": "major",
        "effect": "Amiodarone inhibits CYP2C9, increasing warfarin levels 30-50%.",
        "recommendation": "Reduce warfarin dose by 30-50%. Monitor INR weekly.",
    },
    ("lisinopril", "potassium"): {
        "severity": "major",
        "effect": "ACE inhibitors reduce potassium excretion. Hyperkalemia risk.",
        "recommendation": "Monitor serum potassium. Avoid supplementation unless hypokalemic.",
    },
    ("lisinopril", "metformin"): {
        "severity": "minor",
        "effect": "ACE inhibitors may improve insulin sensitivity slightly.",
        "recommendation": "Generally safe. Monitor blood glucose.",
    },
    ("aspirin", "lisinopril"): {
        "severity": "moderate",
        "effect": "Aspirin may reduce antihypertensive effect of ACE inhibitors.",
        "recommendation": "Use low-dose aspirin (<=100mg). Monitor blood pressure.",
    },
    ("aspirin", "metformin"): {
        "severity": "minor",
        "effect": "No clinically significant interaction.",
        "recommendation": "Safe to combine.",
    },
    ("amiodarone", "simvastatin"): {
        "severity": "major",
        "effect": "Amiodarone inhibits CYP3A4. Simvastatin levels increase, rhabdomyolysis risk.",
        "recommendation": "Do not exceed simvastatin 20mg/day with amiodarone.",
    },
    ("simvastatin", "warfarin"): {
        "severity": "moderate",
        "effect": "Simvastatin may modestly increase warfarin effect.",
        "recommendation": "Monitor INR when starting/stopping simvastatin.",
    },
}

CONTRAINDICATION_DB = {
    "chronic_kidney_disease": ["metformin", "potassium"],
    "liver_disease": ["simvastatin", "metformin"],
    "bleeding_disorder": ["warfarin", "aspirin"],
    "heart_failure": [],  # no absolute contraindications in our simplified model
}


@logged_tool
def check_drug_interaction(drug_a: str, drug_b: str) -> str:
    """Check interaction between two drugs. Returns JSON-like interaction data.

    Args:
        drug_a: First drug name (lowercase)
        drug_b: Second drug name (lowercase)

    Returns:
        Interaction details including severity, effect, and recommendation.
        Returns 'no known interaction' if none found.
    """
    key = tuple(sorted([drug_a.lower(), drug_b.lower()]))
    if key in INTERACTION_DB:
        info = INTERACTION_DB[key]
        return (
            f"INTERACTION: {drug_a} + {drug_b}\n"
            f"  Severity: {info['severity']}\n"
            f"  Effect: {info['effect']}\n"
            f"  Recommendation: {info['recommendation']}"
        )
    return f"No known interaction between {drug_a} and {drug_b}."


@logged_tool
def check_contraindication(drug: str, condition: str) -> str:
    """Check if a drug is contraindicated for a medical condition.

    Args:
        drug: Drug name (lowercase)
        condition: Medical condition (lowercase, underscores for spaces)

    Returns:
        Contraindication status and details.
    """
    condition = condition.lower().replace(" ", "_")
    drug = drug.lower()
    if condition in CONTRAINDICATION_DB:
        if drug in CONTRAINDICATION_DB[condition]:
            return f"CONTRAINDICATED: {drug} is contraindicated in {condition}."
        return f"No contraindication for {drug} in {condition}."
    return f"Unknown condition: {condition}."


@logged_tool
def get_drug_class(drug: str) -> str:
    """Get the pharmacological class of a drug.

    Args:
        drug: Drug name (lowercase)

    Returns:
        Drug class and mechanism of action.
    """
    classes = {
        "warfarin": "Anticoagulant (Vitamin K antagonist)",
        "aspirin": "Antiplatelet / NSAID (COX inhibitor)",
        "lisinopril": "ACE inhibitor (Antihypertensive)",
        "metformin": "Biguanide (Antidiabetic)",
        "amiodarone": "Class III antiarrhythmic (Potassium channel blocker)",
        "simvastatin": "HMG-CoA reductase inhibitor (Statin)",
        "potassium": "Electrolyte supplement",
    }
    drug = drug.lower()
    return classes.get(drug, f"Unknown drug: {drug}")


# ---------------------------------------------------------------------------
# Approach 1: Standard ReAct Agent
# ---------------------------------------------------------------------------


def build_react_agent():
    """Standard ReAct agent for drug interaction checking.

    The ReAct loop will:
      thought_1 -> "I need to check warfarin + aspirin"
      tool_1    -> check_drug_interaction("warfarin", "aspirin")
      obs_1     -> "INTERACTION: major..."
      thought_2 -> "Now check warfarin + lisinopril"
      tool_2    -> check_drug_interaction("warfarin", "lisinopril")
      ...
      thought_45 -> "Now let me also check contraindications..."
      ...

    For 7 drugs + 2 conditions, this easily burns 30-50 reasoning steps.
    Each step re-processes the entire trajectory, paying O(n^2) in tokens.
    The agent may also:
      - Forget to check some pairs (no systematic enumeration)
      - Lose track of which pairs were already checked
      - Fail to aggregate severity levels across all interactions
    """
    return dspy.ReAct(
        signature="medications: str, conditions: str -> risk_assessment: str",
        tools=[check_drug_interaction, check_contraindication, get_drug_class],
        max_iters=60,  # Need many iterations for pairwise checks
    )


# ---------------------------------------------------------------------------
# Approach 2: RLM-Enhanced Agent
# ---------------------------------------------------------------------------

def _build_python_interpreter():
    """Build a DSPy PythonInterpreter that can find Deno even if it's not on PATH."""
    from dspy.primitives.python_interpreter import PythonInterpreter

    # Ensure `deno` is discoverable for DSPy's cache-dir detection.
    # DSPy uses `deno info --json` (via `deno` on PATH) unless DENO_DIR is set.
    if not shutil.which("deno"):
        deno_candidate = os.path.expanduser("~/.deno/bin/deno")
        if os.path.exists(deno_candidate):
            os.environ["PATH"] = f"{os.path.dirname(deno_candidate)}{os.pathsep}{os.environ.get('PATH', '')}"

    # On macOS, this is the common default. Setting it avoids relying on a subprocess call.
    deno_cache_candidate = os.path.expanduser("~/Library/Caches/deno")
    if "DENO_DIR" not in os.environ and os.path.exists(deno_cache_candidate):
        os.environ["DENO_DIR"] = deno_cache_candidate

    # Clear the cached lookup so changes to PATH/DENO_DIR take effect.
    try:
        PythonInterpreter._get_deno_dir.cache_clear()
    except Exception:
        pass

    return PythonInterpreter()


def build_rlm_agent():
    """RLM agent that writes code to orchestrate tool calls.

    Instead of one-thought-one-tool, the RLM will write something like:

    ```python
    # Iteration 1: Systematic pairwise interaction check
    drugs = ["warfarin", "aspirin", "lisinopril", "metformin",
             "amiodarone", "simvastatin", "potassium"]
    from itertools import combinations

    interactions = []
    for a, b in combinations(drugs, 2):
        result = check_drug_interaction(a, b)
        if "INTERACTION" in result:
            interactions.append(result)
    print(f"Found {len(interactions)} interactions:")
    for i in interactions:
        print(i)
        print("---")
    ```

    ```python
    # Iteration 2: Check contraindications
    conditions = ["chronic_kidney_disease", "heart_failure"]
    contraindications = []
    for drug in drugs:
        for cond in conditions:
            result = check_contraindication(drug, cond)
            if "CONTRAINDICATED" in result:
                contraindications.append(result)
    print(f"Found {len(contraindications)} contraindications:")
    for c in contraindications:
        print(c)
    ```

    ```python
    # Iteration 3: Aggregate and classify risk
    major = [i for i in interactions if "major" in i.lower()]
    moderate = [i for i in interactions if "moderate" in i.lower()]

    summary = llm_query(
        f"Summarize these drug interactions into a clinical risk assessment:\\n"
        f"Major interactions ({len(major)}):\\n" + "\\n".join(major) + "\\n"
        f"Moderate interactions ({len(moderate)}):\\n" + "\\n".join(moderate) + "\\n"
        f"Contraindications ({len(contraindications)}):\\n" + "\\n".join(contraindications)
    )
    SUBMIT(risk_assessment=summary)
    ```

    Total: a few iterations, complete coverage, programmatic aggregation.
    """
    return dspy.RLM(
        signature="medications: str, conditions: str -> risk_assessment: str",
        tools=[check_drug_interaction, check_contraindication, get_drug_class],
        max_iterations=20,
        max_llm_calls=20,
        interpreter=_build_python_interpreter(),
    )


# ---------------------------------------------------------------------------
# Approach 3: RLM as a planner that FEEDS a ReAct agent
# ---------------------------------------------------------------------------


class RLMPlannerWithReAct(dspy.Module):
    """Hybrid: RLM plans the tool-calling strategy, ReAct executes.

    This is the most interesting architecture. The RLM doesn't call tools
    directly -- it writes a PLAN (which tools to call, in what order, with
    what arguments) based on programmatic analysis of the input. Then a
    standard ReAct agent executes the plan.

    Why this is powerful:
      - RLM handles the combinatorial reasoning (enumerating pairs, etc.)
      - ReAct handles the actual tool calling with built-in error recovery
      - The plan is inspectable and auditable
      - You can optimize the ReAct component with GEPA while keeping RLM fixed
    """

    def __init__(self):
        self.planner = dspy.RLM(
            signature=(
                "medications: str, conditions: str "
                "-> tool_call_plan: str, expected_interactions: int"
            ),
            max_iterations=10,
            max_llm_calls=5,
            interpreter=_build_python_interpreter(),
        )
        self.executor = dspy.ReAct(
            signature="medications: str, conditions: str, plan: str -> risk_assessment: str",
            tools=[check_drug_interaction, check_contraindication, get_drug_class],
            max_iters=60,
        )

    def forward(self, medications: str, conditions: str):
        # Phase 1: RLM writes the plan programmatically
        plan_result = self.planner(
            medications=medications,
            conditions=conditions,
        )
        # Phase 2: ReAct executes the plan with full context
        return self.executor(
            medications=medications,
            conditions=conditions,
            plan=plan_result.tool_call_plan,
        )


# ---------------------------------------------------------------------------
# Comparison: Why RLM wins on tool calling
# ---------------------------------------------------------------------------

COMPARISON = """
=============================================================================
WHY RLM IMPROVES TOOL CALLING (beyond long context)
=============================================================================

1. SYSTEMATIC ENUMERATION
   -------------------------
   ReAct: "Let me check warfarin+aspirin... now warfarin+lisinopril..."
          (May forget pairs, no guarantee of completeness)

   RLM:   for a, b in combinations(drugs, 2): check(a, b)
          (Provably complete, zero missed pairs)

2. PROGRAMMATIC AGGREGATION
   --------------------------
   ReAct: Must hold all results in the trajectory and reason over
          a growing wall of text to classify/aggregate.

   RLM:   major = [i for i in results if "major" in i]
          (Code-level filtering, sorting, counting)

3. CONDITIONAL LOGIC
   -------------------
   ReAct: "The interaction was major, so I should also check..."
          (Branching is implicit in natural language reasoning)

   RLM:   if severity == "major": deeper_check(drug_a, drug_b)
          (Explicit, auditable branching)

4. TOKEN EFFICIENCY
   ------------------
   ReAct: Each step re-reads the entire trajectory. For N tool calls,
          total tokens ~ O(N^2) as the trajectory grows.

   RLM:   Tool calls happen inside code execution (not in the prompt).
          The LLM only sees printed output. Total tokens ~ O(N).

5. BATCH OPERATIONS
   ------------------
   ReAct: One tool call per reasoning step. No parallelism.

   RLM:   results = [check(a,b) for a,b in pairs]  # all at once
          Or: answers = llm_query_batched(prompts)  # concurrent LLM calls

6. ERROR RECOVERY
   ----------------
   ReAct: If a tool call fails, the agent must reason about why and retry.
          Often gets stuck or gives up.

   RLM:   try:
              result = check(a, b)
          except:
              result = fallback_check(a, b)
          (Programmatic retry with explicit fallback logic)

7. STATE MANAGEMENT
   ------------------
   ReAct: All state lives in the natural language trajectory.
          "Earlier I found that warfarin and aspirin have a major interaction..."

   RLM:   results_dict[("warfarin", "aspirin")] = {"severity": "major", ...}
          (Structured data, no information loss from paraphrasing)

=============================================================================
WHEN TO USE WHICH
=============================================================================

Use ReAct when:
  - Few tool calls needed (< 5)
  - Tools are diverse (search, calculate, fetch -- different tools each step)
  - The reasoning path is genuinely uncertain and exploratory
  - You need the agent to "think out loud" for interpretability

Use RLM when:
  - Many similar tool calls needed (systematic exploration)
  - Results need cross-referencing or aggregation
  - The tool-calling pattern is COMPOSITIONAL (loops, conditionals)
  - Token efficiency matters (large number of tool interactions)
  - You need programmatic error handling

Use RLM+ReAct hybrid when:
  - Planning is combinatorial but execution needs flexibility
  - You want to optimize the executor with GEPA/MIPROv2
  - You need both systematic coverage AND adaptive reasoning
"""


# ---------------------------------------------------------------------------
# Main: run the comparison
# ---------------------------------------------------------------------------


def print_react_trajectory(trajectory: dict):
    """Pretty-print a ReAct agent's reasoning trajectory.

    ReAct trajectories contain keys like:
      thought_0, tool_name_0, tool_args_0, observation_0,
      thought_1, tool_name_1, ...
    """
    step = 0
    while f"thought_{step}" in trajectory or f"tool_name_{step}" in trajectory:
        thought = trajectory.get(f"thought_{step}", "")
        tool = trajectory.get(f"tool_name_{step}", "")
        args = trajectory.get(f"tool_args_{step}", "")
        obs = trajectory.get(f"observation_{step}", "")

        print(f"\n  --- Step {step} ---")
        if thought:
            wrapped = textwrap.fill(str(thought), width=72, initial_indent="    ",
                                    subsequent_indent="    ")
            print(f"  THINK: \n{wrapped}")
        if tool:
            print(f"  TOOL:  {tool}({args})")
        if obs:
            obs_str = str(obs)
            if len(obs_str) > 200:
                obs_str = obs_str[:200] + "..."
            wrapped = textwrap.fill(obs_str, width=72, initial_indent="    ",
                                    subsequent_indent="    ")
            print(f"  OBS:   \n{wrapped}")
        step += 1

    print(f"\n  Total reasoning steps: {step}")


def print_rlm_trajectory(trajectory):
    """Pretty-print an RLM agent's code execution trajectory.

    RLM trajectories show the code the LLM wrote and the output
    of each execution iteration.
    """
    if isinstance(trajectory, str):
        # Sometimes trajectory is a raw string
        sections = trajectory.split("---")
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            print(f"\n  --- Iteration {i} ---")
            # Try to separate code from output
            if ">>>" in section or "print(" in section:
                print(f"  CODE:")
                for line in section.split("\n"):
                    print(f"    {line}")
            else:
                print(f"  OUTPUT:")
                for line in section.split("\n")[:15]:
                    print(f"    {line}")
                if section.count("\n") > 15:
                    print(f"    ... ({section.count(chr(10)) - 15} more lines)")
    elif isinstance(trajectory, dict):
        for key in sorted(trajectory.keys()):
            val = str(trajectory[key])
            print(f"\n  [{key}]")
            for line in val.split("\n")[:10]:
                print(f"    {line}")
            if val.count("\n") > 10:
                print(f"    ... ({val.count(chr(10)) - 10} more lines)")
    elif isinstance(trajectory, list):
        for i, item in enumerate(trajectory):
            print(f"\n  --- Iteration {i} ---")
            item_str = str(item)
            for line in item_str.split("\n")[:15]:
                print(f"    {line}")
            if item_str.count("\n") > 15:
                print(f"    ... ({item_str.count(chr(10)) - 15} more lines)")
    else:
        print(f"  Trajectory type: {type(trajectory)}")
        print(f"  {str(trajectory)[:500]}")


def run_approach(name: str, agent, drugs_list: list[str], verbose: bool = True, **kwargs):
    """Run an agent once and return metrics used for multi-run summaries."""
    TOOL_LOG.reset()
    t0 = time.monotonic()
    expected_pairs = len(list(combinations(drugs_list, 2)))

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    try:
        result = agent(**kwargs)
        elapsed = time.monotonic() - t0
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"\n  FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")
        if verbose:
            TOOL_LOG.print_summary()
        missed = TOOL_LOG.missed_pairs(drugs_list)
        if verbose and missed:
            print(f"\n  Missed drug pairs ({len(missed)}):")
            for a, b in missed:
                print(f"    - {a} + {b}")
        return {
            "ok": False,
            "elapsed": elapsed,
            "tool_calls": len(TOOL_LOG.calls),
            "missed_pairs": len(missed),
            "expected_pairs": expected_pairs,
        }

    if verbose:
        # -- Result --
        risk = getattr(result, "risk_assessment", None) or str(result)
        print(f"\n  RESULT ({elapsed:.1f}s):")
        for line in textwrap.wrap(str(risk), width=72):
            print(f"    {line}")

        # -- Trajectory --
        traj = getattr(result, "trajectory", None)
        if traj:
            print(f"\n  TRAJECTORY:")
            if "thought_0" in (traj if isinstance(traj, dict) else {}):
                print_react_trajectory(traj)
            else:
                print_rlm_trajectory(traj)

        # -- Tool call log --
        TOOL_LOG.print_summary()

    # -- Coverage check --
    missed = TOOL_LOG.missed_pairs(drugs_list)
    if verbose and missed:
        print(f"\n  WARNING: Missed {len(missed)} drug pairs:")
        for a, b in missed:
            print(f"    - {a} + {b}")
    elif verbose:
        print(f"\n  COVERAGE: All {expected_pairs} drug pairs checked.")
    else:
        checked = expected_pairs - len(missed)
        coverage_pct = 100.0 * checked / expected_pairs
        print(
            f"\n  SUMMARY: {elapsed:.1f}s | "
            f"tool calls={len(TOOL_LOG.calls)} | "
            f"pair coverage={checked}/{expected_pairs} ({coverage_pct:.0f}%)"
        )

    return {
        "ok": True,
        "elapsed": elapsed,
        "tool_calls": len(TOOL_LOG.calls),
        "missed_pairs": len(missed),
        "expected_pairs": expected_pairs,
    }


def run_approach_repeated(
    name: str,
    agent_factory,
    runs: int,
    drugs_list: list[str],
    **kwargs,
):
    """Run one approach multiple times and print an average timing summary."""
    stats = []
    for run_idx in range(1, runs + 1):
        run_stats = run_approach(
            f"{name} (run {run_idx}/{runs})" if runs > 1 else name,
            agent_factory(),
            drugs_list,
            verbose=(run_idx == 1),
            **kwargs,
        )
        if run_stats["ok"]:
            stats.append(run_stats)

    if not stats:
        print(f"\n  No successful runs for {name}.")
        return {"name": name, "runs": runs, "successful_runs": 0}

    elapsed_values = [s["elapsed"] for s in stats]
    mean_elapsed = statistics.mean(elapsed_values)
    stdev_elapsed = statistics.stdev(elapsed_values) if len(elapsed_values) > 1 else 0.0
    mean_tool_calls = statistics.mean(s["tool_calls"] for s in stats)
    mean_missed_pairs = statistics.mean(s["missed_pairs"] for s in stats)

    print(
        f"\n  {name} AVERAGE over {len(stats)} successful runs: "
        f"{mean_elapsed:.1f}s +/- {stdev_elapsed:.1f}s | "
        f"tool calls={mean_tool_calls:.1f} | "
        f"missed pairs={mean_missed_pairs:.1f}"
    )

    return {
        "name": name,
        "runs": runs,
        "successful_runs": len(stats),
        "mean_elapsed": mean_elapsed,
        "stdev_elapsed": stdev_elapsed,
        "mean_tool_calls": mean_tool_calls,
        "mean_missed_pairs": mean_missed_pairs,
    }


def main():
    """Demonstrate the difference between ReAct and RLM for tool calling."""
    parser = argparse.ArgumentParser(
        description="Compare ReAct and RLM on compositional tool calling."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per approach for wall-time averaging (default: 3).",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")

    # Configure DSPy; disable LM cache so wall times reflect real inference work.
    lm = dspy.LM(
        "groq/openai/gpt-oss-120b",
        max_tokens=8000,
        temperature=0.1,
        cache=False,
    )
    dspy.configure(lm=lm)

    # The test case: 7 drugs, 2 conditions = 21 pairs + 14 contraindication checks
    medications = (
        "warfarin, amiodarone, simvastatin, aspirin, lisinopril, metformin, potassium"
    )
    conditions = "chronic_kidney_disease, heart_failure"
    drugs_list = [d.strip().lower() for d in medications.split(",")]

    print("\n" + "=" * 70)
    print("  RLM vs ReAct for Tool Calling")
    print("  Scenario: Drug interaction check for 7 medications, 2 conditions")
    print(f"  Expected pairwise checks: {len(list(combinations(drugs_list, 2)))}")
    print(f"  Expected contraindication checks: {len(drugs_list) * 2}")
    print(f"  Timing protocol: cache disabled, {args.runs} runs/approach, report mean")
    print("=" * 70)

    # ------------------------------------------------------------------
    # APPROACH 1: Standard ReAct
    # ------------------------------------------------------------------
    react_summary = run_approach_repeated(
        "APPROACH 1: Standard ReAct",
        build_react_agent,
        args.runs,
        drugs_list,
        medications=medications,
        conditions=conditions,
    )

    # ------------------------------------------------------------------
    # APPROACH 2: RLM Direct
    # ------------------------------------------------------------------
    rlm_summary = run_approach_repeated(
        "APPROACH 2: RLM Direct (programmatic tool orchestration)",
        build_rlm_agent,
        args.runs,
        drugs_list,
        medications=medications,
        conditions=conditions,
    )

    # ------------------------------------------------------------------
    # APPROACH 3: Hybrid (RLM planner + ReAct executor)
    # ------------------------------------------------------------------
    hybrid_summary = run_approach_repeated(
        "APPROACH 3: RLM Planner + ReAct Executor",
        RLMPlannerWithReAct,
        args.runs,
        drugs_list,
        medications=medications,
        conditions=conditions,
    )

    # ------------------------------------------------------------------
    # Timing summary (easy to copy into README)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  WALL-TIME SUMMARY (cache disabled)")
    print("=" * 70)
    for summary in [react_summary, rlm_summary, hybrid_summary]:
        if summary["successful_runs"] == 0:
            print(f"  {summary['name']}: no successful runs")
            continue
        print(
            f"  {summary['name']}: "
            f"{summary['mean_elapsed']:.1f}s +/- {summary['stdev_elapsed']:.1f}s "
            f"over {summary['successful_runs']}/{summary['runs']} runs"
        )

    # ------------------------------------------------------------------
    # Summary comparison
    # ------------------------------------------------------------------
    print(COMPARISON)


if __name__ == "__main__":
    main()
