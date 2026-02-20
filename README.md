# RLM vs ReAct for Compositional Tool Calling

## Thesis

DSPy's RLM (Recursive Language Model) module is often introduced via a
"sandboxed REPL" pattern, where the LLM writes and runs Python to explore an
input iteratively. But RLM's deeper value lies elsewhere: **programmatic
orchestration of tool calls**.

The standard agentic pattern, ReAct (Reasoning + Acting), interleaves one thought
with one tool call per step:

```
think -> pick tool -> call tool -> observe -> think -> pick tool -> ...
```

This works well for exploratory tasks with a few diverse tools. But when a task
requires **compositional tool use** -- calling the same tool many times with
different arguments, cross-referencing results, and aggregating outputs -- ReAct
degrades in predictable ways:

- **Incomplete coverage**: the LLM uses domain heuristics to pick "important"
  pairs rather than enumerating all of them
- **Superlinear token cost**: each step often re-reads a growing trajectory, so
  total tokens can approach O(N^2) as the number of tool calls grows
- **No programmatic aggregation**: classification, filtering, and counting must
  happen in natural language reasoning, which is lossy

RLM sidesteps all of this by writing code:

```python
for a, b in combinations(drugs, 2):
    result = check_drug_interaction(a, b)
    if "INTERACTION" in result:
        interactions.append(result)
```

Once the code is written and executed, the enumeration is complete by
construction (e.g. all `N choose 2` pairs).

## Experiment

### Scenario: Drug Interaction Checker

A patient takes 7 medications and has 2 medical conditions. A safety check
requires:

- **21 pairwise drug interaction checks** (7 choose 2)
- **14 drug-condition contraindication checks** (7 drugs x 2 conditions)
- Aggregation by severity level (major/moderate/minor)
- A structured risk assessment

Three approaches are compared using DSPy v3.1.3 with `groq/openai/gpt-oss-120b`.

The exact tool-call behavior can vary across runs/providers/model versions
(LLMs are not deterministic). For latency claims, use this timing protocol:
initialize `dspy.LM(..., cache=False)`, run each approach 3 times, and report
the mean wall time.

| Approach | Architecture | Description |
|---|---|---|
| **ReAct** | `dspy.ReAct` | Standard reasoning-action loop |
| **RLM Direct** | `dspy.RLM` with tools | LLM writes Python in a sandboxed REPL to orchestrate tool calls |
| **RLM + ReAct Hybrid** (ablation) | `dspy.RLM` planner, `dspy.ReAct` executor | RLM generates a structured plan, ReAct executes it. Included as a controlled ablation (see below). |

### Tools

All three approaches share identical tools backed by simulated databases:

- `check_drug_interaction(drug_a, drug_b)` -- returns severity and clinical
  details for a drug pair
- `check_contraindication(drug, condition)` -- checks whether a drug is
  contraindicated for a medical condition
- `get_drug_class(drug)` -- returns pharmacological classification

### Medications and Conditions

```
Medications: warfarin, amiodarone, simvastatin, aspirin, lisinopril, metformin, potassium
Conditions:  chronic_kidney_disease, heart_failure
```

The interaction database contains 8 known interactions (4 major, 2 moderate,
2 minor) and 2 contraindications (metformin and potassium in CKD).

## Results

Model and runtime configuration used:
- DSPy v3.1.3
- `groq/openai/gpt-oss-120b`
- `dspy.LM(..., cache=False)` (no cache)
- dependencies refreshed with `uv sync -U`
- 3 runs per approach (`--runs 3`)
- wall time reported as mean +/- standard deviation

| Metric | ReAct | RLM Direct | RLM + ReAct |
|---|---|---|---|
| Wall time (mean +/- sd) | 13.1s +/- 1.1s | **8.3s +/- 2.7s** | 65.4s +/- 1.4s |
| Successful runs | 3 / 3 | 3 / 3 | 2 / 3 |
| Pairwise coverage signal | 4-5 / 21 | **21 / 21 in all 3 runs** | Unstable (one 21 / 21 run, one 4 / 21 run, one failure) |
| Notable failure | none | none | `AttributeError: 'NoneType' object has no attribute 'strip'` (1 run) |

Current takeaway:
- ReAct produces strong narrative reasoning but does not reliably enumerate all pairs.
- RLM Direct is fastest in this benchmark and consistently achieves complete pairwise coverage.
- RLM+ReAct remains useful as an ablation architecture, but is much slower and less stable here.

## Analysis

### Why RLM Wins on Compositional Tool Use

| Dimension | ReAct | RLM |
|---|---|---|
| **Enumeration** | Heuristic; may forget pairs | `for a, b in combinations(drugs, 2)` -- provably complete |
| **Aggregation** | Reasons over a growing wall of text | `major = [i for i in results if "major" in i]` |
| **Branching** | Implicit in natural language | `if severity == "major": deeper_check(a, b)` |
| **Token cost** | O(N^2) -- trajectory re-read each step | Often closer to O(N) for the *tool enumeration* portion, because tool calls can happen inside code and the LLM mainly sees summarized/printed output |
| **Parallelism** | One tool call per step | `results = [check(a,b) for a,b in pairs]` in one execution |
| **Error recovery** | Must reason about failure and retry | `try/except` with explicit fallback |
| **State** | Natural language paraphrasing (lossy) | `results_dict[key] = value` (lossless) |

### Model Sensitivity (Important)

These numbers are model-specific. The benchmark above used
`groq/openai/gpt-oss-120b`, and outcomes can change significantly with stronger
or weaker models.

This is consistent with the Recursive Language Models paper: RLM-style
program synthesis/orchestration depends heavily on the base model's capability.
Stronger models generally produce more reliable planning, code generation,
self-correction, and final outputs.

Practical takeaway: treat this repo's metrics as a reference point, not a fixed
ceiling. If you are evaluating RLM seriously, test multiple stronger models and
re-run the same uncached multi-run protocol.

### When to Use Which

**Use ReAct when:**

- Few tool calls are needed (< 5)
- Tools are diverse (search, calculate, fetch -- a different tool each step)
- The reasoning path is genuinely uncertain and exploratory
- You need the agent to "think out loud" for interpretability
- Latency matters and the task is simple

**Use RLM when:**

- Many similar tool calls are needed (systematic exploration)
- Results need cross-referencing, filtering, or aggregation
- The tool-calling pattern is compositional (loops, conditionals)
- Token efficiency matters (large number of tool interactions)
- You need programmatic error handling
- Completeness is more important than speed

**The RLM + ReAct hybrid** is included as an ablation study (see Approach 3
above). It demonstrates that completeness is a property of the plan, not the
executor. In this benchmark, RLM Direct is faster on average (8.3s vs 65.4s)
and more stable. The
hybrid's value is primarily analytical, not practical.

## A Neuro-Symbolic Perspective

RLM, at least as used here, is a neuro-symbolic technique. The LLM (neural)
synthesizes an executable Python program (symbolic artifact), and a sandboxed
interpreter (symbolic executor) runs it deterministically. The program provides
formal guarantees -- completeness of enumeration, structured aggregation -- that
the neural model alone does not reliably achieve (4-5/21 pair coverage for
ReAct vs 21/21 for RLM Direct in this benchmark).

### Kautz Taxonomy Classification

In Henry Kautz's taxonomy of neuro-symbolic architectures (AAAI 2020 Robert S.
Engelmore Memorial Lecture), this approach maps to **Type 6: Neuro[Symbolic]** --
a neural system that generates and manipulates symbolic representations.

The defining characteristic of Type 6 is that the neural component *produces*
symbolic structures which are then executed or reasoned over formally. The neural
model is the outer system; the symbolic program is its output, not its wrapper.
This distinguishes it from Type 2 (Symbolic[Neuro]), where a symbolic framework
calls neural subroutines (e.g. AlphaGo's MCTS calling value networks).

The three experimental approaches actually span the taxonomy:

| Approach | Kautz Type | Rationale |
|---|---|---|
| ReAct | Pure neural (Type 1 at best) | LLM reasons heuristically; no symbolic guarantees |
| RLM Direct | **Type 6: Neuro[Symbolic]** | LLM generates executable program with provable completeness |
| RLM + ReAct | **Type 6 / Type 2 hybrid** | Symbolic plan (Type 6 output) structures a neural executor (Type 2 pattern) |

The progression from ReAct to RLM Direct to the hybrid is a gradient from pure
neural to increasingly neuro-symbolic. In this benchmark, RLM Direct remains
consistently complete while the hybrid is unstable. This highlights where
symbolic structure adds value in the Kautz framework, while also underscoring
model and executor sensitivity.

### What Makes This Neuro-Symbolic

The core division of labor is:

- **Neural:** Understanding the task semantically, synthesizing the appropriate
  program, interpreting tool results
- **Symbolic:** Deterministic enumeration (nested loops, `combinations`),
  filtering (conditionals on structured data), aggregation (dicts and lists),
  and a verification metric (set intersection over expected vs actual pairs)

The LLM does not just "think" in natural language. It *compiles* its reasoning
into executable code. The code provides formal guarantees that pure neural
reasoning cannot: if you write `for a, b in combinations(drugs, 2)`, you get
all pairs by construction, not by heuristic. The verification metric
(`tool_coverage_metric`) is itself purely symbolic -- a deterministic set
intersection with no LLM in the loop.

This is not neuro-symbolic in the classical sense of formal logic, ontologies, or
theorem proving. It is closer to what the literature calls **neurosymbolic
program synthesis** -- the neural model compiles intent into a formal,
verifiable, deterministic artifact. But it captures the essential thesis of
neuro-symbolic AI: neither paradigm alone is sufficient, and the interface
between them is where the value lives.

## Practical Applications

The RLM pattern applies to any domain where a task decomposes into many
structurally similar tool calls that must be exhaustively executed, then
aggregated. The signature is:

1. **Input is a list or set of entities** (drugs, servers, securities, accounts)
2. **The check is pairwise or cross-product** (N choose 2, or N x M)
3. **Results must be filtered, classified, or aggregated**
4. **Completeness matters** (missing one pair could be catastrophic)

### Compliance and Audit

```python
rlm = dspy.RLM(
    "policies: str, systems: str -> compliance_report: str",
    tools=[check_policy_compliance, get_system_config, check_data_residency],
    max_iterations=20,
)
```

Check every system against every regulatory policy. For 50 systems and 20
policies, that is 1000 checks. ReAct may not enumerate all of them without
being explicitly forced to. RLM writes
`for system in systems: for policy in policies: check(system, policy)`.

### Security Vulnerability Scanning

```python
rlm = dspy.RLM(
    "endpoints: str, attack_vectors: str -> vulnerability_report: str",
    tools=[test_endpoint, check_cve_database, verify_tls_config],
    max_iterations=20,
)
```

Test every API endpoint against every known attack vector. Aggregate results by
severity. Flag endpoints with multiple vulnerabilities. The same loop-and-filter
pattern applies.

### Financial Portfolio Risk Analysis

```python
rlm = dspy.RLM(
    "holdings: str, risk_factors: str -> risk_report: str",
    tools=[get_correlation, check_exposure, calculate_var],
    max_iterations=20,
)
```

For N holdings, compute the N*(N-1)/2 pairwise correlations, check each holding
against each risk factor, calculate portfolio-level VaR. ReAct might check the
top 5 correlations; RLM checks all of them.

### Supply Chain Dependency Audit

```python
rlm = dspy.RLM(
    "components: str, suppliers: str -> supply_chain_risk: str",
    tools=[check_supplier_status, find_alternatives, assess_lead_time],
    max_iterations=20,
)
```

For every component, check every supplier for sanctions compliance, financial
health, and lead time. Cross-reference single-source dependencies. The
combinatorial structure is identical.

### Test Matrix Execution

```python
rlm = dspy.RLM(
    "features: str, environments: str -> test_report: str",
    tools=[run_test, check_compatibility, get_test_history],
    max_iterations=20,
)
```

Run every feature test against every environment (OS x browser x version).
Aggregate pass/fail rates. Identify environment-specific regressions. RLM
writes the nested loop; ReAct would check a handful and declare victory.

### The General Pattern

Any time you find yourself thinking "I need to check every X against every Y and
then summarize the results," you want RLM over ReAct. The more Xs and Ys there
are, the greater the advantage.

```python
# The RLM pattern for exhaustive cross-referencing
rlm = dspy.RLM(
    "items_a: str, items_b: str -> report: str",
    tools=[check_pair, get_details],
    max_iterations=20,
    max_llm_calls=50,
)
```

The LLM will write something like:

```python
a_list = [x.strip() for x in items_a.split(",")]
b_list = [x.strip() for x in items_b.split(",")]

results = []
for a in a_list:
    for b in b_list:
        result = check_pair(a, b)
        if is_significant(result):
            results.append({"a": a, "b": b, "details": result})

summary = llm_query(f"Summarize these {len(results)} findings: {results}")
SUBMIT(report=summary)
```

A few iterations. Complete coverage (when you loop exhaustively). Structured
output. No missed pairs within the enumerated space.

## Setup

### Requirements

- Python 3.13+ (matches `.python-version` and `pyproject.toml`)
- [Deno](https://deno.land) (for RLM's sandboxed execution)
- A [Groq](https://groq.com) API key (`GROQ_API_KEY`) for the default model
  configured in the script

### Installation

```bash
# Clone and set up (this repo already includes pyproject.toml + uv.lock)
cd /path/to/project
uv venv -p 3.13
source .venv/bin/activate
uv sync --frozen

# Install Deno (required for RLM)
curl -fsSL https://deno.land/install.sh | sh

# IMPORTANT: Restart your shell after installing Deno, or run:
export PATH="$HOME/.deno/bin:$PATH"
```

Note: `rlm_vs_react_tool_calling.py` will also try to find Deno at
`~/.deno/bin/deno` if it is not on your `PATH`, but exporting `PATH` is still
the simplest/most reliable setup.

### Configuration

Create a `.env` file:

```
GROQ_API_KEY=your-key-here
```

### Known Deno Issue

Observed issue (macOS): if a `package.json`, `deno.json`, or `deno.lock` exists
in a parent directory of the project, Deno may attempt local `node_modules`
resolution instead of using its global cache. This can break RLM's sandbox
permissions because DSPy only whitelists `~/.deno/bin` and the Deno cache
directory (`~/Library/Caches/deno` on macOS) for `--allow-read`.

**Symptom:** `Response ID mismatch during health check: expected 1, got None`

**Fix:** Remove stray `package.json` / `deno.lock` / `node_modules` from
directories above your project. These files cause Deno to resolve `npm:pyodide`
from a local `node_modules` that the sandbox cannot read.

### Running

```bash
source .venv/bin/activate
uv run --env-file .env -p 3.13 -- python rlm_vs_react_tool_calling.py
```

By default, the script now runs each approach 3 times with `cache=False` and
prints a `WALL-TIME SUMMARY` with mean and standard deviation. You can change
the repeat count:

```bash
uv run --env-file .env -p 3.13 -- python rlm_vs_react_tool_calling.py --runs 3
```

If you want to capture the full run output:

```bash
uv run --env-file .env -p 3.13 -- python rlm_vs_react_tool_calling.py | tee output_runs3_uncached.txt
```

To reproduce the benchmark format shown above:

```bash
uv sync -U
uv run --env-file .env -p 3.13 -- python rlm_vs_react_tool_calling.py --runs 3 | tee output_runs3_uncached.txt
```

## Optional: GEPA Optimization (Training)

This repo also includes an optional GEPA training script, `gepa_train_rlm.py`,
that compiles an optimized version of the RLM agent. This is useful if you want
to demonstrate DSPy's differentiator: optimizing the *tool-orchestration
behavior* (what code the RLM writes, how it filters, how it formats outputs).

Important notes:
- GEPA uses additional LLM calls and will incur cost.
- The default metric is a deterministic tool-call "coverage" check (did the
  agent actually enumerate all `N choose 2` interaction calls and all `N x M`
  contraindication calls). This is cheap and directly tests the thesis. You can
  also choose an LLM-based metric (`semantic_f1`), which is more expensive.
- `gepa_train_rlm.py` lets you choose a separate **reflection model** (used by
  GEPA to propose instruction mutations) that can be stronger than the
  **program model** you're optimizing.

Dry-run (no LLM calls):

```bash
uv run --env-file .env -p 3.13 -- python gepa_train_rlm.py --dry-run
```

Small-budget smoke run (will incur some cost):

```bash
uv run --env-file .env -p 3.13 -- python gepa_train_rlm.py --max-metric-calls 6 --metric coverage
```

Program/reflection example:

```bash
uv run --env-file .env -p 3.13 -- python gepa_train_rlm.py --program-model groq/openai/gpt-oss-20b --reflection-model groq/openai/gpt-oss-120b --max-metric-calls 6 --metric coverage
```

## Disclaimer

This project uses a **simulated** drug interaction/contraindication database to
demonstrate tool-calling patterns. It is **not medical advice** and should not
be used for clinical decisions.

## References

- [DSPy documentation](https://dspy.ai)
- [DSPy RLM module](https://dspy.ai/api/modules/RLM/)
- [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, and Khattab, 2025)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)
- [The Third AI Summer](https://www.youtube.com/watch?v=_cQITY0SPiw) -- Henry Kautz, AAAI 2020 Robert S. Engelmore Memorial Lecture (neuro-symbolic taxonomy)
