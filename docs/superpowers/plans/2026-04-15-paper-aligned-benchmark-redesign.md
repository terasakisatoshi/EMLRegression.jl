# Paper-Aligned Benchmark Redesign Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current benchmark suites with paper-aligned EML-composed targets and recovery metrics so Section 4.3-style blind recovery rates can be measured directly.

**Architecture:** Keep the current trainable EML tree core, but redesign the benchmark layer around discrete target trees. Targets, recovery verdicts, suite configs, and summaries all become snapped-tree-centric so reported success rates reflect the paper’s intended experiment.

**Tech Stack:** Julia 1.12.6, Lux.jl, Optimisers.jl, StableRNGs.jl, JSON3.jl, CSV.jl, DataFrames.jl, Test

---

## File Map

- Modify: `src/targets/TargetRegistry.jl`
- Modify: `src/snapping/Snap.jl`
- Modify: `src/eval/Metrics.jl`
- Modify: `src/eval/Recovery.jl`
- Modify: `scripts/run_suite.jl`
- Modify: `scripts/summarize_results.jl`
- Modify: `scripts/run_experiment.jl`
- Modify: `scripts/export_recovered_formulas.jl`
- Modify: `experiments/configs/must_pass_depth2.toml`
- Modify: `experiments/configs/must_pass_depth3.toml`
- Modify: `experiments/configs/challenge_depth4.toml`
- Modify: `README.md`
- Modify: `test/targets/test_targets.jl`
- Modify: `test/eval/test_recovery.jl`
- Modify: `test/integration/test_smoke_ln.jl`

### Task 1: Replace Named Targets With Paper-Aligned EML-Composed Targets

### Task 2: Extend Recovery Metrics Around Snapped-Tree Success

### Task 3: Rewrite Suite Configs And Batch Execution Around Paper Targets

### Task 4: Upgrade Summaries To Report Recovery, Ambiguity, And Structure Rates

### Task 5: Run The Redesigned Suites, Update Docs, And Lock In The New Baseline
