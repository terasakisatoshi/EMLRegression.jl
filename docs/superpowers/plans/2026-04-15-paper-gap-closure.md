# Paper Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the highest-impact gaps between the current Julia implementation and the paper by tightening recovery evaluation, integrating numerical stability into training, broadening benchmark reporting, and making hyperparameter search practical.

**Architecture:** Keep the current trainable EML tree structure, but make the evaluation contract snapped-tree-centric and make the training loop explicitly record and react to numerical instability. Extend the benchmark layer so summaries expose paper-relevant recovery metrics, then add sweep-friendly experiment configs on top of that stricter foundation.

**Tech Stack:** Julia, Lux.jl, Zygote, Optimisers.jl, CSV.jl, DataFrames.jl

---

### Task 1: Tighten Recovery Success

**Files:**
- Modify: `src/eval/Metrics.jl`
- Modify: `src/eval/Recovery.jl`
- Modify: `test/eval/test_recovery.jl`
- Modify: `test/integration/test_smoke_ln.jl`

- [ ] Step 1: Write failing tests for strict snapped-tree recovery success.
- [ ] Step 2: Run the focused recovery tests and verify they fail for the current numerical-match behavior.
- [ ] Step 3: Implement strict `success` / `reason` semantics from `snap_status` and `structure_match`.
- [ ] Step 4: Run focused recovery and integration tests and verify they pass.

### Task 2: Integrate Numerical Stability Into Training

**Files:**
- Modify: `src/training/TrainConfig.jl`
- Modify: `src/training/TrainLoop.jl`
- Modify: `src/eval/Recovery.jl`
- Modify: `test/training/test_stability.jl`
- Modify: `test/training/test_train_loop.jl`

- [ ] Step 1: Write failing tests for clamped / flagged non-finite forward values in training.
- [ ] Step 2: Run the focused training tests and verify they fail.
- [ ] Step 3: Add config knobs and training-loop stability handling with explicit failure reporting.
- [ ] Step 4: Run focused training tests and verify they pass.

### Task 3: Expand Benchmark Reporting

**Files:**
- Modify: `scripts/summarize_results.jl`
- Modify: `test/integration/test_smoke_ln.jl`
- Modify: `docs/src/manual.md`
- Modify: `docs/src/paper.md`
- Modify: `docs/reproduction-section-4-3.md`

- [ ] Step 1: Write failing summary-script expectations for strict recovery metrics.
- [ ] Step 2: Run the summary-script tests and verify they fail.
- [ ] Step 3: Extend summary outputs and docs to surface strict recovery, ambiguity, and structure diagnostics clearly.
- [ ] Step 4: Run focused integration/docs checks and verify they pass.

### Task 4: Add Sweep-Friendly Experiment Controls

**Files:**
- Modify: `experiments/configs/*.toml`
- Create or modify: `scripts/run_suite.jl`
- Modify: `test/integration/test_smoke_ln.jl`

- [ ] Step 1: Write failing tests for the new sweep-oriented benchmark config expectations.
- [ ] Step 2: Run focused integration tests and verify they fail.
- [ ] Step 3: Add or update configs and runner behavior so depth 3/4 sweeps can be executed without ad hoc editing.
- [ ] Step 4: Run integration tests and at least one representative suite command and verify they pass.
