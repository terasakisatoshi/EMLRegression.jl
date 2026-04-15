# TrainLoop Autodiff Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace finite-difference gradients in training with Zygote-based autodiff while preserving the current training API and benchmark flow.

**Architecture:** Keep the current `Lux` layer, recursive EML tree evaluation, and `Optimisers.Adam` update loop. Only the gradient-computation path changes: the training step will evaluate the same real-valued loss closure and obtain parameter gradients with `Zygote.gradient`.

**Tech Stack:** Julia, Lux.jl, Optimisers.jl, Zygote.jl, Test

---

### Task 1: Add regression tests for autodiff-backed training

**Files:**
- Modify: `test/training/test_train_loop.jl`

- [ ] **Step 1: Write the failing test**

Add a test that computes a training-style loss over `EMLTreeLayer` parameters and asserts that autodiff returns a gradient object with nonzero `left_logits`/`right_logits` entries.

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'`
Expected: FAIL because the regression test references an autodiff helper or behavior that is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

Implement the smallest production change needed so the new regression test passes without changing external training APIs.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'`
Expected: PASS for all training tests.

### Task 2: Replace finite-difference gradients in the training loop

**Files:**
- Modify: `src/training/TrainLoop.jl`
- Modify: `src/EMLRegression.jl`

- [ ] **Step 1: Remove finite-difference gradient usage from `run_training`**

Switch the training step from `_finite_difference_gradient(...)` to a Zygote-backed gradient over the same scalar loss closure.

- [ ] **Step 2: Keep loss real-valued and optimizer-compatible**

Ensure the loss path still returns a real scalar and the resulting gradient matches the existing parameter tree used by `Optimisers.update`.

- [ ] **Step 3: Remove obsolete helpers**

Delete the finite-difference helper functions if they are no longer used, and export any new helper only if tests need direct access.

- [ ] **Step 4: Run focused verification**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'`
Expected: PASS

### Task 3: Verify end-to-end behavior did not regress

**Files:**
- No additional file changes required unless a regression is found

- [ ] **Step 1: Run the full test suite**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 2: Sanity-check paper-aligned training behavior**

Run: `~/.juliaup/bin/julia --project=. scripts/run_suite.jl --config experiments/configs/must_pass_depth2.toml`
Expected: raw artifacts are produced without training-loop errors

- [ ] **Step 3: Review resulting worktree**

Run: `git status --short`
Expected: only intended files are modified
