# Parallel Test Runner Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ParallelTestRunner.jl` を使って `test/` をファイル単位で並列実行できるようにする。

**Architecture:** `test/runtests.jl` を自動検出ランナーへ置き換え、各テストファイルはそのまま独立実行させる。テスト入口変更に合わせて、テスト依存と利用ドキュメントを同期更新する。

**Tech Stack:** Julia 1.12.6, `Pkg.test`, `ParallelTestRunner.jl`

---

### Task 1: Add a regression test for runner CLI support

**Files:**
- Create: `test/test_parallel_test_runner.jl`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run the targeted test to verify it fails**
Run: `~/.juliaup/bin/julia --project=. test/test_parallel_test_runner.jl`
Expected: FAIL because `test/runtests.jl --list` does not enumerate discovered test files yet.
- [ ] **Step 3: Implement the minimal runner changes**
- [ ] **Step 4: Run the targeted test to verify it passes**
Run: `~/.juliaup/bin/julia --project=. test/test_parallel_test_runner.jl`
Expected: PASS

### Task 2: Switch the test entry point to ParallelTestRunner

**Files:**
- Modify: `Project.toml`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Add `ParallelTestRunner` to test dependencies**
- [ ] **Step 2: Replace manual `include` dispatch with `runtests(EMLRegression, ARGS)`**
- [ ] **Step 3: Re-run the targeted regression test**

### Task 3: Update test documentation

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/src/manual.md`

- [ ] **Step 1: Replace category-based examples with parallel-runner examples**
- [ ] **Step 2: Mention `--list` and `--jobs=N` usage**
- [ ] **Step 3: Re-read the docs for consistency with the new entry point**

### Task 4: Verify serial and parallel test execution

**Files:**
- No code changes

- [ ] **Step 1: Run full serial tests**
Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'`
- [ ] **Step 2: Run full parallel tests**
Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["--jobs=2"])'`
- [ ] **Step 3: Inspect output and confirm both commands exit successfully**
