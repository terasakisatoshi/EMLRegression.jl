# Repository Guidelines

## Project Structure & Module Organization
`EMLRegression` is a Julia package for reproducing Section 4.3 of the EML paper. Core package code lives in `src/`, with domain areas split by responsibility: `trees/`, `models/`, `training/`, `targets/`, `snapping/`, `eval/`, and `symbolics/`. Entry-point exports are collected in `src/EMLRegression.jl`.

Tests live under `test/` and mostly mirror the source layout, for example `src/training/TrainLoop.jl` pairs with `test/training/test_train_loop.jl`. CLI-style experiment runners are in `scripts/`. Configurable experiment inputs are in `experiments/configs/*.toml`. Generated artifacts go to `results/raw/`, `results/summaries/`, and `results/figures/`. Documenter sources are in `docs/src/`.

## Build, Test, and Development Commands
Use Julia 1.12.6 and keep commands project-scoped.

- `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.instantiate()'` installs package dependencies.
- `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'` runs the full test suite.
- `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'` runs one test group from `test/runtests.jl`.
- `~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config experiments/configs/must_pass_depth2.toml --target ln --seed 1` runs a single experiment.
- `~/.juliaup/bin/julia --project=. scripts/summarize_results.jl` rebuilds summary CSV output.
- `~/.juliaup/bin/julia --project=docs docs/make.jl` builds the Documenter site into `docs/build/`.

## Coding Style & Naming Conventions
Follow the existing Julia style: 4-space indentation, concise docstrings for public APIs, and explicit `using` imports near the top of each file. Keep module/type names in `CamelCase` (`TrainConfig`, `MasterTree`), functions in `snake_case` (`run_training`, `build_master_tree`), and tests named `test_*.jl`.

No formatter or linter config is committed yet, so match surrounding code closely and avoid opportunistic style rewrites.

## Testing Guidelines
Add tests in the mirrored area under `test/`, then register them through `test/runtests.jl` when introducing a new group. Prefer focused unit tests for module logic and add or update integration coverage when changing experiment flows or recovery behavior. There is no formal coverage gate, but changes should pass targeted tests and the full `Pkg.test()` suite before review.

## Commit & Pull Request Guidelines
Recent history uses short, imperative prefixes such as `feat:`, `docs:`, and `results:`. Keep commits scoped to one logical change. Pull requests should include a concise summary, the exact verification commands you ran, and any affected experiment config or result paths. If a change alters generated outputs, call out whether `results/raw/` files were overwritten; the runner intentionally reuses filenames per `config-target-seed`.
