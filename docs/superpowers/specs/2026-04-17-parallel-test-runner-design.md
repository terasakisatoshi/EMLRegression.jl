# ParallelTestRunner Design

## Goal

`test/runtests.jl` を `ParallelTestRunner.jl` ベースの自動検出ランナーに置き換え、`Pkg.test(test_args=["--jobs=4"])` のような並列実行を可能にする。

## Current State

- `test/runtests.jl` はカテゴリ名で `include` を切り替える直列構成。
- 各テストファイルはすでに `using Test` / `using EMLRegression` を自前で持っており、ファイル単位実行に寄せやすい。
- README / manual / AGENTS はカテゴリ別 `test_args` を前提に説明している。

## Proposed Change

1. `Project.toml` のテスト依存に `ParallelTestRunner` を追加する。
2. `test/runtests.jl` を `using EMLRegression`, `using ParallelTestRunner`, `runtests(EMLRegression, ARGS)` のみへ置き換える。
3. `--list` が利用可能であることを確認する回帰テストを追加する。
4. README / `docs/src/manual.md` / `AGENTS.md` のテスト実行例を、`Pkg.test()` と `Pkg.test(test_args=["--jobs=4"])` ベースに更新する。

## Error Handling

- 並列数未指定時は `ParallelTestRunner` の既定動作に従う。
- 部分実行は旧カテゴリ指定ではなく、`--list` で確認できるファイル名フィルタに寄せる。

## Testing

- 追加した `--list` 回帰テストを先に失敗させる。
- 変更後にそのテストを通す。
- 最後に `Pkg.test()` と `Pkg.test(test_args=["--jobs=2"])` を実行して、直列・並列の両方が通ることを確認する。
