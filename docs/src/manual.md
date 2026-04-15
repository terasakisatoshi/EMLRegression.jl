# 使い方マニュアル

## 前提

- Julia `1.12` 系
- リポジトリ直下を作業ディレクトリにすること
- 現在のマイルストーンは CPU-first です

## 1. セットアップ

パッケージ本体の依存を解決します。

```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

必要なら Julia 自体の確認も行います。

```bash
~/.juliaup/bin/julia --version
```

## 2. テストを流す

最低限、先にテストを通してから実験を回すのが安全です。

```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'
```

カテゴリ別テストを見たい場合は `test_args` を使います。

```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["models"])'
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'
```

## 3. 単発実験を回す

まずは `ln` の smoke run が最も追いやすい入口です。

```bash
~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config experiments/configs/must_pass_depth2.toml --target ln --seed 1
```

成功すると `results/raw/` に JSON が 1 本出力されます。

### 出力の読み方

raw JSON には次のような情報が入ります。

- `config_name`
- `depth`
- `target`
- `seed`
- `success`
- `reason`
- `snap_status`
- `structure_match`
- `numerical_match`
- `training_failure_reason`
- `formula`
- `train_loss`
- `hardening_loss`

論文の最小例から入りたい場合は、先に [チュートリアル](tutorial.md) を読むと流れが掴みやすいです。

## 4. スイート実験を回す

must-pass の深さ 2:

```bash
~/.juliaup/bin/julia --project=. scripts/run_suite.jl --config experiments/configs/must_pass_depth2.toml
```

must-pass の深さ 3:

```bash
~/.juliaup/bin/julia --project=. scripts/run_suite.jl --config experiments/configs/must_pass_depth3.toml
```

challenge の深さ 4:

```bash
~/.juliaup/bin/julia --project=. scripts/run_suite.jl --config experiments/configs/challenge_depth4.toml
```

## 5. 結果を集計する

summary CSV を作るには次を使います。

```bash
~/.juliaup/bin/julia --project=. scripts/summarize_results.jl
```

このコマンドは、`results/raw/` に存在する JSON をすべて集計します。

summary には strict recovery と数値一致を分けて見るための列も入ります。

- `success_rate`
- `numerical_match_rate`
- `snap_ok_rate`
- `ambiguous_rate`
- `structure_match_rate`
- `training_failure_rate`

### 重要

- `results/raw/*.json` は `config-target-seed` ごとに同じファイル名です
- 同じ条件で再実行するとファイルは上書きされます
- ある 1 回のスイートだけを別集計したい場合は、事前に `results/raw/` を退避してください

## 6. 回復した式を見る

```bash
~/.juliaup/bin/julia --project=. scripts/export_recovered_formulas.jl
```

現時点の実装では、ここに出る式は非常に簡単な形です。これは現状の snapping とモデル表現が最小実装だからです。

## 7. Documenter サイトをビルドする

この `docs/` ディレクトリ自体も Documenter.jl で管理しています。

```bash
~/.juliaup/bin/julia --project=docs docs/make.jl
```

ビルド後の HTML は `docs/build/` に出ます。

## 8. `servedocs` でローカル確認する

編集しながらブラウザで確認したい場合は、`LiveServer.jl` の `servedocs` を使えます。

```bash
~/.juliaup/bin/julia --project=docs -e 'using LiveServer; servedocs(foldername="docs", buildfoldername="build", launch_browser=false)'
```

このリポジトリでは `docs/` を Documenter のルートにしているため、`foldername="docs"` を明示しています。

必要ならポートも指定できます。

```bash
~/.juliaup/bin/julia --project=docs -e 'using LiveServer; servedocs(foldername="docs", buildfoldername="build", launch_browser=false, host="127.0.0.1", port=8001)'
```

起動後に `http://localhost:8000/` または指定したポートへアクセスしてください。

## よくある見え方

### `success_count` がずっと 0

現時点のベースラインでは既知の状態です。これは実験導線の実装が主目的で、まだ論文レベルの recovery 性能が入っていないためです。

### `overwriting existing raw result:` と表示される

同じ条件の JSON を再実行で上書きしたという意味です。異常ではありません。
