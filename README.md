# EMLRegression

`EMLRegression` は、論文 [All elementary functions from a single operator](./All%20elementary%20functions%20from%20a%20single%20operator.pdf) の Section 4.3 を Julia で再現するための研究用リポジトリです。現在は `Lux.jl` を用いた CPU-first の再現系を対象にしており、複素数内部表現、EML 木、学習ループ、snapping、blind recovery 判定、実験スクリプト、結果集計の土台を実装しています。

## このリポジトリの目的

このプロジェクトの主目標は、論文の 4.3 節に出てくる「trainable EML trees」を Julia で再構成し、次の流れを再現可能な形にすることです。

- EML 木の定義
- `Lux.jl` ベースの学習可能モデル化
- hardening を含む学習ループ
- learned parameters の snapping
- snapped tree の評価と blind recovery 集計
- 単発実験とスイート実験の CLI 化

将来的には論文全体のパイプラインへ広げることを想定していますが、現時点の実装対象は Section 4.3 の再現基盤です。

## 現在の実装範囲

現時点で入っているもの:

- Julia パッケージ骨格
- master tree の最小構造
- `Lux.jl` ベースの EML レイヤの最小実装
- 複素数値の安定性チェック
- must-pass / challenge target の registry
- 学習ループの骨格
- snapping と recovered formula の最小出力
- recovery 判定と smoke experiment
- 実験設定ファイルと CLI
- raw JSON と summary CSV の生成

まだ本格化が必要なもの:

- 論文どおりの master-formula レベルの木構造
- 実際の最適化更新を伴う学習性能
- より強い snapping / ambiguity handling
- 論文レベルの recovery 成功率
- より厳密な symbolic verification

## 動作環境

- Julia `1.12.6`
- CPU-first
- 現在のマイルストーンでは GPU 実行は対象外

## セットアップ

依存解決:

```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

必要に応じて Julia 本体の確認:

```bash
~/.juliaup/bin/julia --version
```

## テスト

全テスト:

```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'
```

カテゴリ別テスト例:

```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["models"])'
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["targets"])'
```

## 単発実験

`ln` の smoke run:

```bash
~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config experiments/configs/must_pass_depth2.toml --target ln --seed 1
```

成功すると `results/raw/` に 1 本の JSON が出力されます。

## スイート実験

深さ 2 の must-pass セット:

```bash
~/.juliaup/bin/julia --project=. scripts/run_suite.jl --config experiments/configs/must_pass_depth2.toml
```

深さ 3 の must-pass セット:

```bash
~/.juliaup/bin/julia --project=. scripts/run_suite.jl --config experiments/configs/must_pass_depth3.toml
```

深さ 4 の challenge セット:

```bash
~/.juliaup/bin/julia --project=. scripts/run_suite.jl --config experiments/configs/challenge_depth4.toml
```

## 結果の集計と確認

集計 CSV を生成:

```bash
~/.juliaup/bin/julia --project=. scripts/summarize_results.jl
```

回復した式の一覧を表示:

```bash
~/.juliaup/bin/julia --project=. scripts/export_recovered_formulas.jl
```

主な出力先:

- `results/raw/`: 各 run の raw JSON
- `results/summaries/`: 集計結果
- `results/figures/`: 図の出力先

### 結果の扱い

- `results/raw/*.json` は `config-target-seed` ごとに同じファイル名を使います。
- そのため、同じ条件で再実行すると対応する raw JSON は上書きされます。
- `scripts/summarize_results.jl` は、その時点で `results/raw/` に存在する JSON をすべて集計します。
- 直近の 1 スイートだけを集計したい場合は、事前に `results/raw/` を退避するか整理してから実行してください。
- 現在のベースラインでは `success_count = 0` が並ぶのが既知の状態です。解釈は [First Baseline Report](./docs/first-baseline-report.md) を参照してください。

## ディレクトリ構成

```text
src/
  EMLRegression.jl         パッケージ入口
  trees/                   木構造
  models/                  Lux レイヤ
  training/                学習設定と学習ループ
  targets/                 ターゲット関数 registry
  snapping/                snapping 処理
  eval/                    recovery 判定
  symbolics/               recovered formula の出力

scripts/
  run_experiment.jl        単発実験
  run_suite.jl             スイート実験
  summarize_results.jl     集計
  export_recovered_formulas.jl

experiments/configs/
  実験設定 TOML

test/
  ユニットテストと smoke test

results/
  raw/                     実験ごとの JSON
  summaries/               集計結果
  figures/                 図の出力先

docs/
  reproduction-section-4-3.md
  first-baseline-report.md
```

## 関連ドキュメント

- [Section 4.3 Reproduction Guide](./docs/reproduction-section-4-3.md)
- [First Baseline Report](./docs/first-baseline-report.md)
- [Design Spec](./docs/superpowers/specs/2026-04-15-eml-lux-symbolic-regression-design.md)
- [Implementation Plan](./docs/superpowers/plans/2026-04-15-eml-lux-symbolic-regression.md)

## 現時点の注意

- ベースラインは通るものの、blind recovery 成功率はまだ 0 のままです。
- 現状のモデルは研究用 scaffold としては成立していますが、論文の本格再現としては未完成です。
- したがって、この README は「再現済み」の主張ではなく、「再現基盤と実験導線が揃っている」段階の説明です。
