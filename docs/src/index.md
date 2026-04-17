# EMLRegression.jl

`EMLRegression.jl` は、論文「All elementary functions from a single operator」の Section 4.3 を Julia で再構成するための研究用パッケージです。

現在の実装は、`Lux.jl` を使った CPU-first の最小再現系です。複素数内部表現、EML 演算子、最小の master tree、学習ループ、snapping、blind recovery 判定、CLI スクリプトまでを一通り備えています。

## このドキュメントの読み方

- [使い方マニュアル](manual.md)
  実験を動かすためのセットアップ、テスト、単発実験、スイート実験、結果確認をまとめています。
- [チュートリアル](tutorial.md)
  論文の `exp` / `ln` の基本例を手で確認し、そのあと現在の実装で `exp`, `ln`, `neg`, `inv`, `add`, `mul` を回します。
- [学習方法](training.md)
  現在の学習実装が何をしているか、論文の理想形とどこが違うかを説明します。
- [学習チュートリアル](training-tutorial.md)
  `TrainConfig` から `run_training`、`run_experiment`、`snap_model` までをコードで追います。
- [発展: 低レベル学習 API](advanced-training.md)
  `MasterTree`、`EMLTreeLayer`、`Lux.setup`、`Lux.apply` を直接触ります。
- [論文の要点と解説](paper.md)
  論文全体の狙いと、このリポジトリが担当している範囲を日本語で整理しています。
- [Section 4.3 との差分整理](section-4-3-gap.md)
  論文の Section 4.3 と現実装がどこで一致し、どこで意図的にずれているかを整理しています。
- [Section 4.3 の一次情報一覧](section-4-3-sources.md)
  Supplementary Information、Zenodo snapshot、EML toolkit など、忠実化のために当たる一次情報をまとめています。
- [API リファレンス](api.md)
  公開 API の要点をまとめています。

## 現在の位置づけ

このプロジェクトは「論文を完全再現した」段階ではありません。現時点では次を重視しています。

- 実験導線が明確であること
- 小さいコードベースで EML 木の流れを追えること
- 将来の高精度実装に拡張しやすい骨格を持つこと

そのため、blind recovery 成功率そのものよりも、研究用 scaffold としての読みやすさと再実行性を優先しています。

## すぐに試す

```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.instantiate()'
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'
~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config experiments/configs/must_pass_depth2.toml --target ln --seed 1
~/.juliaup/bin/julia --project=. scripts/summarize_results.jl
```

Documenter サイト自身のビルドは次で行えます。

```bash
~/.juliaup/bin/julia --project=docs docs/make.jl
```
