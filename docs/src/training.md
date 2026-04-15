# 学習方法

コードを動かしながら追いたい場合は、まず [学習チュートリアル](training-tutorial.md) を見てください。`TrainConfig` から `run_experiment` までを一通り触れます。さらに低レベル API を直接触りたい場合は [発展: 低レベル学習 API](advanced-training.md) を使ってください。

## 学習フローの全体像

現在の `EMLRegression.jl` は、論文 4.3 節の trainable EML tree をそのまま完全再現しているわけではありません。代わりに、次の骨格を明示的に追える最小実装を採っています。

1. ターゲット関数を選ぶ
2. その arity に応じた `MasterTree` を構築する
3. `EMLTreeLayer` を `Lux.jl` レイヤとして初期化する
4. バッチごとに入力をサンプルし、ターゲット値を計算する
5. モデル出力とターゲットの MSE を記録する
6. 学習後に logits を snapping して離散木へ落とす
7. blind recovery 判定を行う

## 現在の `EMLTreeLayer`

現状のレイヤは、終端候補に対する重み付き選択を行い、その結果を `eml(weighted, 1)` に通す最小形です。

これは論文の最終目標よりかなり簡略化されています。まだ次の要素は入っていません。

- 論文と同規模の深さ 5/6 実験
- basin-of-attraction の大規模比較
- PyTorch `complex128` 実装との厳密一致

## `TrainConfig`

学習設定は [`TrainConfig`](@ref) で与えます。主要フィールドは次のとおりです。

- `depth`
  木の深さ
- `target`
  学習対象の関数名
- `batch_size`
  各ステップでサンプルする点の数
- `steps`
  学習ステップ数
- `init_strategy`
  初期値の与え方。`small_gaussian`、`zero_bias_to_inputs`、`margin_biased`、`subtree_favoring`、`target_tree_noise` を選べます
- `target_noise_std`
  `init_strategy = :target_tree_noise` のときだけ使う Gaussian noise の強さ
- `hardening_steps`
  hardening を何ステップ回すか
- `hardening_weight`
  hardening 項の重み

## ターゲット関数とサンプリング

ターゲット関数は [`TargetSpec`](@ref) として登録されています。各ターゲットには次が含まれます。

- 変数の数 `arity`
- must-pass か challenge か
- 入力サンプラ
- 真の評価関数

複素数内部表現を採っているため、`ln` や `sqrt` のような関数でも同じ配列型で処理できます。ただし branch cut や数値安定性の問題は残ります。

## 数値安定性

現在は次の補助関数で、極端に悪い値が出ていないかを観測できます。

- [`inspect_complex_values`](@ref)
- [`finite_or_flag`](@ref)
- [`clamp_complex_magnitude`](@ref)

これらは論文再現のための「完成形の対策」ではなく、まず壊れ方を把握するための計測手段です。

## snapping

学習後の logits を離散化する処理は [`snap_logits`](@ref) と [`snap_model`](@ref) にあります。

現状は plain argmax だけではなく、`top-k` 候補を beam search する `search_recovered_tree` を先に通します。これは depth 3/4 の strict recovery を押し上げるための実装上の工夫です。

## blind recovery 判定

最終判定は [`recovery_verdict`](@ref) と [`run_experiment`](@ref) が担当します。

現在は:

- snapping が成功したか
- 数値的に一致したか

を主に見ています。理想的にはここに、より厳密な symbolic verification や外挿点での評価も加えるべきです。

## 今後の強化ポイント

- より深い木での basin-of-attraction 実験
- depth 5/6 を含む成功率比較
- 論文の複素数安定化戦略との厳密な整合
- `Symbolics.jl` による外挿含みの式検証

このページは「今の実装が何をしているか」を理解するためのものであり、「論文どおりに再現できた」と主張するためのものではありません。
