# 学習チュートリアル

このページは、[学習方法](training.md) の概説を実際のコードにつなぐための実践編です。

ここで目指すのは次の 4 点です。

1. `TrainConfig` を作る
2. `run_training` を呼んでメトリクスを見る
3. `run_experiment` で recovery 判定まで流す
4. `analyze_snap` と `hard_project` で現在の snap 状態を確認する

## 1. 最小の設定を作る

まずは `eml_depth2` を題材に、小さい設定を作ります。

```julia
using EMLRegression

cfg = TrainConfig(
    depth = 2,
    target = :eml_depth2,
    batch_size = 16,
    search_iters = 8,
    hardening_iters = 4,
)
```

ここで重要なのは次の 4 つです。

- `depth`
  木の深さ
- `target`
  今回は `:eml_depth2`
- `search_iters`
  search phase のステップ数
- `hardening_iters`
  hardening 用に記録するステップ数

現在の実装は研究用 scaffold なので、まずは `steps` を小さくして全体の流れを確認するのが適切です。

## 2. `run_training` を呼ぶ

```julia
using StableRNGs

training = run_training(cfg; rng = StableRNG(1))
```

返ってくる `training` は [`TrainingResult`](@ref) です。最初に見ると分かりやすい項目は次です。

```julia
training.failure_reason
length(training.metrics[:soft_rmse])
length(training.metrics[:hard_rmse])
training.metrics[:soft_rmse][1]
training.metrics[:soft_rmse][end]
training.summary[:hardening_iter]
```

今の実装ではパラメータ更新がまだ入っていないため、「本格的に学習して loss が下がる」ことよりも、

- 学習ループが回る
- サンプル生成と評価が通る
- メトリクスが収集される

ことを確認する段階だと考えてください。

## 3. `run_experiment` で recovery 判定まで流す

次は学習、snapping、blind recovery 判定までを一気に流します。

```julia
experiment = run_experiment(cfg; rng = StableRNG(1))
```

確認したい項目は次です。

```julia
experiment.recovery.success
experiment.recovery.fit_success
experiment.recovery.symbol_success
experiment.recovery.stable_symbol_success
experiment.snap.n_uncertain
```

ここでは「学習から snap/recovery 判定までつながっている」ことを確認すれば十分です。

## 4. `analyze_snap` と `hard_project` を自分で呼んでみる

`run_experiment` の内部では、学習後に logits の曖昧さを見て hard projection を作っています。これを明示的に書くと次のようになります。

```julia
snap_info = analyze_snap(training.params; snap_threshold = cfg.snap_threshold)
snapped = hard_project(training.params)

snap_info.n_uncertain
snapped.leaf_logits
snapped.blend_logits
```

このコードで、

- 学習済み params のどこがまだ曖昧かを見る
- hard projection 後の leaf/gate logits を確認する

という流れが見えます。

## 5. ターゲット関数の入力と評価を確認する

学習対象そのものも、少しだけ覗いておくと理解が進みます。

```julia
target = get_target(:eml_depth2)
xs = sample_domain(target, 4; rng_seed = 1)
ys = evaluate_target(target, xs)

target.arity
target.tier
xs
ys
```

これで、

- どのような入力点がサンプルされるか
- その target がそこでどう評価されるか

を直接見られます。

## 6. 1 本にまとめた最小スクリプト

上の流れをまとめると、次のようになります。

```julia
using EMLRegression
using StableRNGs

cfg = TrainConfig(
    depth = 2,
    target = :eml_depth2,
    batch_size = 16,
    search_iters = 8,
    hardening_iters = 4,
)

training = run_training(cfg; rng = StableRNG(1))

println("failure_reason = ", training.failure_reason)
println("soft_rmse_count = ", length(training.metrics[:soft_rmse]))
println("hard_rmse_count = ", length(training.metrics[:hard_rmse]))
println("last_soft_rmse = ", training.metrics[:soft_rmse][end])

experiment = run_experiment(cfg; rng = StableRNG(1))
println("fit_success = ", experiment.recovery.fit_success)
println("symbol_success = ", experiment.recovery.symbol_success)
println("stable_symbol_success = ", experiment.recovery.stable_symbol_success)
println("n_uncertain = ", experiment.snap.n_uncertain)

snap_info = analyze_snap(training.params; snap_threshold = cfg.snap_threshold)
snapped = hard_project(training.params)
println("manual_n_uncertain = ", snap_info.n_uncertain)
println("manual_leaf_logits = ", snapped.leaf_logits)
```

これは REPL でも、`julia --project=.` でそのまま流しても動きます。

## 7. 何がまだ入っていないか

このチュートリアルで触ったコードは、Section 4.3 の最小骨格です。まだ入っていないものは多くあります。

- 実際の勾配更新
- ノードごとの独立パラメータ
- 本格的な hardening
- より賢い snapping
- symbolic verification

したがって、このページの目的は「現在のコードがどうつながっているかを手で追う」ことです。

## 8. 次の一歩

さらに内部構造まで触りたい場合は、[発展: 低レベル学習 API](advanced-training.md) に進んでください。そこでは `MasterTree`、`EMLTreeLayer`、`Lux.setup`、`Lux.apply` を直接触ります。
