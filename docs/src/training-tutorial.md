# 学習チュートリアル

このページは、[学習方法](training.md) の概説を実際のコードにつなぐための実践編です。

ここで目指すのは次の 4 点です。

1. `TrainConfig` を作る
2. `run_training` を呼んでメトリクスを見る
3. `run_experiment` で recovery 判定まで流す
4. `snap_model` と `formula_string` で現在の recovered 表現を確認する

## 1. 最小の設定を作る

まずは `ln` を題材に、小さい設定を作ります。

```julia
using EMLRegression

cfg = TrainConfig(
    depth = 2,
    target = :ln,
    batch_size = 16,
    steps = 8,
    hardening_steps = 4,
)
```

ここで重要なのは次の 4 つです。

- `depth`
  木の深さ
- `target`
  今回は `:ln`
- `steps`
  学習ループのステップ数
- `hardening_steps`
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
length(training.metrics[:train_loss])
length(training.metrics[:hardening_loss])
training.metrics[:train_loss][1]
training.metrics[:train_loss][end]
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
experiment.recovery.reason
experiment.recovered_tree
formula_string(experiment.recovered_tree)
```

現時点では `formula_string` は非常に簡単な文字列表現しか返しません。これはまだ最小実装だからです。ここでは「学習から recovered tree までつながっている」ことを確認すれば十分です。

## 4. `snap_model` を自分で呼んでみる

`run_experiment` の内部では、学習後に layer と params から recovered tree を作っています。これを明示的に書くと次のようになります。

```julia
target = get_target(cfg.target)
variables = target.arity == 1 ? (:x,) : (:x, :y)

tree = build_master_tree(depth = cfg.depth, variables = variables)
layer = EMLTreeLayer(tree)

recovered = snap_model(layer, training.params)

formula_string(recovered)
```

このコードで、

- `cfg.target` から arity を取得する
- それに応じて `MasterTree` を組む
- その tree と学習済み params から snapping する

という流れが見えます。

## 5. ターゲット関数の入力と評価を確認する

学習対象そのものも、少しだけ覗いておくと理解が進みます。

```julia
target = get_target(:ln)
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
    target = :ln,
    batch_size = 16,
    steps = 8,
    hardening_steps = 4,
)

training = run_training(cfg; rng = StableRNG(1))

println("failure_reason = ", training.failure_reason)
println("train_loss_count = ", length(training.metrics[:train_loss]))
println("hardening_loss_count = ", length(training.metrics[:hardening_loss]))
println("last_train_loss = ", training.metrics[:train_loss][end])

experiment = run_experiment(cfg; rng = StableRNG(1))
println("recovery_success = ", experiment.recovery.success)
println("recovery_reason = ", experiment.recovery.reason)
println("recovered_formula = ", formula_string(experiment.recovered_tree))

target = get_target(cfg.target)
variables = target.arity == 1 ? (:x,) : (:x, :y)
tree = build_master_tree(depth = cfg.depth, variables = variables)
layer = EMLTreeLayer(tree)
recovered = snap_model(layer, training.params)

println("manual_snap_formula = ", formula_string(recovered))
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
