# 発展: 低レベル学習 API

このページは、`EMLRegression.jl` の内部構造を直接触りたい人向けです。

標準の [学習チュートリアル](training-tutorial.md) では `run_training` と `run_experiment` を入口にしましたが、ここではさらに下の層を見ます。

扱うものは次です。

- `build_master_tree`
- `EMLTreeLayer`
- `Lux.setup`
- `Lux.apply`

## 1. `MasterTree` を手で組む

まずは 1 変数の tree を作ります。

```julia
using EMLRegression

tree = build_master_tree(depth = 2, variables = (:x,))

tree.depth
tree.variables
tree.terminals
length(tree.nodes)
```

ここで見ておきたいのは、

- `terminals` に `:const1` と `:x` が入ること
- 深さ 2 の完全二分木に対応するノード数が入ること

です。

## 2. `EMLTreeLayer` を作る

```julia
layer = EMLTreeLayer(tree)
```

現状の `EMLTreeLayer` は、tree の終端候補に対して softmax 的な重み付けを行い、その重み付き入力を `eml(weighted, 1)` に通す最小実装です。

## 3. `Lux.setup` で初期パラメータを作る

```julia
using Lux
using StableRNGs

ps, st = Lux.setup(StableRNG(1), layer)

ps
st
```

現時点では `ps` の中心は `logits` です。初期状態では 0 ベクトルなので、各 terminal 候補は同じ重みから始まります。

## 4. `Lux.apply` を直接呼ぶ

1 変数入力に対して layer を評価します。

```julia
xs = ComplexF64[0.5 + 0.0im, 1.0 + 0.0im, 1.5 + 0.0im]

preds, st2 = Lux.apply(layer, xs, ps, st)

preds
st2
```

この `preds` が、今の最小 EML layer の出力です。

## 5. 2 変数入力も試す

`add` や `mul` のような 2 変数 target に近い形も触れます。

```julia
tree_xy = build_master_tree(depth = 2, variables = (:x, :y))
layer_xy = EMLTreeLayer(tree_xy)
ps_xy, st_xy = Lux.setup(StableRNG(1), layer_xy)

xs_xy = (
    ComplexF64[0.5 + 0.0im, 1.0 + 0.0im],
    ComplexF64[1.5 + 0.0im, 2.0 + 0.0im],
)

preds_xy, st_xy2 = Lux.apply(layer_xy, xs_xy, ps_xy, st_xy)

preds_xy
st_xy2
```

これで `:x` と `:y` の terminal を持つ layer がどう評価されるかを確認できます。

## 6. `get_target` と組み合わせる

低レベル API と target 情報をつなぐ最小例です。

```julia
target = get_target(:add)
variables = target.arity == 1 ? (:x,) : (:x, :y)

tree = build_master_tree(depth = 2, variables = variables)
layer = EMLTreeLayer(tree)
ps, st = Lux.setup(StableRNG(1), layer)

xs = sample_domain(target, 4; rng_seed = 1)
ys = evaluate_target(target, xs)
preds, _ = Lux.apply(layer, xs, ps, st)

ys
preds
```

ここまで来ると、`run_training` が内部でやっていることをほぼ手でなぞれます。

## 7. このページの位置づけ

この low-level API は、将来:

- ノードごとのパラメータ化
- より深い tree
- 実際の optimizer 更新
- custom loss

を入れるときの土台になります。

一方で、日常的に実験を回すだけなら、まずは [学習チュートリアル](training-tutorial.md) から始める方が自然です。
