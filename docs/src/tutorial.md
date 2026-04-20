# チュートリアル: 論文の基本例を最小再現する

このチュートリアルでは、原論文「All elementary functions from a single operator」に出てくる最も基本的な例を、現在の `EMLRegression.jl` で追います。

狙いは 2 つです。

1. 論文の中心演算子 `eml(x, y) = exp(x) - log(y)` が本当に `exp` や `ln` を作れることを手で確かめる
2. このリポジトリの最小実装で `exp`, `ln`, `-x`, `1/x`, `x+y`, `x*y` の target を実際に回してみる

このページは、論文の完全再現ではなく「まず何が起きているかを掴む」ための入口です。

## 0. 準備

依存を解決してテストを通します。

```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.instantiate()'
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'
```

## 1. 原論文の最初の例をそのまま試す

論文の冒頭で、著者は次の 2 つを直接挙げています。

```math
e^x = \mathrm{eml}(x, 1)
```

```math
\ln x = \mathrm{eml}(1, \mathrm{eml}(\mathrm{eml}(1, x), 1))
```

まず Julia でそのまま評価します。

```bash
~/.juliaup/bin/julia --project=. -e '
using EMLRegression
x = 2.0 + 0.0im
println("exp via eml     = ", eml(x, 1.0 + 0.0im))
println("exp reference   = ", exp(x))
println("log via eml     = ", eml(1.0 + 0.0im, eml(eml(1.0 + 0.0im, x), 1.0 + 0.0im)))
println("log reference   = ", log(x))
'
```

期待する見方は単純です。

- `exp via eml` と `exp reference` が一致する
- `log via eml` と `log reference` が一致する

この 2 つが見えれば、論文の「EML 1 個から `exp` と `ln` を起こせる」という最小主張はその場で確認できます。

## 2. `+` と `*` の考え方

論文では、まず古典的な exp-log の関係

```math
x \times y = e^{\ln x + \ln y}
```

```math
x + y = \ln(e^x \times e^y)
```

を出発点にしています。ここで重要なのは、`exp` と `ln` が作れてしまえば、加算や乗算もそこから再構成できるという流れです。

現在のリポジトリは、これらの完全な EML 展開式を人手で並べるよりも、`target = :add` や `target = :mul` を実験対象として直接回す作りになっています。したがって、このチュートリアルでは:

- 数式レベルでは「`exp` と `ln` から `+` と `*` へ進む」という論文の発想を押さえる
- 実装レベルでは `add` と `mul` を学習ターゲットとして動かす

という分担にします。

## 3. `+` と `*` を数値で確認する

まずは pure EML 展開の前に、論文が出発点にしている exp-log の関係をそのまま数値確認します。

```bash
~/.juliaup/bin/julia --project=. -e '
x = 0.7 + 0.0im
y = 1.2 + 0.0im

mul_via_explog = exp(log(x) + log(y))
add_via_explog = log(exp(x) * exp(y))

println("mul via exp-log = ", mul_via_explog)
println("mul reference   = ", x * y)
println("add via exp-log = ", add_via_explog)
println("add reference   = ", x + y)
'
```

ここでは `x` と `y` を正の実数にしています。これは branch cut の話を避けて、「`exp` と `ln` ができれば `+` と `*` へ進める」という論文の骨格だけを見るためです。

今の実装では、`add` と `mul` の pure EML 形を最短で人手展開する代わりに、これらをそのまま学習ターゲットとして回します。

## 4. `0`, `x^{-1}`, `-x` を EML だけで作る

論文の Figure 2 には `-x` と `x^{-1}` の木も載っています。ここでは、現在の Julia 実装でその考え方を追える形に書き下します。

まず `0` を作ります。

```math
0 = \mathrm{eml}(1, \mathrm{eml}(\mathrm{eml}(1,1),1))
```

理由は:

- `eml(1,1) = e`
- `eml(e,1) = e^e`
- `eml(1,e^e) = e - \ln(e^e) = 0`

次に `\log(0) = -\infty` を作ります。論文でも、EML では内部的に複素数や IEEE754 的な特異値が重要になると説明されています。

```math
\log(0) =
\mathrm{eml}\!\left(
1,
\mathrm{eml}\!\left(
\mathrm{eml}(1, 0),
1
\right)
\right)
```

この `\log(0)` を使うと、逆数は

```math
x^{-1} = \mathrm{eml}(\mathrm{eml}(\log(0), x), 1)
```

と書けます。なぜなら

```math
\mathrm{eml}(\log(0), x) = e^{\log(0)} - \log(x) = 0 - \log(x) = -\log(x)
```

なので

```math
\mathrm{eml}(-\log(x), 1) = e^{-\log(x)} = x^{-1}
```

となるからです。

さらに `-x` は

```math
-x = \log((e^x)^{-1})
```

なので、`exp` と `inv` を組み合わせて作れます。

以下の Julia スニペットで確認できます。

```bash
~/.juliaup/bin/julia --project=. -e '
using EMLRegression
one = 1.0 + 0.0im
x = 2.0 + 0.0im

zero_eml() = eml(one, eml(eml(one, one), one))
logzero_eml() = eml(one, eml(eml(one, zero_eml()), one))
inv_eml(z) = eml(eml(logzero_eml(), z), one)
neg_eml(z) = eml(one, eml(eml(one, inv_eml(eml(z, one))), one))

println(\"zero via eml = \", zero_eml())
println(\"logzero via eml = \", logzero_eml())
println(\"inv via eml = \", inv_eml(x))
println(\"inv reference = \", inv(x))
println(\"neg via eml = \", neg_eml(x))
println(\"neg reference = \", -x)
'
```

ここで大事なのは、`inv` と `neg` が「EML のきれいな有限多項式」だけで済むとは限らず、`0` や `-Inf` のような中間値を経由しうることです。これは論文の実装注意とも一致します。

## 5. この実装で `exp`, `ln`, `add`, `mul` を単発実験する

まず `exp` と `ln` の単発実験を回します。

```bash
~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config experiments/configs/must_pass_depth2.toml --target exp --seed 1
~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config experiments/configs/must_pass_depth2.toml --target ln --seed 1
```

続いて、二変数ターゲットとして `add` と `mul` を回します。

```bash
~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config experiments/configs/must_pass_depth2.toml --target add --seed 1
~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config experiments/configs/must_pass_depth2.toml --target mul --seed 1
```

対応する raw 結果はここに保存されます。

- `results/raw/must_pass_depth2-exp-seed1.json`
- `results/raw/must_pass_depth2-ln-seed1.json`
- `results/raw/must_pass_depth2-add-seed1.json`
- `results/raw/must_pass_depth2-mul-seed1.json`

中身を見るには例えば:

```bash
sed -n '1,120p' results/raw/must_pass_depth2-exp-seed1.json
sed -n '1,120p' results/raw/must_pass_depth2-ln-seed1.json
sed -n '1,120p' results/raw/must_pass_depth2-add-seed1.json
sed -n '1,120p' results/raw/must_pass_depth2-mul-seed1.json
```

見るべき項目は次です。

- `target`
- `success`
- `reason`
- `formula`
- `train_loss`

現時点の実装では、ここで論文どおりの recovery 成功を期待するのではなく、「学習から raw JSON 出力までの流れが通っているか」を確認します。

## 6. must-pass セットを回す

論文の基本例に近い 6 ターゲット

- `exp`
- `ln`
- `neg`
- `inv`
- `add`
- `mul`

は、すでに `must_pass_depth2.toml` に入っています。まとめて回すには:

```bash
~/.juliaup/bin/julia --project=. scripts/run_suite.jl --config experiments/configs/must_pass_depth2.toml
```

このスイートは、各ターゲットに対して複数 seed を流します。

## 7. 集計結果を確認する

スイート実行後に集計します。

```bash
~/.juliaup/bin/julia --project=. scripts/summarize_results.jl
```

結果は `results/summaries/summary.csv` に保存されます。

```bash
sed -n '1,80p' results/summaries/summary.csv
```

ここでは、少なくとも `must_pass_depth2` に対して

- `exp`
- `ln`
- `neg`
- `inv`
- `add`
- `mul`

の各行が出てくることを確認してください。

## 8. このチュートリアルで何が再現できたか

この時点で確認できたことは次です。

- 論文が明示した `exp(x)` と `ln(x)` の EML 表現を、そのまま Julia で評価できた
- 論文の exp-log 的な再構成として `x+y` と `x*y` を数値確認できた
- 論文の基本ターゲット集合 `exp`, `ln`, `-x`, `1/x`, `x+y`, `x*y` に対応する実験コマンドを実行できた
- raw JSON と summary CSV を通じて、Section 4.3 型の実験導線を追えた

まだ確認できていないこともあります。

- 論文レベルの exact recovery 成功率
- 深い master tree の本格最適化
- `+` や `*` の短い EML 展開式そのものの自動回収

このコードベースは、そこへ向かうための最小 scaffold です。

## 9. 次に読む場所

チュートリアルの次は、この順で読むと追いやすいです。

1. [学習方法](training.md)
2. [論文の要点と解説](paper.md)
3. `scripts/run_experiment.jl`
4. `src/eval/Recovery.jl`
5. `src/targets/TargetRegistry.jl`
