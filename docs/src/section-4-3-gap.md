# Section 4.3 との差分整理

このページでは、論文「All elementary functions from a single operator」の Section 4.3
「Symbolic Regression by continuous optimization」と、現在の `EMLRegression.jl`
 実装の差分だけを整理します。

前提として、このリポジトリは Section 4.3 の完全再現ではなく、
「Julia で追跡・改良しやすい研究用 scaffold」を目標にしています。

## 要約

現在の実装は、論文の発想である

- 固定深さの EML master tree
- 連続最適化
- hardening
- snapping
- blind recovery 判定

を取り入れています。

一方で、論文そのものと比べると、特に次の点が異なります。

- master formula の内部表現
- 学習対象のパラメタ化
- snapping 後処理
- 数値安定化の作り
- target suite と評価規模

## 1. Master Formula の表現が違う

論文の Section 4.3 では、各入力を

```math
\alpha_i + \beta_i x + \gamma_i f
```

という形で表し、各ノード入力が

- `1`
- 入力変数 `x`
- 直前の部分木 `f`

のどれになるかを、連続パラメタから表現します。

現在の実装はこの形をそのままは採用していません。代わりに、各辺ごとに

- `:const1`
- `:x`
- `:y`
- 子ノード

の候補を並べ、その上で logits と softmax によって候補値を混合しています。

つまり、論文は「線形結合で 1 / x / f を表す master formula」ですが、
現実装は「候補ソースを選ぶ master tree」です。

この差は本質的です。現実装は論文よりも構造が単純で追いやすい一方、
論文が想定している parameterization と探索空間そのものは一致していません。

## 2. 学習しているものが違う

論文では、`α, β, γ` を logits として扱い、softmax で simplex 上の重みに変換します。
Section 4.3 の最初の `ln x` の例では、simplex reparameterization と `NMinimize` が使われています。

現在の実装では、学習対象は node ごとの left/right choice logits です。
各ノード入力は「候補集合の確率混合」で計算され、`α + βx + γf` 型の実数係数そのものは持ちません。

そのため、論文に近い意味での

- 線形結合係数の学習
- simplex 頂点への丸め

ではなく、

- 候補選択の sharpen
- argmax による離散化

を行っています。

## 3. Snapping 後処理は現実装の方が工学的に強い

論文本文では、hardening 後に重みを `0` または `1` に clamp して、
exact symbolic values へ落とす流れが説明されています。

現実装はそれより後処理が強く、単純な argmax snapping だけで終わりません。
現在は次を行います。

- argmax による初期 snapping
- active node 上での top-k neighbor 展開
- beam search
- greedy refinement
- left-packing
- `log` 系の等価変形 rewrite

そのため、論文の「重みを 0/1 に丸めた結果」と、
現実装の「離散木探索を含む recovered tree」は同じ意味ではありません。

現実装は、学習で得た logits をそのまま読むというより、
学習結果を初期点として離散復元を補強しています。

## 4. 数値安定化の実装詳細が違う

論文は、`torch.complex128` を使った Python 実装を前提に、

- multiply composed exponentials による overflow
- complex arithmetic に起因する `NaN`
- `exp(x)` の引数 clamp
- 実部・虚部の careful inspection

を強調しています。

現実装でも数値安定化は入っていますが、具体的な作りは別です。

- `ComplexF64` を使用
- `exp` 前に実部のみ clamp
- ノード出力の絶対値を上限で clamp
- 出力と内部ノードに対する `NaN` / `Inf` 検査

したがって、「複素数で不安定になりやすい EML 学習を抑える」という方向性は同じでも、
論文実装そのものを再現したわけではありません。

## 5. 最適化条件が一致していない

論文本文では、Machine Learning 版の実験は

- Python
- `torch.complex128`
- Adam
- multi-stage optimization
- hardening

で行われています。

現実装は Julia ベースで、

- `Lux.jl`
- `Zygote`
- `Optimisers.Adam`

を使っています。

つまり、「Adam で学習する」という大枠は共通ですが、

- autodiff 系
- 複素数演算の実装
- 安定化の挿し込み方
- 学習の収束特性

は一致しません。

## 6. Target Suite がかなり小さい

論文では、composed EML から作った 2 変数関数を使って systematic experiments を行っています。
本文では 1000 超の runs、複数 seed、複数 initialization strategy を使った比較が述べられています。

現在の実装の paper-aligned target は、深さ 2 から 6 までの curated な少数例に限られます。

- `depth2_exp`
- `depth2_double_exp`
- `depth3_log`
- `depth4_nested`
- `depth5_affine_log`
- `depth6_inverse_logy`

このため、現在の結果は論文の成功率表を再現するというより、
論文の Section 4.3 のメカニズムを点検するための小規模ベンチマークです。

## 7. 評価規模と成功率はまだ論文水準に届いていない

論文本文の記述では、blind recovery はおおむね次の傾向です。

- depth 2: 100%
- depth 3-4: 約 25%
- depth 5: 1% 未満
- depth 6: `0/448`

さらに、正解木の近傍から始める basin-of-attraction 実験では、
depth 5 と 6 でも 100% 戻ると述べています。

現実装はこの方向性を検証し始めてはいますが、2026-04-19 時点の
`section-4-3-remaining-work` worktree にある local rerun も
まだ full paper sweep ではなく、
`depth3` / `depth4` の `4`-seed random-start slice、
`depth5_manual_noise12` / `depth6_manual_noise12` の `4`-seed manual rerun、
そして `depth2` の anchor run に留まります。

- seed 数がまだ少ない
- blind recovery は tuned 条件に依存する
- depth 5 blind recovery は未達
- depth 6 は小規模 negative control 段階

その一方で、この local rerun からは次の偏りが暫定的に見えています。

- `depth3` は random init 4 family × `4` seeds で `16/16` fit, `16/16` symbolic, `16/16` stable symbolic まで改善し、この rerun では `nonfinite_steps=0`, `nonfinite_grad_steps=0`, `nan_restarts=0`
- `depth4` は random init 4 family × `4` seeds で `14/16` fit, `14/16` symbolic, `14/16` stable symbolic。`biased`, `uniform`, `xy_biased` は `4/4` stable だが、`random_hot` は `2/4` stable に留まる
- `depth5_manual_noise12` は `4/4` stable symbolic
- `depth6_manual_noise12` は current Julia の post-recovery metric では `4/4` stable symbolic だが、新しく export した extern-style pre-recovery metric では `0/4` stable symbolic, `mean_pre_recovery_n_uncertain = 2.5` で、README の `0/4` stable symbolic と整合する
- したがって、現在の immediate gap は `depth3` ではなく `depth4 random_hot` の wider-seed tendency と untouched `depth5_random` / `depth6_random` families へ移っている

という状態です。

したがって、現在の結果を論文の成功率と直接比較するのは危険です。
現状は「構造と実験導線はあるが、統計規模と stable-success 水準は未再現」であり、
特に stable-success は post-recovery Julia metric と extern-style pre-recovery metric を分けて読む必要があります。
ここでの `nonfinite_steps` / `nan_restarts` は gradient scrub 後の recorded failure を見ている点には注意が必要です。
次の実装上の焦点候補は、まず `depth4 random_hot` を wider seed で確認し、その後に untouched `depth5_random` / `depth6_random` を埋めることです。

## 8. 現実装の strict success 判定は論文本文より厳しい

現在の `recovery_verdict` は、成功を

- `snap_status == :ok`
- `structure_match == true`
- `numerical_match == true`
- 学習中に非有限値 failure がない

の同時成立で判定しています。

論文本文は、成功時に machine epsilon squared レベルの MSE と exact symbolic recovery を述べていますが、
現実装のように

- ambiguity count
- active-node margin
- strict structure match

を明示的に JSON と summary に落としているわけではありません。

この点では、現実装は論文より厳密に診断可能ですが、
そのぶん success 判定も保守的です。

## 読み方の指針

Section 4.3 と現実装の距離感は、次のように理解するのが実態に近いです。

- 論文のアイデア: かなり取り込んでいる
- 論文の parameterization: まだ別物
- 論文の評価規模: まだ未到達
- 論文の成功率: まだ未再現
- 実装修正の次の焦点候補: `depth4 random_hot` の wider-seed 確認、その次に `depth5_random` / `depth6_random` へ広げること

つまり、現コードベースは
「Section 4.3 を Julia で研究し直すための再構成版」
であって、
「論文の 4.3 実装を忠実に移植した版」
ではありません。

## 今後の改善軸

論文への忠実度を上げるなら、優先順位が高いのは次です。

- `α + βx + γf` 型 master formula への移行
- simplex / logits の扱いを論文寄りに統一
- target suite の拡張
- seed 数と initialization sweep の大規模化
- basin-of-attraction 実験の depth 6 までの整備
- 複素数安定化の設計を論文実装に近づけること
