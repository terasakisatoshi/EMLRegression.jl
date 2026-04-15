# 論文の要点と解説

## 論文の主張

論文「All elementary functions from a single operator」の中心的な主張は、定数 `1` と 1 つの 2 項演算子

```math
\mathrm{eml}(x, y) = e^x - \ln(y)
```

だけで、科学技術計算でよく使う多くの初等関数を表現できる、というものです。

著者たちはこれを、ブール代数における NAND や NOR のような「普遍演算子」の連続版として位置づけています。

## 論文全体の流れ

論文は大まかに次の 4 本立てです。

1. 単一演算子による表現可能性の議論
2. 候補演算子の探索と EML の発見
3. EML を使ったコンパイルや回路表現
4. 学習可能な EML 木による symbolic regression

このリポジトリが今扱っているのは、主に 4 です。

## Section 4.3 の意味

Section 4.3 では、式探索を完全な離散探索としてではなく、「学習可能な master tree」の連続最適化として扱います。

発想は次の通りです。

- 深さを固定した木構造を用意する
- 各ノードや端点の選択を連続パラメータで表す
- 勾配法で目標関数に近づける
- 最後に 0/1 に近い表現へ硬化させる
- snapping により離散式として読み出す

このアプローチの面白さは、離散的な symbolic regression とニューラルネット的な連続最適化の中間にある点です。

## なぜ複素数を使うのか

EML は `log` を含むため、実数だけで扱うと定義域制約が厳しくなります。論文でも複素数中間値が重要です。

このリポジトリでも、内部表現は `ComplexF64` を前提にしています。これは便利ですが、同時に次の難しさも増やします。

- branch cut の扱い
- `log(0)` 近傍の不安定性
- 指数関数の急激な発散
- 数値的には一致していても symbolic には別物である可能性

## 現在の実装との対応

現在のリポジトリは、Section 4.3 を次のように切り出しています。

- EML 演算子: 実装済み
- 完全二分 master tree: 実装済み
- node-wise な左右選択 logits: 実装済み
- `Adam` + autodiff による学習ループ: 実装済み
- hardening と complexity penalty: 実装済み
- ターゲット関数群: 実装済み
- 実験設定と CLI: 実装済み
- snapping と blind recovery の枠組み: 実装済み
- 論文どおりの安定化と recovery 基準: 未完成
- 論文レベルの成功率: 未達

つまり、研究再現の「導線」と最小限の学習系は揃っているが、「性能の本丸」と評価基準の厳密さはこれからです。

## 論文との主なギャップ

現時点で目立つギャップは次のとおりです。

- strict success を測れるようにはなったが、論文レベルの回復率には届いていない
  現在の `success` は snapped tree の `snap_status == :ok`、`structure_match == true`、`numerical_match == true` を要求します。ただし、この stricter metric を導入した結果、depth 3/4 はなお未達であることが明確になりました。
- 深さ 3 は tuned sweep で回復できるが、デフォルト設定ではまだ弱い
  `must_pass_depth3_sweep-cooler_hardening` では strict recovery まで到達していますが、baseline 相当の設定ではまだ不安定です。論文のような systematic な成功率評価はこれからです。
- 深さ 4 は tuned schedule と初期化で前進したが、論文レベルの systematic experiments には届いていない
  `challenge_depth4_sweep-longer_cool` は `2/2` の strict recovery です。さらに `challenge_depth4_init_sweep` では `8` seeds の比較を行い、`zero_bias_to_inputs` が `6/8`、`small_gaussian` が `5/8`、`margin_biased` が `3/8` でした。depth 4 で blind recovery 自体は確認できていますが、論文のような大規模比較にはまだ遠いです。
- 実験規模がかなり小さい
  現在は varied seeds と initialization strategies の sweep を入れ始めましたが、規模はなお小さいです。論文では 1000 超の runs が報告されています。
- 深さ 5/6 の検証がない
  リポジトリのターゲットと設定は深さ 2-4 までです。論文は depth 5 で 1% 未満、depth 6 で `0/448` まで評価しています。
- basin-of-attraction の再現実験がない
  近い実験は入りました。`challenge_depth4_basin_sweep` では target tree に対応する logits 初期値へ Gaussian noise を加え、`depth4_nested` で `σ=0.05`, `0.10`, `0.25` の各設定が `8/8` で strict recovery でした。ただし、論文の depth 5/6 まで含む basin-of-attraction 結果そのものはまだ未再現です。
- 数値安定化は入ったが、論文の安定化戦略とはまだ差がある
  現実装も output clamp と nonfinite flagging を学習ループに入れていますが、論文が強調する複素数の実部・虚部 inspection や clamping 戦略を完全には再現していません。
- 最適化の細部は論文実装と一致していない
  現在の学習ループは `Zygote` による autodiff と `Optimisers.Adam` を使っていますが、論文の PyTorch `complex128` 実装と同一条件ではありません。特に安定化処理と収束挙動の差は残っています。
- ターゲット集合が小さい
  現在の paper-aligned suite は curated な 4 target に限られています。論文は composed EML 由来の target 群で systematic に比較しています。
- 実装全体がまだ scaffold 段階
  現在の実装は Section 4.3 の再現基盤としては読めますが、論文レベルの optimization quality と recovery rate には達していません。

なお、以前の差分整理で挙げられがちだった「有限差分で勾配を計算している」という点は、現行 `HEAD` には当たりません。現在の学習ループは `Zygote` による autodiff を使っています。

また、現在の raw JSON / summary では strict recovery と numerical match を分けて観測できるようにしてあります。したがって、depth 3 のような「数値は合うが snapped structure は外す」ケースを集計上で切り分けられます。

## この実装をどう読むべきか

このコードベースは、現時点では次の用途に向いています。

- 論文の 4.3 節の構造を Julia で追う
- 実験設定、実行、集計の流れを理解する
- どこを強化すれば論文に近づくかを見積もる

逆に、次の期待にはまだ応えません。

- 論文と同等の recovery 成功率
- 深さごとの本格比較
- EML 自体の発見過程の再現

## 読み進め方のおすすめ

1. [使い方マニュアル](manual.md) で実験を 1 回動かす
2. [学習方法](training.md) で現在の簡略化点を確認する
3. [`run_experiment`](@ref) からコードを追う
4. `scripts/` と `experiments/configs/` を見て実験の入口を理解する

この順で読むと、論文のアイデアと現在のコードの距離感が見えやすくなります。
