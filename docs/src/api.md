# API リファレンス

このページでは、実験を読むうえで重要な公開 API をまとめます。

## 中核演算子

```@docs
eml
```

## 木構造

```@docs
MasterTree
build_master_tree
EMLTreeLayer
```

## 数値安定性

```@docs
FailureReason
StabilityReport
inspect_complex_values
finite_or_flag
clamp_complex_magnitude
```

## 学習

```@docs
TrainConfig
TrainingResult
run_training
```

## ターゲット関数

```@docs
TargetSpec
get_target
sample_domain
evaluate_target
```

## snapping と式出力

```@docs
RecoveredTree
snap_logits
snap_model
search_recovered_tree
refine_recovered_tree
evaluate_recovered
formula_string
```

## recovery 判定

```@docs
RecoveryVerdict
recovery_verdict
numerical_match
run_experiment
```
