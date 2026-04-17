# Section 4.3 の一次情報一覧

このページは、論文「All elementary functions from a single operator」の
Section 4.3 をできるだけ忠実に再現するために参照すべき一次情報を整理したものです。

優先順位は、

1. 論文本文
2. Supplementary Information
3. 凍結スナップショット
4. EML toolkit の training / verification 実装

の順です。

## 最優先

- arXiv 論文ページ  
  <https://arxiv.org/abs/2603.21852>

- 論文 PDF  
  <https://arxiv.org/pdf/2603.21852>

- Supplementary Information PDF  
  <https://arxiv.org/src/2603.21852v2/anc/SupplementaryInformation.pdf>

- Zenodo 凍結スナップショット  
  <https://zenodo.org/records/19183008>

- GitHub リポジトリ  
  <https://github.com/VA00/SymbolicRegressionPackage>

## EML toolkit 相当

Zenodo と GitHub の記述から、`EML toolkit` 相当の再現資料は
`VA00/SymbolicRegressionPackage` リポジトリ内の `EML_toolkit/` です。

- `EML_toolkit/` 概要  
  <https://github.com/VA00/SymbolicRegressionPackage/tree/master/EML_toolkit>

- `EML_toolkit/README.md`  
  <https://raw.githubusercontent.com/VA00/SymbolicRegressionPackage/master/EML_toolkit/README.md>

- `EmL_training/`  
  <https://github.com/VA00/SymbolicRegressionPackage/tree/master/EML_toolkit/EmL_training>

- `Log_fit.nb`  
  <https://github.com/VA00/SymbolicRegressionPackage/blob/master/EML_toolkit/EmL_training/Log_fit.nb>

- `PyTorch_v16_final/`  
  <https://github.com/VA00/SymbolicRegressionPackage/tree/master/EML_toolkit/EmL_training/PyTorch_v16_final>

- `PyTorch_v16_final/README.md`  
  <https://github.com/VA00/SymbolicRegressionPackage/blob/master/EML_toolkit/EmL_training/PyTorch_v16_final/README.md>

- `depth_2_to_6_headless.sh`  
  <https://github.com/VA00/SymbolicRegressionPackage/blob/master/EML_toolkit/EmL_training/PyTorch_v16_final/depth_2_to_6_headless.sh>

- `tree_prototype_torch_v16_final.py`  
  <https://github.com/VA00/SymbolicRegressionPackage/blob/master/EML_toolkit/EmL_training/PyTorch_v16_final/tree_prototype_torch_v16_final.py>

- `requirements.txt`  
  <https://github.com/VA00/SymbolicRegressionPackage/blob/master/EML_toolkit/EmL_training/PyTorch_v16_final/requirements.txt>

- `EmL_verification/`  
  <https://github.com/VA00/SymbolicRegressionPackage/tree/master/EML_toolkit/EmL_verification>

- `EmL_compiler/`  
  <https://github.com/VA00/SymbolicRegressionPackage/tree/master/EML_toolkit/EmL_compiler>

- `EmL_figures/`  
  <https://github.com/VA00/SymbolicRegressionPackage/tree/master/EML_toolkit/EmL_figures>

- `EmL_recognizer/`  
  <https://github.com/VA00/SymbolicRegressionPackage/tree/master/EML_toolkit/EmL_recognizer>

## 周辺の基盤コード

- `SymbolicRegression.m`  
  <https://github.com/VA00/SymbolicRegressionPackage/blob/master/SymbolicRegression.m>

- `SymbolicRegressionPackage_Examples.nb`  
  <https://github.com/VA00/SymbolicRegressionPackage/blob/master/SymbolicRegressionPackage_Examples.nb>

- `v1.0` タグ  
  <https://github.com/VA00/SymbolicRegressionPackage/tree/v1.0>

## 補助的な参照先

- 著者ホームページ  
  <https://th.if.uj.edu.pl/~odrzywolek/>

## 読む順番

実装忠実化のためには、次の順で読むのが効率的です。

1. arXiv 本文
2. Supplementary Information
3. Zenodo snapshot
4. `EML_toolkit/README.md`
5. `Log_fit.nb`
6. `PyTorch_v16_final/README.md`
7. `tree_prototype_torch_v16_final.py`
8. `depth_2_to_6_headless.sh`
9. `EmL_verification/`

## この一覧の位置づけ

この一覧は「Section 4.3 の差分整理」を補うための資料ページです。
差分そのものは [Section 4.3 との差分整理](section-4-3-gap.md) を参照してください。
