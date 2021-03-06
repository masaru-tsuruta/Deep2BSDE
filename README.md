# Deep2BSDE  
Code based on Beck et al 2019  
Beck, C., Weinan, E., & Jentzen, A. (2019). Machine learning approximation algorithms for high-dimensional fully nonlinear partial differential equations and second-order backward stochastic differential equations. Journal of Nonlinear Science, 29(4), 1563-1619.

- Deep2BSDE_4_1_original.py: Appendix A.1 Original code for TF 1.2 or 1.3
- Deep2BSDE_4_1_tf2.py: TF1向けをTF2向けに修正。単純にTF2で動作しない箇所をtf.compat.v1を用い修正。加えてfとgammaの式を修正。
- Deep2BSDE_4_3_original.py: Appendix A.3 Original code for TF 1.2 or 1.3
- Deep2BSDE_4_3_tf2.py: TF1向けをTF2向けに修正。単純にTF2で動作しない箇所をtf.compat.v1を用い修正。加えてfとgammaの式を修正。sigmaの式も修正。（tensorflow 1.15を用いた環境で実行したところ、オリジナルのコードではpaperのTable6の結果にならなかった）
- Deep2BSDE_4_3_tf2_modified.py: tf.compat.v1を使用せずTF2向けにコードを大きく修正したもの。上記と比べ若干速度向上。
- Deep2BSDE_BS_tf2.py: Black Sholes Model 1 process model(TF2対応コードで作成)
- Deep2BSDE_BS_tf2_v2.py: Black Sholes Model 10 processes model(TF2対応コードで作成)
