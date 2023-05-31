GEMM优化

1. 基于算法优化：借助辅助矩阵，减少矩阵乘法次数

   1. strassen(O^2.83)
   2. coppersmith-winograd(O^2.3)
2. 基于软件优化：优化访存，增加cache hit rate，并行计算提速

   1. 改变循环顺序(ijk)-> (ikj)，在一千阶矩阵下，GEMM提升五倍
   2. 循环拆分blocking
   3. SIMD向量化指令


todo..

1. LU分解
2. MONTE CARLO
3. openMP PSO优化
