#!/usr/bin/env python3
"""DivCustom 指令流水ASCII图"""

print("""
================================================================================
                    DivCustom 算子指令流水图
================================================================================

【AI Vector Core 架构】

  ┌─────────────────────────────────────────────────────────────────┐
  │                        Global Memory (GM)                          │
  │          x[8,256]           y[8,256]           z[8,256]            │
  └────────────┬──────────────────┬───────────────────┬────────────┘
               │         MTE2         │          MTE3          │
               │       (CopyIn)       │        (CopyOut)       │
               ▼                       │                        ▲
  ┌─────────────────────────────────────────────────────────────────┐
  │                      Unified Buffer (UB)                           │
  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
  │  │ inQueueX │  │ inQueueY │  │ outQueueZ│  │  tmpBuf   │   │
  │  │ (VECIN)  │  │ (VECIN)  │  │ (VECOUT) │  │ (VECCALC)│   │
  │  └────┬──────┘  └────┬──────┘  └─────┬─────┘  └───────────┘   │
  └─────────┬─────────┬───────────────┬─────────────────────────────┘
            │         │               │
            ▼         ▼               ▲
  ┌─────────────────────────────────────────────────────────────────┐
  │                       Vector Unit                                  │
  │                     Div(z, x, y, 256)                              │
  └─────────────────────────────────────────────────────────────────┘


【单个Tile的执行流水】 (Block 0 示例)

  Cycles:    0      10      20      30      40      50      60      70
            │───────│───────│───────│───────│───────│───────│───────│
            │       │       │       │       │       │       │
  Scalar:   ██████████████████████████████████████████ (42.7%)
            [loop ctrl, addr calc, queue ops]
            │       │       │       │       │       │       │
  MTE2:           █████████████████████ (20.5%)
                  [DataCopy x,y from GM -> UB]
            │       │       │       │       │       │       │
  Vector:               █ (0.8%)
                        [Div compute]
            │       │       │       │       │       │       │
  MTE3:                       ██████ (5.3%)
                              [DataCopy z from UB -> GM]
            │       │       │       │       │       │       │


【8核并行执行时序图】

  Time -->
  Block 0:  [S]███[MTE2]██████[V][MTE3]███[S]
  Block 1:    [S]███[MTE2]██████[V][MTE3]███[S]
  Block 2:      [S]███[MTE2]██████[V][MTE3]███[S]
  Block 3:        [S]███[MTE2]██████[V][MTE3]███[S]
  Block 4:          [S]███[MTE2]██████[V][MTE3]███[S]
  Block 5:            [S]███[MTE2]██████[V][MTE3]███[S]
  Block 6:              [S]███[MTE2]██████[V][MTE3]███[S]
  Block 7:                [S]███[MTE2]██████[V][MTE3]███[S]

  图例:  [S]=Scalar  [MTE2]=搬入  [V]=Vector计算  [MTE3]=搬出


【Kernel代码与流水线对应】

  ┌───────────────────────────────────────────────────────────────────┐
  │ void CopyIn(idx) {                          <- Scalar (loop)      │
  │     xLocal = inQueueX.AllocTensor();        <- Scalar (queue)     │
  │     yLocal = inQueueY.AllocTensor();        <- Scalar (queue)     │
  │     DataCopy(xLocal, xGm[idx*TILE], TILE);  <- MTE2 (GM->UB)      │
  │     DataCopy(yLocal, yGm[idx*TILE], TILE);  <- MTE2 (GM->UB)      │
  │     inQueueX.EnQue(xLocal);                 <- Scalar (queue)     │
  │     inQueueY.EnQue(yLocal);                 <- Scalar (queue)     │
  │ }                                                                  │
  ├───────────────────────────────────────────────────────────────────┤
  │ void Compute(idx) {                         <- Scalar             │
  │     xLocal = inQueueX.DeQue();              <- Scalar (queue)     │
  │     yLocal = inQueueY.DeQue();              <- Scalar (queue)     │
  │     zLocal = outQueueZ.AllocTensor();       <- Scalar (queue)     │
  │     Div(zLocal, xLocal, yLocal, TILE);      <- Vector (计算)       │
  │     outQueueZ.EnQue(zLocal);                <- Scalar (queue)     │
  │     inQueueX.FreeTensor(xLocal);            <- Scalar (queue)     │
  │     inQueueY.FreeTensor(yLocal);            <- Scalar (queue)     │
  │ }                                                                  │
  ├───────────────────────────────────────────────────────────────────┤
  │ void CopyOut(idx) {                         <- Scalar (loop)      │
  │     zLocal = outQueueZ.DeQue();             <- Scalar (queue)     │
  │     DataCopy(zGm[idx*TILE], zLocal, TILE);  <- MTE3 (UB->GM)      │
  │     outQueueZ.FreeTensor(zLocal);           <- Scalar (queue)     │
  │ }                                                                  │
  └───────────────────────────────────────────────────────────────────┘


【优化建议】

  1. Scalar占比过高 (42.7%)
     - 原因: Queue操作(Alloc/EnQue/DeQue/Free)和循环控制开销大
     - 方案: 增大TILE_SIZE从256到512或1024，减少迭代次数

  2. Vector利用率低 (0.8%)
     - 原因: 输入数据过小 (2048元素), 单次计算量不足
     - 方案: 处理更大的tensor或融合更多计算

  3. Double Buffer效果有限
     - 原因: tileNum=1, 无流水线重叠机会
     - 方案: 增加输入规模以充分利用double buffer

================================================================================
""")
