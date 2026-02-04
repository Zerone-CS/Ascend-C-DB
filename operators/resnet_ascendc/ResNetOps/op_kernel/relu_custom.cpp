/**
 * AscendC ReLU Kernel - 完整算子工程示例
 * 
 * AscendC 核心概念:
 * - TPipe: 流水线管理器
 * - TQue: 双缓冲队列，实现计算/搬运重叠
 * - GlobalTensor: 全局内存绑定
 * - LocalTensor: UB (Unified Buffer) 上的局部张量
 * - DataCopy: GM <-> UB 数据搬运
 * - Relu: 向量计算 API
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // 双缓冲，保证流水线重叠
constexpr int32_t TILE_SIZE = 256; // 每次处理 256 个元素

class KernelRelu {
public:
    __aicore__ inline KernelRelu() {}

    // Init: 初始化全局内存绑定和缓冲区
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        // 多核分块: 每个 AI Core 处理一部分数据
        uint32_t blockNum = GetBlockNum();   // 总核数
        uint32_t blockIdx = GetBlockIdx();   // 当前核索引
        uint32_t lengthPerBlock = (totalLength + blockNum - 1) / blockNum;
        uint32_t startOffset = blockIdx * lengthPerBlock;
        this->processLength = (startOffset + lengthPerBlock > totalLength)
                              ? (totalLength - startOffset) : lengthPerBlock;
        if (startOffset >= totalLength) this->processLength = 0;

        // 绑定全局内存 (GM)
        xGm.SetGlobalBuffer((__gm__ float*)x + startOffset, processLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + startOffset, processLength);

        // 初始化 UB 缓冲区
        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_SIZE * sizeof(float));
    }

    // Process: 主处理循环
    __aicore__ inline void Process() {
        uint32_t loopCount = (processLength + TILE_SIZE - 1) / TILE_SIZE;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t len = (i == loopCount - 1) 
                           ? (processLength - i * TILE_SIZE) : TILE_SIZE;
            CopyIn(i, len);
            Compute(len);
            CopyOut(i, len);
        }
    }

private:
    // CopyIn: GM -> UB
    __aicore__ inline void CopyIn(uint32_t idx, uint32_t len) {
        LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        // DataCopy 要求 32 字节对齐 (float32: 8 个元素)
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        DataCopy(xLocal, xGm[idx * TILE_SIZE], alignedLen);
        inQueue.EnQue(xLocal);
    }

    // Compute: 核心计算
    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<float> xLocal = inQueue.DeQue<float>();
        LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        
        // ======== AscendC 核心 API ========
        // Relu: y = max(0, x)
        Relu(yLocal, xLocal, alignedLen);
        // ==================================
        
        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    // CopyOut: UB -> GM
    __aicore__ inline void CopyOut(uint32_t idx, uint32_t len) {
        LocalTensor<float> yLocal = outQueue.DeQue<float>();
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        DataCopy(yGm[idx * TILE_SIZE], yLocal, alignedLen);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;                              // 流水线管理器
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;   // 输入队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue; // 输出队列
    GlobalTensor<float> xGm, yGm;            // 全局内存绑定
    uint32_t processLength;
};

// 入口函数: extern "C" __global__ __aicore__ 是必须的
extern "C" __global__ __aicore__ void relu_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelRelu op;
    op.Init(x, y, tilingData.totalLength);
    op.Process();
}
