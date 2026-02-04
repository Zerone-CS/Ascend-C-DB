/**
 * kernel_utils.h - Kernel侧常用宏和工具
 * 
 * Kernel侧使用，包含常用宏和辅助函数
 */
#ifndef KERNEL_UTILS_H
#define KERNEL_UTILS_H

#include "kernel_operator.h"

namespace AscendC {

// ============ 常用常量 ============

constexpr int32_t BUFFER_NUM = 2;  // Double Buffer
constexpr int32_t DEFAULT_TILE_SIZE = 256;  // 默认分块大小

// ============ 对齐宏 ============

// 32字节对齐的元素数 (float)
#define ALIGN_FLOAT(count) (((count) + 7) / 8 * 8)

// 32字节对齐的元素数 (half)
#define ALIGN_HALF(count) (((count) + 15) / 16 * 16)

// 通用对齐宏
#define ALIGN_COUNT(count, align) (((count) + (align) - 1) / (align) * (align))

// ============ Tiling 计算 ============

// 计算分块数
#define CALC_TILE_NUM(total, tile_size) (((total) + (tile_size) - 1) / (tile_size))

// 计算最后一块大小
#define CALC_LAST_TILE_SIZE(total, tile_size) \
    (((total) % (tile_size)) == 0 ? (tile_size) : ((total) % (tile_size)))

// 计算当前块大小
#define CALC_CURRENT_TILE_SIZE(idx, tile_num, total, tile_size) \
    (((idx) == (tile_num) - 1) ? CALC_LAST_TILE_SIZE(total, tile_size) : (tile_size))

// ============ 多核负载均衡 ============

/**
 * 计算当前核的工作范围
 * 
 * @param total 总元素数
 * @param startOffset 输出: 起始偏移
 * @param processLength 输出: 处理长度
 */
__aicore__ inline void CalcWorkRange(uint32_t total, 
                                      uint32_t& startOffset, 
                                      uint32_t& processLength) {
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    
    uint32_t baseLength = total / blockNum;
    uint32_t remainder = total % blockNum;
    
    if (blockIdx < remainder) {
        processLength = baseLength + 1;
        startOffset = blockIdx * processLength;
    } else {
        processLength = baseLength;
        startOffset = remainder * (baseLength + 1) + (blockIdx - remainder) * baseLength;
    }
    
    // 边界检查
    if (startOffset >= total) {
        processLength = 0;
    }
}

// ============ Softmax 辅助 ============

/**
 * Softmax数值稳定计算
 * y = exp(x - max(x)) / sum(exp(x - max(x)))
 */
template<typename T>
__aicore__ inline void StableSoftmax(LocalTensor<T>& y,
                                      LocalTensor<T>& x,
                                      LocalTensor<T>& work,
                                      uint32_t count) {
    // Step 1: max = ReduceMax(x)
    ReduceMax(work, x, work, count);
    T maxVal = work.GetValue(0);
    
    // Step 2: y = x - max
    Adds(y, x, -maxVal, count);
    
    // Step 3: y = exp(y)
    Exp(y, y, count);
    
    // Step 4: sum = ReduceSum(y)
    ReduceSum(work, y, work, count);
    T sumVal = work.GetValue(0);
    
    // Step 5: y = y / sum
    Muls(y, y, static_cast<T>(1) / sumVal, count);
}

}  // namespace AscendC

#endif  // KERNEL_UTILS_H
