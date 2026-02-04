/**
 * Embedding Lookup - AscendC 实现
 * 根据 token ID 查找 embedding 向量
 * output[i] = embedding_table[token_ids[i]]
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelEmbedding {
public:
    __aicore__ inline KernelEmbedding() {}
    
    __aicore__ inline void Init(GM_ADDR tokenIds, GM_ADDR embTable, GM_ADDR output,
                                 uint32_t seqLen, uint32_t vocabSize, uint32_t hiddenDim) {
        this->seqLen = seqLen;
        this->vocabSize = vocabSize;
        this->hiddenDim = hiddenDim;
        uint32_t alignedHiddenDim = ((hiddenDim + 7) / 8) * 8;
        this->alignedHiddenDim = alignedHiddenDim;
        
        tokenIdsGm.SetGlobalBuffer((__gm__ int32_t*)tokenIds, seqLen);
        embTableGm.SetGlobalBuffer((__gm__ float*)embTable, vocabSize * hiddenDim);
        outputGm.SetGlobalBuffer((__gm__ float*)output, seqLen * hiddenDim);
        
        pipe.InitBuffer(outQueue, BUFFER_NUM, alignedHiddenDim * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < seqLen; i++) {
            int32_t tokenId = tokenIdsGm.GetValue(i);
            if (tokenId >= 0 && tokenId < (int32_t)vocabSize) {
                LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
                DataCopy(outLocal, embTableGm[tokenId * hiddenDim], alignedHiddenDim);
                outQueue.EnQue(outLocal);
                outLocal = outQueue.DeQue<float>();
                DataCopy(outputGm[i * hiddenDim], outLocal, alignedHiddenDim);
                outQueue.FreeTensor(outLocal);
            }
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<int32_t> tokenIdsGm;
    GlobalTensor<float> embTableGm, outputGm;
    uint32_t seqLen, vocabSize, hiddenDim, alignedHiddenDim;
};

extern "C" __global__ __aicore__ void embedding_custom(GM_ADDR tokenIds, GM_ADDR embTable, GM_ADDR output,
                                                        GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelEmbedding op;
    op.Init(tokenIds, embTable, output, tiling_data.seqLen, tiling_data.vocabSize, tiling_data.hiddenDim);
    op.Process();
}
