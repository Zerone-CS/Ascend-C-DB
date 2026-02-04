#pragma once

#include <cstdint>
#include <acl/acl.h>

#include "tiling/platform/platform_ascendc.h"
#include "include/matmul/block/block_scheduler_policy.h"
#include "include/matmul/block/block_mmad_builder.h"
#include "include/matmul/kernel/kernel_matmul.h"
#include "include/matmul/device/device_matmul.h"
#include "include/utils/host_utils.h"
#include "include/utils/layout_utils.h"
#include "include/utils/status_utils.h"

using namespace Act;
using namespace Act::Gemm;

namespace CubeOps {

using L1TileShape = AscendC::Shape<_128, _256, _256>;
using L0TileShape = AscendC::Shape<_128, _256, _64>;

using MatmulAType = half;
using MatmulBType = half;
using MatmulCType = half;

using LayoutA = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutC = layout::RowMajor;

using MatmulScheduler = IterateKScheduler;

using MatmulBlockMmad = Block::BlockMmadBuilder<
    MatmulAType, LayoutA, MatmulBType, LayoutB, MatmulCType, LayoutC, MatmulCType, LayoutC,
    L1TileShape, L0TileShape, MatmulScheduler, MatmulMultiBlockWithLayout<>>;

using MatmulBlockEpilogue = Block::BlockEpilogueEmpty;
using MatmulProblemShape = MatmulShape;

using CubeMatmulKernel = Kernel::KernelMatmul<MatmulProblemShape, MatmulBlockMmad, MatmulBlockEpilogue, MatmulScheduler>;
using CubeMatmulArguments = typename CubeMatmulKernel::Arguments;
using CubeDeviceMatmul = Device::DeviceMatmul<CubeMatmulKernel>;

inline void CubeMatMul(uint8_t* A, uint8_t* B, uint8_t* C,
                       int64_t M, int64_t N, int64_t K,
                       void* stream = nullptr) {
    uint8_t* workspaceDevice;
    MatmulShape shape{M, N, K, 1};

    CubeMatmulArguments args = {
        shape,
        {A, B, C, nullptr},
        {}
    };

    CubeDeviceMatmul mm;
    size_t workspaceSize = CubeDeviceMatmul::GetWorkspaceSize(args);

    aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (mm.CanImplement(args)) {
        mm.InitParams(args, workspaceDevice);
        mm();
    }

    aclrtFree(workspaceDevice);
}

} // namespace CubeOps
