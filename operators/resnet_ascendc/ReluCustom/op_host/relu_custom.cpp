/**
 * AscendC ReLU - Host 端代码
 * 包含: Tiling 函数 + 算子注册
 */
#include "relu_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

// Tiling 函数: 运行在 CPU，根据输入 shape 计算分块策略
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    ReluCustomTilingData tiling;
    
    // 获取输入 shape
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t totalLength = 1;
    for (int i = 0; i < xShape->GetStorageShape().GetDimNum(); i++) {
        totalLength *= xShape->GetStorageShape().GetDim(i);
    }
    
    tiling.set_totalLength(totalLength);
    
    // 设置使用的 AI Core 数量
    context->SetBlockDim(8);
    
    // 保存 tiling 数据
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

// Shape 推导
namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context) {
    // 输出 shape = 输入 shape
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

} // namespace ge

// 算子注册
namespace ops {

class ReluCustom : public OpDef {
public:
    explicit ReluCustom(const char* name) : OpDef(name) {
        // 定义输入
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        
        // 定义输出
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        
        // 绑定推导函数
        this->SetInferShape(ge::InferShape)
             .SetInferDataType(ge::InferDataType);
        
        // 绑定 Tiling 函数
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        
        // 指定支持的硬件
        this->AICore().AddConfig("ascend910b");
    }
};

// 注册算子
OP_ADD(ReluCustom);

} // namespace ops
