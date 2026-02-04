/*
 * AscendC MatMul Custom Operator - Host Implementation
 * 
 * 简化版本，使用自定义 tiling 结构
 */
#include "matmul_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* aShape = context->GetInputShape(0);
    const gert::StorageShape* bShape = context->GetInputShape(1);
    
    if (aShape == nullptr || bShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    int32_t M = aShape->GetStorageShape().GetDim(0);
    int32_t Ka = aShape->GetStorageShape().GetDim(1);
    int32_t Kb = bShape->GetStorageShape().GetDim(0);
    int32_t N = bShape->GetStorageShape().GetDim(1);
    
    if (Ka != Kb) {
        return ge::GRAPH_FAILED;
    }
    
    MatmulCustomTilingData tiling;
    tiling.set_M(M);
    tiling.set_N(N);
    tiling.set_K(Ka);
    tiling.set_usedCoreNum(1);
    
    context->SetBlockDim(1);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* aShape = context->GetInputShape(0);
    const gert::Shape* bShape = context->GetInputShape(1);
    gert::Shape* cShape = context->GetOutputShape(0);
    
    if (aShape == nullptr || bShape == nullptr || cShape == nullptr) {
        return GRAPH_FAILED;
    }
    
    cShape->SetDimNum(2);
    cShape->SetDim(0, aShape->GetDim(0));
    cShape->SetDim(1, bShape->GetDim(1));
    
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class MatmulCustom : public OpDef {
public:
    explicit MatmulCustom(const char* name) : OpDef(name)
    {
        this->Input("a").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MatmulCustom);

}  // namespace ops
