#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn_relu_custom.h"

namespace py = pybind11;

static bool g_acl_initialized = false;
static aclrtStream g_stream = nullptr;

static void ensure_init() {
    if (!g_acl_initialized) {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclrtCreateStream(&g_stream);
        g_acl_initialized = true;
    }
}

static void cleanup() {
    if (g_acl_initialized) {
        if (g_stream) {
            aclrtDestroyStream(g_stream);
            g_stream = nullptr;
        }
        aclrtResetDevice(0);
        aclFinalize();
        g_acl_initialized = false;
    }
}

py::array_t<float> relu_forward(py::array_t<float, py::array::c_style | py::array::forcecast> input) {
    ensure_init();
    
    auto buf = input.request();
    int64_t N = 1;
    std::vector<ssize_t> shape;
    for (ssize_t i = 0; i < buf.ndim; i++) {
        N *= buf.shape[i];
        shape.push_back(buf.shape[i]);
    }
    
    if (N % (8 * 256) != 0) {
        throw std::runtime_error("Total element count must be divisible by 2048");
    }
    
    size_t size = N * sizeof(float);
    float* hostX = static_cast<float*>(buf.ptr);
    
    auto result = py::array_t<float>(shape);
    auto result_buf = result.request();
    float* hostY = static_cast<float*>(result_buf.ptr);
    
    void *devX, *devY;
    aclrtMalloc(&devX, size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devY, size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(devX, size, hostX, size, ACL_MEMCPY_HOST_TO_DEVICE);
    
    int64_t tensor_shape[] = {N};
    int64_t strides[] = {1};
    
    aclTensor *xTensor = aclCreateTensor(tensor_shape, 1, ACL_FLOAT, strides, 0, 
                                          ACL_FORMAT_ND, tensor_shape, 1, devX);
    aclTensor *yTensor = aclCreateTensor(tensor_shape, 1, ACL_FLOAT, strides, 0,
                                          ACL_FORMAT_ND, tensor_shape, 1, devY);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnReluCustomGetWorkspaceSize(xTensor, yTensor, &workspaceSize, &executor);
    
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    
    aclnnReluCustom(workspace, workspaceSize, executor, g_stream);
    aclrtSynchronizeStream(g_stream);
    
    aclrtMemcpy(hostY, size, devY, size, ACL_MEMCPY_DEVICE_TO_HOST);
    
    aclDestroyTensor(xTensor);
    aclDestroyTensor(yTensor);
    if (workspace) aclrtFree(workspace);
    aclrtFree(devX);
    aclrtFree(devY);
    
    return result;
}

PYBIND11_MODULE(relu_custom_npu, m) {
    m.doc() = "AscendC ReluCustom operator for NPU";
    m.def("relu", &relu_forward, "Execute ReLU on NPU using AscendC custom kernel");
    m.def("finalize", &cleanup, "Clean up ACL resources");
}
