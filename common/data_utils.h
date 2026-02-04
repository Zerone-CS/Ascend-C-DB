/**
 * data_utils.h - 通用数据读写工具
 * 
 * Host侧使用，用于加载/保存测试数据
 */
#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <iostream>

namespace ascendc {
namespace utils {

/**
 * 从二进制文件读取数据
 */
template<typename T>
bool ReadFile(const std::string& path, T* data, size_t count) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(data), count * sizeof(T));
    if (file.gcount() != static_cast<std::streamsize>(count * sizeof(T))) {
        std::cerr << "Read size mismatch: expected " << count * sizeof(T) 
                  << ", got " << file.gcount() << std::endl;
        return false;
    }
    return true;
}

/**
 * 写入数据到二进制文件
 */
template<typename T>
bool WriteFile(const std::string& path, const T* data, size_t count) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to create file: " << path << std::endl;
        return false;
    }
    file.write(reinterpret_cast<const char*>(data), count * sizeof(T));
    return true;
}

/**
 * 读取文件到vector
 */
template<typename T>
std::vector<T> ReadFileToVector(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<T> data(fileSize / sizeof(T));
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    return data;
}

/**
 * 比较两个数组，返回最大绝对误差
 */
template<typename T>
double CompareArrays(const T* a, const T* b, size_t count) {
    double maxDiff = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (diff > maxDiff) {
            maxDiff = diff;
        }
    }
    return maxDiff;
}

/**
 * 打印数组前n个元素
 */
template<typename T>
void PrintArray(const T* data, size_t count, size_t maxPrint = 10) {
    size_t n = std::min(count, maxPrint);
    std::cout << "[";
    for (size_t i = 0; i < n; ++i) {
        std::cout << data[i];
        if (i < n - 1) std::cout << ", ";
    }
    if (count > maxPrint) std::cout << ", ...";
    std::cout << "]" << std::endl;
}

/**
 * 计算数据大小 (字节)
 */
template<typename T>
size_t DataSize(size_t count) {
    return count * sizeof(T);
}

/**
 * 32字节对齐
 */
inline size_t Align32(size_t size) {
    return (size + 31) & ~31;
}

/**
 * 元素数对齐 (float: 8个元素 = 32字节)
 */
template<typename T>
size_t AlignCount(size_t count) {
    size_t align = 32 / sizeof(T);
    return (count + align - 1) / align * align;
}

}  // namespace utils
}  // namespace ascendc

#endif  // DATA_UTILS_H
