#ifndef DATA_UTILS_H
#define DATA_UTILS_H
#include <iostream>
#include <fstream>
#include <cstring>

extern bool ReadFile(const std::string &filePath, size_t fileSize, void *buffer, size_t bufferSize)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file " << filePath << std::endl;
        return false;
    }
    file.read(static_cast<char *>(buffer), fileSize);
    file.close();
    return true;
}

extern bool WriteFile(const std::string &filePath, void *buffer, size_t size)
{
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file " << filePath << std::endl;
        return false;
    }
    file.write(static_cast<char *>(buffer), size);
    file.close();
    return true;
}

#endif
