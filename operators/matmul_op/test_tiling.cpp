/*
 * Test tiling computation to debug the issue
 */
#include <iostream>
#include <iomanip>
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

int main() {
    std::cout << "=== Testing Matmul Tiling ===\n";
    
    // Use actual platform info from a simulated context
    // For now, just test the TCubeTiling layout
    
    optiling::TCubeTiling cubeTiling;
    
    // Check the data size
    std::cout << "\noptiling::TCubeTiling::GetDataSize() = " << cubeTiling.GetDataSize() << " bytes\n";
    std::cout << "Expected (50 * 4) = " << 50 * sizeof(int32_t) << " bytes\n";
    
    // Set some test values
    cubeTiling.set_usedCoreNum(1);
    cubeTiling.set_M(16);
    cubeTiling.set_N(16);
    cubeTiling.set_Ka(16);
    cubeTiling.set_Kb(16);
    cubeTiling.set_singleCoreM(16);
    cubeTiling.set_singleCoreN(16);
    cubeTiling.set_singleCoreK(16);
    cubeTiling.set_baseM(16);
    cubeTiling.set_baseN(16);
    cubeTiling.set_baseK(16);
    
    // Serialize to buffer
    std::vector<uint8_t> buffer(1024, 0);
    cubeTiling.SaveToBuffer(buffer.data(), buffer.size());
    
    // Print the serialized data
    std::cout << "\nSerialized tiling data (first 60 bytes as int32):" << std::endl;
    int32_t* intPtr = reinterpret_cast<int32_t*>(buffer.data());
    for (int i = 0; i < 15; i++) {
        std::cout << "  [" << std::setw(2) << i << "] " 
                  << std::setw(10) << intPtr[i] << std::endl;
    }
    
    return 0;
}
