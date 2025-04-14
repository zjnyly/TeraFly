project(transformer)

find_package(Threads REQUIRED)
include_directories(/tools/Xilinx/Vitis_HLS/2023.2/include/)
include_directories(/tools/Xilinx/Vitis_HLS/2023.2/lnx64/tools/fpo_v7_1)
link_directories(/tools/Xilinx/Vitis_HLS/2023.2/lnx64/tools/fpo_v7_1)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#link_directories(${XILINX_XRT}/lib)
#file(GLOB XILINX_LIBRARIES "/tools/Xilinx/Vitis_HLS/2023.2/lib/lnx64.o/*.so")
file(GLOB XILINX_LIBRARIES "/tools/Xilinx/Vitis_HLS/2023.2/lnx64/tools/fpo_v7_1/*.so")

add_executable(loopLynx loopLynx_tb.cpp loopLynx.cpp loopLynx.h params.h)
#target_compile_options(loopLynx PRIVATE -O3)

target_compile_options(loopLynx PRIVATE 
    -O3                    # 最高优化等级
    -march=native          # 根据具体架构启用指令集
    -funroll-loops         # 循环展开
    -fopenmp               # 开启 OpenMP 支持
    -ffast-math            # 快速数学计算
    -fno-exceptions        # 禁用异常处理，提升性能
    -DNDEBUG               # 禁用断言
)

target_link_libraries(loopLynx ${XILINX_LIBRARIES} Threads::Threads)

add_compile_definitions(HLS_NO_XIL_FPO_LIB)