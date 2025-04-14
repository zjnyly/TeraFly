g++ -o loopLynx host.cpp -I/opt/xilinx/xrt/include -I/tools/Xilinx/Vivado/2023.2/include -I/tools/Xilinx/Vitis_HLS/2023.2/include/  -O3  -std=c++1y -I. -fmessage-length=0 -L/opt/xilinx/xrt/lib -pthread -lOpenCL -lxilinxopencl -lxrt_core -lxrt_coreutil -lrt -lstdc++
./loopLynx ./build_dir.hw.xilinx_u50lv_gen3x4_xdma_2_202010_1/loopLynx.xclbin
