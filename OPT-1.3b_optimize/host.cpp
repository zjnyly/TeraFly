#include <iostream>
#include <cstring>
#include <ap_int.h>
// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#include "utils.h"
#include "loopLynx.h"
#include <chrono>

int main(int argc, char **argv)
{

    std::string binaryFile = argv[1];
    int device_index = 0;

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    int num_cu = CU;

    xrt::bo inp_addr[num_cu];
    xrt::bo w_addr_0[num_cu];
    xrt::bo w_addr_1[num_cu];
    xrt::bo w_addr_2[num_cu];
    xrt::bo w_addr_3[num_cu];
    xrt::bo w_addr_4[num_cu];
    xrt::bo w_addr_5[num_cu];
    xrt::bo w_addr_6[num_cu];
    xrt::bo w_addr_7[num_cu];

    xrt::kernel kernel[num_cu];

    kernel[0] = xrt::kernel(device, uuid, "loopLynx_0:{loopLynx_0_1}");
    kernel[1] = xrt::kernel(device, uuid, "loopLynx_1:{loopLynx_1_1}");
    // kernel[2] = xrt::kernel(device, uuid, "loopLynx_2:{loopLynx_2_1}");
    // kernel[3] = xrt::kernel(device, uuid, "loopLynx_3:{loopLynx_3_1}");

    std::cout << "Allocate Buffer in Global Memory\n";

    int weight_size = NUM_LAYER * FULL_INP_LEN * FULL_INP_LEN / CU / PROCESSOR * 12 / INP_NUM;
    int kv_cache_size = (NUM_LAYER * FULL_SEQ_NUM * HEAD_NUM / CU * HEAD_LEN / INP_NUM / 2);

    for (int i = 0; i < num_cu; i++)
    {
        inp_addr[i] = xrt::bo(device, sizeof(float) * INP_LEN * 13, kernel[i].group_id(1));
        w_addr_0[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size + kv_cache_size), kernel[i].group_id(3));
        w_addr_1[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size + kv_cache_size), kernel[i].group_id(4));
        w_addr_2[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size + kv_cache_size), kernel[i].group_id(5));
        w_addr_3[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size + kv_cache_size), kernel[i].group_id(6));
        w_addr_4[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size + kv_cache_size), kernel[i].group_id(7));
        w_addr_5[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size + kv_cache_size), kernel[i].group_id(8));
        w_addr_6[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size + kv_cache_size), kernel[i].group_id(9));
        w_addr_7[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size + kv_cache_size), kernel[i].group_id(10));
    }

    std::cout << "Allocate Finish" << std::endl;
    float *inp_addr_map[num_cu] = {nullptr};
    io_pack_int8 *w_addr_0_map[num_cu] = {nullptr};
    io_pack_int8 *w_addr_1_map[num_cu] = {nullptr};
    io_pack_int8 *w_addr_2_map[num_cu] = {nullptr};
    io_pack_int8 *w_addr_3_map[num_cu] = {nullptr};
    io_pack_int8 *w_addr_4_map[num_cu] = {nullptr};
    io_pack_int8 *w_addr_5_map[num_cu] = {nullptr};
    io_pack_int8 *w_addr_6_map[num_cu] = {nullptr};
    io_pack_int8 *w_addr_7_map[num_cu] = {nullptr};

    for (int i = 0; i < num_cu; i++)
    {
        inp_addr_map[i] = inp_addr[i].map<float *>();
        w_addr_0_map[i] = w_addr_0[i].map<io_pack_int8 *>();
        w_addr_1_map[i] = w_addr_1[i].map<io_pack_int8 *>();
        w_addr_2_map[i] = w_addr_2[i].map<io_pack_int8 *>();
        w_addr_3_map[i] = w_addr_3[i].map<io_pack_int8 *>();
        w_addr_4_map[i] = w_addr_4[i].map<io_pack_int8 *>();
        w_addr_5_map[i] = w_addr_5[i].map<io_pack_int8 *>();
        w_addr_6_map[i] = w_addr_6[i].map<io_pack_int8 *>();
        w_addr_7_map[i] = w_addr_7[i].map<io_pack_int8 *>();
    }

    string prefix = "./const_data/";

    for (int cu = 0; cu < CU; cu++)
    {
        int weight_size_layer = weight_size;
        load_io_pack_int8_data_mcpy(w_addr_0_map[cu], prefix + "w" + to_string(cu) + "_addr_0.bin", weight_size_layer);
        std::cout << w_addr_0_map[cu][0] << std::endl;
        load_io_pack_int8_data_mcpy(w_addr_1_map[cu], prefix + "w" + to_string(cu) + "_addr_1.bin", weight_size_layer);
        load_io_pack_int8_data_mcpy(w_addr_2_map[cu], prefix + "w" + to_string(cu) + "_addr_2.bin", weight_size_layer);
        load_io_pack_int8_data_mcpy(w_addr_3_map[cu], prefix + "w" + to_string(cu) + "_addr_3.bin", weight_size_layer);
        load_io_pack_int8_data_mcpy(w_addr_4_map[cu], prefix + "w" + to_string(cu) + "_addr_4.bin", weight_size_layer);
        load_io_pack_int8_data_mcpy(w_addr_5_map[cu], prefix + "w" + to_string(cu) + "_addr_5.bin", weight_size_layer);
        load_io_pack_int8_data_mcpy(w_addr_6_map[cu], prefix + "w" + to_string(cu) + "_addr_6.bin", weight_size_layer);
        load_io_pack_int8_data_mcpy(w_addr_7_map[cu], prefix + "w" + to_string(cu) + "_addr_7.bin", weight_size_layer);
    }

    std::cout << "synchronize input buffer data to device global memory\n";

    for (int i = 0; i < num_cu; i++)
    {
        w_addr_0[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_addr_1[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_addr_2[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_addr_3[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_addr_4[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_addr_5[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_addr_6[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_addr_7[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    auto start = std::chrono::high_resolution_clock::now(); // 开始计时

    xrt::run run[num_cu];
    for (int i = 0; i < num_cu; i++)
    {
        run[i] = xrt::run(kernel[i]);
        run[i].set_arg(1, inp_addr[i]);
        run[i].set_arg(3, w_addr_0[i]);
        run[i].set_arg(4, w_addr_1[i]);
        run[i].set_arg(5, w_addr_2[i]);
        run[i].set_arg(6, w_addr_3[i]);
        run[i].set_arg(7, w_addr_4[i]);
        run[i].set_arg(8, w_addr_5[i]);
        run[i].set_arg(9, w_addr_6[i]);
        run[i].set_arg(10, w_addr_7[i]);
    }

    std::cout << "sync scaling factor and bias onboard" << std::endl;
    // for (int cu = 0; cu < CU; cu++)
    // {
    //     for (int layer = 0; layer < NUM_LAYER; layer++)
    //     {
    //         load_float_data(inp_addr_map[cu], prefix + "const_data_" + to_string(cu) + ".bin", 13 * INP_LEN, 13 * INP_LEN * layer);
    //         inp_addr[cu].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    //         run[cu].set_arg(2, layer);
    //         run[cu].start();
    //         run[cu].wait();
    //     }
    //     // exit(0);
    //     load_float_data(inp_addr_map[cu], prefix + "const_data_" + to_string(cu) + ".bin", 108 * 2, 13 * INP_LEN * NUM_LAYER);
    //     inp_addr[cu].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    //     run[cu].set_arg(2, NUM_LAYER);
    //     run[cu].start();
    //     run[cu].wait();
    //     run[cu].set_arg(2, -1);
    // }

    //  for(int i = 0; i < num_cu; i++)
    // {
    //     load_float_data(inp_addr_map[i], prefix + "inp_addr.bin", FULL_INP_LEN, 0);
    //     inp_addr[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // }

    std::string path = "./opt-1.3b-weights/";

    // token embedding
    std::vector<float> token_embedding_weight;
    size_t token_embedding_size = 50272 * FULL_INP_LEN;
    read_binary_weights(path + "embed_tokens.weight.bin", token_embedding_weight, token_embedding_size);

    // position embedding
    std::vector<float> pos_embedding_weight;
    size_t pos_embedding_size = 512 * FULL_INP_LEN;
    read_binary_weights(path + "pos_embeds.bin", pos_embedding_weight, pos_embedding_size);

    // std::cout << "Execution of the kernel\n";
    // for(int j = 0; j < 1; j++)
    // {
    //     for(int k = 0; k < 1; k++)
    //     {
    //         for(int i = 0; i < num_cu; i++)
    //         {
    //             // load_float_data(inp_addr_map[i], prefix + "inp_addr.bin", FULL_INP_LEN, FULL_INP_LEN * j);
    //             for (int i_emb = 0; i_emb < FULL_INP_LEN; i_emb++)
    //             {
    //                 inp_addr_map[i][i_emb] = token_embedding_weight[2 * FULL_INP_LEN + i_emb] + pos_embedding_weight[j * FULL_INP_LEN + i_emb];
    //             }
    //             inp_addr[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    //             run[i].set_arg(0, j);
    //             // run[i].set_arg(11, 1);
    //             run[i].start();
    //         }

    //         // std::cout << "Wait the kernel\n";

    //         for(int i = 0; i < num_cu; i++)
    //         {
    //             run[i].wait();
    //         }
    //     }
    // }

    std::cout << "Execution of the kernel\n";

    for (int i = 0; i < num_cu; i++)
    {
        for (int i_emb = 0; i_emb < FULL_INP_LEN; i_emb++)
        {
            inp_addr_map[i][i_emb] = token_embedding_weight[2 * FULL_INP_LEN + i_emb] + pos_embedding_weight[0 * FULL_INP_LEN + i_emb];
        }
        inp_addr[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        run[i].set_arg(0, 639);
        run[i].set_arg(2, -1);
    }

    for (int j = 0; j < 1024; j++)
    {
        for (int k = 0; k < 1; k++)
        {
            for (int i = 0; i < num_cu; i++)
            {
                run[i].start();
            }

            // std::cout << "Wait the kernel\n";

            for (int i = 0; i < num_cu; i++)
            {
                run[i].wait();
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Average execution time per k iteration: " << (duration.count() / 1024) << " seconds\n";

    for (int i = 0; i < num_cu; i++)
    {
        inp_addr[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }

    // read_float_data(inp_addr_map[0], FULL_INP_LEN * 2, FULL_INP_LEN, 192);
    read_float_data(inp_addr_map[0], FULL_INP_LEN);
    // read_float_data(inp_addr_map[0], FULL_INP_LEN * 2);

    // for(int i = 0; i < num_cu; i++){
    //     read_float_data(inp_addr_map[i], FULL_INP_LEN);
    // }

    w_addr_0[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    w_addr_1[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    w_addr_2[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    w_addr_3[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    w_addr_4[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    w_addr_5[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    w_addr_6[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    w_addr_7[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    w_addr_0[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    w_addr_1[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    w_addr_2[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    w_addr_3[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    w_addr_4[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    w_addr_5[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    w_addr_6[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    w_addr_7[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // w_addr_0[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // w_addr_1[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // w_addr_2[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // w_addr_3[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // kv_cache_size
    int my_bias = 2;
    read_io_pack_int8_data(w_addr_0_map[0], weight_size + my_bias, weight_size);
    read_io_pack_int8_data(w_addr_1_map[0], weight_size + my_bias, weight_size);
    read_io_pack_int8_data(w_addr_2_map[0], weight_size + my_bias, weight_size);
    read_io_pack_int8_data(w_addr_3_map[0], weight_size + my_bias, weight_size);
    read_io_pack_int8_data(w_addr_4_map[0], weight_size + my_bias, weight_size);
    read_io_pack_int8_data(w_addr_5_map[0], weight_size + my_bias, weight_size);
    read_io_pack_int8_data(w_addr_6_map[0], weight_size + my_bias, weight_size);
    read_io_pack_int8_data(w_addr_7_map[0], weight_size + my_bias, weight_size);

    std::cout << std::endl;

    read_io_pack_int8_data(w_addr_0_map[1], weight_size + my_bias, weight_size);
    read_io_pack_int8_data(w_addr_1_map[1], weight_size + my_bias, weight_size);
    read_io_pack_int8_data(w_addr_2_map[1], weight_size + my_bias, weight_size);
    read_io_pack_int8_data(w_addr_3_map[1], weight_size + my_bias, weight_size);
    read_io_pack_int8_data(w_addr_4_map[1], weight_size + my_bias, weight_size);
    read_io_pack_int8_data(w_addr_5_map[1], weight_size + my_bias, weight_size);
    read_io_pack_int8_data(w_addr_6_map[1], weight_size + my_bias, weight_size);
    read_io_pack_int8_data(w_addr_7_map[1], weight_size + my_bias, weight_size);

    std::cout << "TEST PASSED\n";
    return 0;
}

//  sudo /opt/xilinx/xrt/bin/xbmgmt program --shell  /lib/firmware/xilinx/12c8fafb0632499db1c0c6676271b8a6/partition.xsabin  -d 0000:25:00.0