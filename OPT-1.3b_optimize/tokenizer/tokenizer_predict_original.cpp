#include <iostream>
#include <sstream>
#include <string>
#include "json.hpp"
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"
#include "utils.h"
#include "loopLynx.h"
using namespace std;

void read_binary_weights(const std::string &filename, std::vector<float> &weights, size_t size)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    weights.resize(size);
    file.read(reinterpret_cast<char *>(weights.data()), size * sizeof(float));
    if (!file)
    {
        std::cerr << "Error reading from file: " << filename << std::endl;
    }
}

void read_input_ids(const std::string &filename, std::vector<int64_t> &weights, size_t size)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    weights.resize(size);
    file.read(reinterpret_cast<char *>(weights.data()), size * sizeof(int64_t));
    if (!file)
    {
        std::cerr << "Error reading from file: " << filename << std::endl;
    }
}

void layerNorm(
    float *inp,
    float *outp,
    float *ln_weight,
    float *ln_bias,
    int embedding_size)
{
    float mean = 0.0, var = 0.0;
    for (int i_emb = 0; i_emb < embedding_size; i_emb++)
    {
        float data = inp[i_emb];
        mean += data;
        float square = data * data;
        var += square;
    }

    const float scale = 1 / (float)embedding_size;
    mean *= scale;
    float mean_square = (float)mean * (float)mean;
    var *= scale;
    var -= mean_square;
    const float eps = 0.000010;
    float sqrt_denomenator_rev = 1 / sqrt(var + eps);

    for (int i_emb = 0; i_emb < embedding_size; i_emb++)
    {
        outp[i_emb] = (inp[i_emb] - mean) * sqrt_denomenator_rev * ln_weight[i_emb] + ln_bias[i_emb];
    }
}

void matrixMultiply(const float *weights,
                    const float *input,
                    float *output,
                    size_t numRows,
                    size_t numCols)
{
    for (size_t i = 0; i < numRows; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < numCols; ++j)
        {
            sum += weights[i * numCols + j] * input[j];
        }
        output[i] = sum;
    }
}

int argmax(const float *array, size_t size)
{
    int maxIndex = 0;
    float maxValue = std::numeric_limits<float>::lowest(); // 初始化为最小值

    for (size_t i = 0; i < size; ++i)
    {
        if (array[i] > maxValue)
        {
            maxValue = array[i];
            maxIndex = static_cast<int>(i);
        }
    }
    return maxIndex;
}


int main(int argc, char **argv)
{

    std::string binaryFile = argv[1];
    int device_index = 0;

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile.c_str());

    const int num_cu = CU;

    xrt::bo inp_addr[num_cu];
    xrt::bo outp_addr[num_cu];
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


    int weight_size_ = NUM_LAYER * FULL_INP_LEN * FULL_INP_LEN / CU / PROCESSOR * 12 / INP_NUM;
    int kv_cache_size = (NUM_LAYER * FULL_SEQ_NUM * HEAD_NUM / CU * HEAD_LEN / INP_NUM / 2);

    for(int i = 0; i < num_cu; i++){
        inp_addr[i] = xrt::bo(device, sizeof(float) * INP_LEN * 13, kernel[i].group_id(1));
        w_addr_0[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size_ + kv_cache_size), kernel[i].group_id(3));
        w_addr_1[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size_ + kv_cache_size), kernel[i].group_id(4));
        w_addr_2[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size_ + kv_cache_size), kernel[i].group_id(5));
        w_addr_3[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size_ + kv_cache_size), kernel[i].group_id(6));
        w_addr_4[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size_ + kv_cache_size), kernel[i].group_id(7));
        w_addr_5[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size_ + kv_cache_size), kernel[i].group_id(8));
        w_addr_6[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size_ + kv_cache_size), kernel[i].group_id(9));
        w_addr_7[i] = xrt::bo(device, sizeof(io_pack_int8) * (weight_size_ + kv_cache_size), kernel[i].group_id(10));
    }   

    std::cout<<"Allocate Finish"<<std::endl;
    float * inp_addr_map[num_cu] = {nullptr}; 
    io_pack_int8 * w_addr_0_map[num_cu] = {nullptr}; 
    io_pack_int8 * w_addr_1_map[num_cu] = {nullptr}; 
    io_pack_int8 * w_addr_2_map[num_cu] = {nullptr}; 
    io_pack_int8 * w_addr_3_map[num_cu] = {nullptr}; 
    io_pack_int8 * w_addr_4_map[num_cu] = {nullptr}; 
    io_pack_int8 * w_addr_5_map[num_cu] = {nullptr}; 
    io_pack_int8 * w_addr_6_map[num_cu] = {nullptr}; 
    io_pack_int8 * w_addr_7_map[num_cu] = {nullptr}; 
 
    for(int i = 0; i < num_cu; i++){
        inp_addr_map[i] = inp_addr[i].map<float * >();
        w_addr_0_map[i] = w_addr_0[i].map<io_pack_int8 * >();
        w_addr_1_map[i] = w_addr_1[i].map<io_pack_int8 * >();
        w_addr_2_map[i] = w_addr_2[i].map<io_pack_int8 * >();
        w_addr_3_map[i] = w_addr_3[i].map<io_pack_int8 * >();
        w_addr_4_map[i] = w_addr_4[i].map<io_pack_int8 * >();
        w_addr_5_map[i] = w_addr_5[i].map<io_pack_int8 * >();
        w_addr_6_map[i] = w_addr_6[i].map<io_pack_int8 * >();
        w_addr_7_map[i] = w_addr_7[i].map<io_pack_int8 * >();
    }   
    
    string prefix = "../const_data/";
    int load_layer = 24;

    for (int cu = 0; cu < num_cu; cu++)
    {   
        load_io_pack_int8_data(w_addr_0_map[cu], prefix + "w" + to_string(cu) + "_addr_0.bin", weight_size_);
        load_io_pack_int8_data(w_addr_1_map[cu], prefix + "w" + to_string(cu) + "_addr_1.bin", weight_size_);
        load_io_pack_int8_data(w_addr_2_map[cu], prefix + "w" + to_string(cu) + "_addr_2.bin", weight_size_);
        load_io_pack_int8_data(w_addr_3_map[cu], prefix + "w" + to_string(cu) + "_addr_3.bin", weight_size_);
        load_io_pack_int8_data(w_addr_4_map[cu], prefix + "w" + to_string(cu) + "_addr_4.bin", weight_size_);
        load_io_pack_int8_data(w_addr_5_map[cu], prefix + "w" + to_string(cu) + "_addr_5.bin", weight_size_);
        load_io_pack_int8_data(w_addr_6_map[cu], prefix + "w" + to_string(cu) + "_addr_6.bin", weight_size_);
        load_io_pack_int8_data(w_addr_7_map[cu], prefix + "w" + to_string(cu) + "_addr_7.bin", weight_size_);
    }

    std::cout << "synchronize input buffer data to device global memory\n";

    for(int i = 0; i < num_cu; i++){
        w_addr_0[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_addr_1[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_addr_2[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_addr_3[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_addr_4[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_addr_5[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_addr_6[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        w_addr_7[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    std::cout << "Execution of the kernel\n";
    xrt::run run[num_cu];
    for(int i = 0; i < num_cu; i++){
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

    std::cout<<"sync scaling factor and bias onboard"<<std::endl;
    for (int cu = 0; cu < CU; cu++)
    {   
        for(int layer = 0; layer < NUM_LAYER; layer++)
        {
            load_float_data(inp_addr_map[cu], prefix + "const_data_" + to_string(cu) + ".bin", 13 * INP_LEN, 13 * INP_LEN * layer);
            inp_addr[cu].sync(XCL_BO_SYNC_BO_TO_DEVICE);
            run[cu].set_arg(2, layer);
            run[cu].start();
            run[cu].wait();
        }
        // exit(0);
        load_float_data(inp_addr_map[cu], prefix + "const_data_" + to_string(cu) + ".bin", 4 * INP_LEN, 13 * INP_LEN * NUM_LAYER);
        inp_addr[cu].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        run[cu].set_arg(2, NUM_LAYER);
        run[cu].start();
        run[cu].wait();
        run[cu].set_arg(2, -1);
    }

    std::string path = "./model_weights/";
    // std::vector<int64_t> input_ids;
    // size_t input_size = 1 * 512;
    // read_input_ids(path + "input_ids.bin", input_ids, input_size);

    // token embedding
    std::vector<float> token_embedding_weight;
    size_t token_embedding_size = 50272 * 2048;
    read_binary_weights(path + "embed_tokens.weight.bin", token_embedding_weight, token_embedding_size);

    // position embedding
    std::vector<float> pos_embedding_weight;
    size_t pos_embedding_size = 512 * 2048;
    read_binary_weights(path + "pos_embeds.bin", pos_embedding_weight, pos_embedding_size);

    // final_layer_norm
    std::vector<float> layer_norm_weight, layer_norm_bias;
    read_binary_weights(path + "final_layer_norm.weight.bin", layer_norm_weight, 2048);
    read_binary_weights(path + "final_layer_norm.bias.bin", layer_norm_bias, 2048);

    // linear layer
    int64_t in_features = 2048;
    int64_t out_features = 50272;
    size_t weight_size = in_features * out_features;
    std::vector<float> lm_weight;
    read_binary_weights(path + "lm_head.weight.bin", lm_weight, weight_size);

    // hidden layer ground truth
    // size_t hidden_layer_size = 512 * 2048;
    // std::vector<float> hidden_layer_weight;
    // read_binary_weights(path + "hidden_states.bin", hidden_layer_weight, hidden_layer_size);

    // load vocab
    std::ifstream vocab_file(path + "vocab_rev.json");
    nlohmann::json vocab;
    vocab_file >> vocab;
    vocab_file.close();


    std::vector<int64_t> input_sizes;
    read_input_ids("inputs_hardware_full/sizes.bin", input_sizes, 4869);
  
    // for(int i = 0; i < 1000; i++)
    // {
    //     std::ostringstream oss;
    //     oss << "inputs_hardware/input_" << std::setw(4) << std::setfill('0') << i << ".bin";
    //     std::string path = oss.str();
    //     std::vector<int64_t> input_ids;
    //     read_input_ids(path, input_ids, input_sizes[i]);
    //     for(int j = 0; j < input_sizes[i]; j++)
    //     {
    //         std::cout<<input_ids[j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

    int hit = 0;
    for (int batch = 0; batch < 4869; batch++){
        
        // batch = 2;
        std::cout<<"input sequence "<<batch<<std::endl;
        std::ostringstream oss;
        oss << "inputs_hardware_full/input_" << std::setw(4) << std::setfill('0') << batch << ".bin";
        std::string path = oss.str();
        std::vector<int64_t> input_ids;
        read_input_ids(path, input_ids, input_sizes[batch]);

        int last_id = 0;
        for (int token = 0; token < input_sizes[batch] - 1; token++)
        {
            int input_id = input_ids[token];
            float hidden_state[FULL_INP_LEN];
            float out_hidden_state[FULL_INP_LEN];
            float out_layer_norm[FULL_INP_LEN];
            float out_logits[50272];
            int token_idx = token;
            if(token_idx==107) token_idx = 0;
            
            // std::cout<<token<<" input token "<< vocab[to_string(input_id)];
            // embedding
            for (int i_emb = 0; i_emb < FULL_INP_LEN; i_emb++)
            {
                inp_addr_map[0][i_emb] = token_embedding_weight[input_id * FULL_INP_LEN + i_emb] + pos_embedding_weight[token_idx * FULL_INP_LEN + i_emb];
                inp_addr_map[1][i_emb] = token_embedding_weight[input_id * FULL_INP_LEN + i_emb] + pos_embedding_weight[token_idx * FULL_INP_LEN + i_emb];
                // inp_addr_map[2][i_emb] = token_embedding_weight[input_id * FULL_INP_LEN + i_emb] + pos_embedding_weight[token_idx * FULL_INP_LEN + i_emb];
                // inp_addr_map[3][i_emb] = token_embedding_weight[input_id * FULL_INP_LEN + i_emb] + pos_embedding_weight[token_idx * FULL_INP_LEN + i_emb];
            }

                for (int i = 0; i < num_cu; i++)
                {
                    // load_float_data(inp_addr_map[i], prefix + "inp_addr.bin", FULL_INP_LEN, FULL_INP_LEN * j);
                    inp_addr[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    run[i].set_arg(0, token_idx);
                    run[i].start();
                }

                // std::cout << "Wait the kernel\n";

                for (int i = 0; i < num_cu; i++)
                {
                    run[i].wait();
                }

            for(int i = 0; i < num_cu; i++)
            {
                inp_addr[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            }


            // for(int i = 0; i < 1; i++){
            //     read_float_data(inp_addr_map[i], FULL_INP_LEN);
            // }
            // if(token == 1 ) exit(0);

            // w_addr_0[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            // w_addr_1[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            // w_addr_2[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            // w_addr_3[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            // w_addr_4[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            // w_addr_5[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            // w_addr_6[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            // w_addr_7[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);

            // int my_bias = 2;
            // read_io_pack_int8_data(w_addr_0_map[0], weight_size_ + my_bias, weight_size_);
            // read_io_pack_int8_data(w_addr_1_map[0], weight_size_ + my_bias, weight_size_);
            // read_io_pack_int8_data(w_addr_2_map[0], weight_size_ + my_bias, weight_size_);
            // read_io_pack_int8_data(w_addr_3_map[0], weight_size_ + my_bias, weight_size_);
            // read_io_pack_int8_data(w_addr_4_map[0], weight_size_ + my_bias, weight_size_);
            // read_io_pack_int8_data(w_addr_5_map[0], weight_size_ + my_bias, weight_size_);
            // read_io_pack_int8_data(w_addr_6_map[0], weight_size_ + my_bias, weight_size_);
            // read_io_pack_int8_data(w_addr_7_map[0], weight_size_ + my_bias, weight_size_);
            
            for (int i_emb = 0; i_emb < FULL_INP_LEN; i_emb++)
            {
                // out_hidden_state[i_emb] = hidden_layer_weight[token * 768 + i_emb];
                
                out_hidden_state[i_emb] = inp_addr_map[0][i_emb];
                // std::cout<<out_hidden_state[i_emb]<<", ";
            }
            // std::cout<<std::endl;
            layerNorm(out_hidden_state, out_layer_norm, layer_norm_weight.data(), layer_norm_bias.data(), FULL_INP_LEN);
            matrixMultiply(lm_weight.data(), out_layer_norm, out_logits, 50272, FULL_INP_LEN);
            int maxIndex = argmax(out_logits, 50272);
            last_id = maxIndex;
            auto json_idx = to_string(maxIndex);
            // if (vocab.contains(json_idx))
            // {
            //     // std::cout << vocab[json_idx] << ", ";
            //     std::cout << "output "<< vocab[json_idx] << std::endl;
            // }
            // else
            // {
            //     std::cout << "\nnot exist\n"
            //               << std::endl;
            // }
        }
        // std::cout<<last_id<<std::endl;
        // std::cout<<input_ids[input_sizes[batch] - 1]<<std::endl;
        if(last_id == input_ids[input_sizes[batch] - 1])
        {
            std::cout<<"match"<<std::endl;
            hit++;
            std::cout<<"rate "<< (float) hit / (float) batch<<std::endl;
        }
    }
    return 0;
}