void value_generator(
    io_pack_int8 q_value_head,
    hls::stream<io_pack_int8> &head_loader,
    int seq_id)
{
    for (int round = 0; round <= seq_id; round++)
    {
        head_loader.write(q_value_head);
    }
}

void Attention_head_wise(
    io_pack_int8 q_value_head,
    hls::stream<io_pack_int8> &weight_loader,
    hls::stream<int> &sub_acc_result,
    int seq_id)
{
#pragma HLS DATAFLOW
    hls::stream<io_pack_int8> head_loader;
#pragma HLS STREAM variable = head_loader depth = 2
#pragma HLS BIND_STORAGE variable = head_loader type = fifo impl = srl
    value_generator(q_value_head, head_loader, seq_id);
    {Mul_Adder_Tree}(head_loader, weight_loader, sub_acc_result, seq_id + 1);
}

void attention_layer_new(
    hls::stream<io_pack_int8> &q_loader,
    hls::stream<io_pack_int8> &weight_loader,
    hls::stream<int> &sub_acc_result,
    int seq_id)
{
#pragma HLS DATAFLOW
    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
        io_pack_int8 q_value_head = q_loader.read();
        Attention_head_wise(q_value_head, weight_loader, sub_acc_result, seq_id);
    }
}

void merge_sub_result(
    hls::stream<int> sub_acc_result[ATTENTION_CHANNELS],
    hls::stream<int> result[HEAD_PARALLEL],
    int seq_id)
{
    int sub_result[ATTENTION_CHANNELS];
    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
        for (int round = 0; round <= seq_id; round++)
        {
            for (int channel = 0; channel < ATTENTION_CHANNELS; channel++)
            {
#pragma HLS UNROLL
                sub_result[channel] = sub_acc_result[channel].read();
            }

            for (int parallel = 0; parallel < HEAD_PARALLEL; parallel++)
            {
#pragma HLS UNROLL
                // int acc_result = sub_result[parallel * 4 + 1] + sub_result[parallel * 4 + 2] + sub_result[parallel * 4 + 3] + sub_result[parallel * 4 + 3];
                // int acc_result = sub_result[parallel * 2] + sub_result[parallel * 2 + 1];
                // int acc_result = sub_result[parallel];
                {ACC_RESULT_SELECT}
                result[parallel].write(acc_result);
            }
        }
    }
}

void requant_attn(
    hls::stream<int> &acc_result,
    hls::stream<float> &requant_result,
    int seq_id, int layer_id, const float attn_alpha[NUM_LAYER])
{

    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
        for (int round = 0; round <= seq_id; round++)
        {
            int acc = acc_result.read();
            float outp = acc * attn_alpha[layer_id];
            requant_result.write(outp);
        }
    }
}

void softmax_new(
    hls::stream<float> &inp,
    hls::stream<int8_t> outp[HEAD_LEN/INP_NUM],
    int layer_id, int seq_id)
{
#pragma HLS DATAFLOW
    typedef ap_fixed<32 + {LOG_FULL_SEQ_LEN}, {LOG_FULL_SEQ_LEN}> sfm_fixed;
    hls::stream<sfm_fixed> inp_sumRow_fixed_pipe;
#pragma HLS STREAM variable = inp_sumRow_fixed_pipe depth = 64
#pragma HLS BIND_STORAGE variable = inp_sumRow_fixed_pipe type = fifo impl = srl
    hls::stream<float> inverse_inp_sumRow_pipe;
#pragma HLS STREAM variable = inverse_inp_sumRow_pipe depth = 64
#pragma HLS BIND_STORAGE variable = inverse_inp_sumRow_pipe type = fifo impl = srl
    hls::stream<float> max_float_pipe;
#pragma HLS STREAM variable = max_float_pipe depth = 64
#pragma HLS BIND_STORAGE variable = max_float_pipe type = fifo impl = srl
    hls::stream<float> inp_value_bypass;
#pragma HLS STREAM variable = inp_value_bypass depth = 4096
#pragma HLS BIND_STORAGE variable = inp_value_bypass type = fifo impl = bram
    hls::stream<float> inp_value_normalized_bypass;
#pragma HLS STREAM variable = inp_value_normalized_bypass depth = 4096
#pragma HLS BIND_STORAGE variable = inp_value_normalized_bypass type = fifo impl = bram

    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
        float max_val = -1e9;
        for (int i_seq = 0; i_seq <= seq_id; i_seq++)
        {
#pragma HLS pipeline II = 1
            float input_data = inp.read();
            max_val = fmax(max_val, input_data);
            inp_value_bypass.write(input_data);
        }
        max_float_pipe.write(max_val);
    }

    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
        float max_val = max_float_pipe.read();
        sfm_fixed inp_sumRow_fixed = 0.0;
        for (int i_seq = 0; i_seq <= seq_id; i_seq++)
        {
#pragma HLS pipeline II = 1
            const float small_value = 1e-6;
            float original_data = inp_value_bypass.read();
            float input_data = original_data - max_val;
            float exp_val = exp(input_data) + small_value;
            inp_sumRow_fixed += sfm_fixed(exp_val);
            inp_value_normalized_bypass.write(exp_val);
        }
        inp_sumRow_fixed_pipe.write(inp_sumRow_fixed);
    }

    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
        auto inp_sumRow_fixed = inp_sumRow_fixed_pipe.read();
        float inverse_inp_sumRow = 1 / (float)inp_sumRow_fixed;
        inverse_inp_sumRow_pipe.write(inverse_inp_sumRow);
    }

    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
        float inverse_inp_sumRow = inverse_inp_sumRow_pipe.read();
        for (int i_seq = 0; i_seq <= seq_id; i_seq++)
        {
#pragma HLS pipeline II = 1
            float input_data = inp_value_normalized_bypass.read();
            float normalize_1 = input_data * inverse_inp_sumRow;
            float normalize_2 = normalize_1 * 127.0;
            float result;
            if (normalize_2 <= -128)
            {
                result = 0;
            }
            else if (normalize_2 >= 127)
            {
                result = 127;
            }
            else
            {
                result = normalize_2;
            }
            int8_t quantize = round(result);
            for(int parallel = 0; parallel < HEAD_LEN/INP_NUM; parallel++)
            {
        #pragma HLS UNROLL
                outp[parallel].write(quantize);
            }
        }
    }
}

void loader_new(
    io_pack_int8 *w_addr,
    hls::stream<io_pack_int8> &kv_stream,
    int layer_id, int seq_id)
{
    int memory_bias = (layer_id * KV_CACHE_SIZE) + MEMORY_SIZE;
    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
        for (int row = 0; row <= seq_id; row++)
        {
#pragma HLS PIPELINE II = 1
            int idx = memory_bias + head * FULL_SEQ_NUM + row;
            io_pack_int8 data_pack = w_addr[idx];
            kv_stream.write(data_pack);
        }
    }
}

void Mac_32(
    hls::stream<io_pack_int8> &loader_0,
    hls::stream<io_pack_int8> &loader_1,
    hls::stream<router_pack_int> &block_C_drainer,
    int ROUND)
{
    int C[INP_NUM] = {0};
    for (int k = 0; k < ROUND; k++)
    {
#pragma HLS PIPELINE II = 1
        io_pack_int8 data_pack_0 = loader_0.read();
        io_pack_int8 data_pack_1 = loader_1.read();
        for (int i = 0; i < INP_NUM; i++)
        {
#pragma HLS UNROLL
            ap_int<8> a = data_pack_0.range(8 * i + 7, 8 * i);
            ap_int<8> b = data_pack_1.range(8 * i + 7, 8 * i);
            ap_int<16> result = a * b;
            C[i] += result;
        }
    }

    router_pack_int result;
    for (int serial = 0; serial < INP_NUM / ROUTE_NUM; serial++)
    {
        for (int parallel = 0; parallel < ROUTE_NUM; parallel++)
        {
#pragma HLS UNROLL
            result.range(32 * parallel + 31, 32 * parallel) = C[serial * ROUTE_NUM + parallel];
        }
        block_C_drainer.write(result);
    }
}

void ctx_token_generator(
    hls::stream<int8_t> &token_loader,
    hls::stream<io_pack_int8> &head_loader,
    int seq_id)
{
    for (int round = 0; round <= seq_id; round++)
    {
        int8_t data = token_loader.read();
        io_pack_int8 data_pack;
        for (int parallel = 0; parallel < INP_NUM; parallel++)
        {
#pragma HLS UNROLL
            data_pack.range(parallel * 8 + 7, parallel * 8) = data;
        }
        head_loader.write(data_pack);
    }
}

void context_head_wise(
    hls::stream<int8_t> &token_loader,
    hls::stream<io_pack_int8> &weight_loader,
    hls::stream<router_pack_int> &block_drainer,
    int seq_id)
{
#pragma HLS DATAFLOW

    hls::stream<io_pack_int8> head_loader;
#pragma HLS STREAM variable = head_loader depth = 2
#pragma HLS BIND_STORAGE variable = head_loader type = fifo impl = srl

    ctx_token_generator(token_loader, head_loader, seq_id);
    Mac_32(head_loader, weight_loader, block_drainer, seq_id + 1);
}

void requant_ctx(
    hls::stream<router_pack_int> &block_drainer,
    hls::stream<router_pack_float> &requant_result,
    int seq_id, int layer_id, const float ctx_alpha[NUM_LAYER])
{

    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
        for (int serial = 0; serial < INP_NUM / ROUTE_NUM; serial++)
        {
            router_pack_int acc_data_pack = block_drainer.read();
            router_pack_float data_pack;
            for (int parallel = 0; parallel < ROUTE_NUM; parallel++)
            {
#pragma HLS PIPELINE II = 1
                int data = acc_data_pack.range(32 * parallel + 31, 32 * parallel);
                float data_fp = data * ctx_alpha[layer_id];
                float result;
                if (data_fp <= -128)
                {
                    result = -128;
                }
                else if (data_fp >= 127)
                {
                    result = 127;
                }
                else
                {
                    result = data_fp;
                }
                converter_t converter;
                converter.f = result;
                data_pack.range(32 * parallel + 31, 32 * parallel) = converter.i;
            }
            requant_result.write(data_pack);
        }
    }
}

void packet_merger(
    hls::stream<router_pack_float> requant_result[ATTENTION_CHANNELS],
    hls::stream<router_pack_float> &for_router)
{
    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
        for (int parallel = 0; parallel < ATTENTION_CHANNELS; parallel++)
        {

            for (int serial = 0; serial < INP_NUM / ROUTE_NUM; serial++)
            {
#pragma HLS PIPELINE II = 1
                router_pack_float data_pack = requant_result[parallel].read();
                for_router.write(data_pack);
            }
        }
    }
}

void context_layer_new(
    hls::stream<int8_t> &token_loader,
    hls::stream<io_pack_int8> &weight_loader,
    hls::stream<router_pack_int> &block_drainer,
    int layer_id, int seq_id)
{
#pragma HLS DATAFLOW
    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
        context_head_wise(token_loader, weight_loader, block_drainer, seq_id);
    }
}

void q_loader_new(
    int8_t *q_buffer,
    hls::stream<io_pack_int8> q_stream[ATTENTION_CHANNELS])
{
    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
        int HEAD_BIAS = head * HEAD_PARALLEL * HEAD_LEN;
        for (int parallel = 0; parallel < HEAD_PARALLEL; parallel++)
        {
            int PARALLEL_BIAS = parallel * HEAD_LEN + HEAD_BIAS;
            for (int pe = 0; pe < (HEAD_LEN / INP_NUM); pe++)
            {
#pragma HLS PIPELINE II = 1
                int PE_BIAS = pe * INP_NUM + PARALLEL_BIAS;
                io_pack_int8 q_value_head;
                for (int parallel = 0; parallel < INP_NUM; parallel++)
                {
#pragma HLS UNROLL
                    q_value_head.range(8 * parallel + 7, 8 * parallel) = q_buffer[PE_BIAS + parallel];
                }
                q_stream[parallel * (HEAD_LEN / INP_NUM) + pe].write(q_value_head);
            }
        }
    }
}

void attention_wrapper_new(
    int8_t *q_buffer,
{MEM_PORT_REGION}
    hls::stream<router_pack_float> &stream_previous,
    hls::stream<router_pack_float> &stream_next,
    int8_t output_buffer[FULL_INP_LEN],
    int layer_id, int seq_id, const int device_id)
{
    hls::stream<io_pack_int8> q_stream[ATTENTION_CHANNELS];
#pragma HLS STREAM variable = q_stream depth = 2
#pragma HLS BIND_STORAGE variable = q_stream type = fifo impl = srl
    hls::stream<io_pack_int8> k_stream[ATTENTION_CHANNELS];
#pragma HLS STREAM variable = k_stream depth = 2
#pragma HLS BIND_STORAGE variable = k_stream type = fifo impl = srl
    hls::stream<io_pack_int8> v_stream[ATTENTION_CHANNELS];
#pragma HLS STREAM variable = v_stream depth = 2
#pragma HLS BIND_STORAGE variable = v_stream type = fifo impl = srl
    hls::stream<int> sub_acc_result[ATTENTION_CHANNELS];
#pragma HLS STREAM variable = sub_acc_result depth = 2
#pragma HLS BIND_STORAGE variable = sub_acc_result type = fifo impl = srl
    hls::stream<int> merged_result[HEAD_PARALLEL];
#pragma HLS STREAM variable = merged_result depth = 2
#pragma HLS BIND_STORAGE variable = merged_result type = fifo impl = srl
    hls::stream<float> attn_requant_result[HEAD_PARALLEL];
#pragma HLS STREAM variable = attn_requant_result depth = 2
#pragma HLS BIND_STORAGE variable = attn_requant_result type = fifo impl = srl
    hls::stream<int8_t> sfm_out[HEAD_PARALLEL][HEAD_LEN / INP_NUM];
#pragma HLS STREAM variable = sfm_out depth = 1024
#pragma HLS BIND_STORAGE variable = sfm_out type = fifo // impl = uram
    hls::stream<router_pack_int> ctx_acc_drainer[ATTENTION_CHANNELS];
#pragma HLS STREAM variable = ctx_acc_drainer depth = 32
#pragma HLS BIND_STORAGE variable = ctx_acc_drainer type = fifo impl = srl
    hls::stream<router_pack_float> ctx_requant_result[ATTENTION_CHANNELS];
#pragma HLS STREAM variable = ctx_requant_result depth = 32
#pragma HLS BIND_STORAGE variable = ctx_requant_result type = fifo impl = srl
    hls::stream<router_pack_float> ctx_requant_merged;
#pragma HLS STREAM variable = ctx_requant_merged depth = 32
#pragma HLS BIND_STORAGE variable = ctx_requant_merged type = fifo impl = srl
    hls::stream<router_pack_float> router_out;
#pragma HLS STREAM variable = router_out depth = 32
#pragma HLS BIND_STORAGE variable = router_out type = fifo impl = srl

#pragma HLS DATAFLOW

#include "{CONST_DATA_DIR}/attn_alpha.txt"
#include "{CONST_DATA_DIR}/ctx_alpha.txt"

{KV_LOADER_REAGION}
    // loader_new(w_addr_0, k_stream[0], layer_id, seq_id);
    q_loader_new(q_buffer, q_stream);

    for(int PE = 0; PE < ATTENTION_CHANNELS; PE++)
    {
#pragma HLS UNROLL
        attention_layer_new(q_stream[PE], k_stream[PE], sub_acc_result[PE], seq_id);
    }

    merge_sub_result(sub_acc_result, merged_result, seq_id);

    for(int PE = 0; PE < HEAD_PARALLEL; PE++)
    {
#pragma HLS UNROLL
        requant_attn(merged_result[PE], attn_requant_result[PE], seq_id, layer_id, attn_alpha);
        softmax_new(attn_requant_result[PE], sfm_out[PE], layer_id, seq_id);
    }
   
    for(int outer = 0; outer < HEAD_PARALLEL; outer++)
    {
#pragma HLS UNROLL
        for(int inner = 0; inner < (HEAD_LEN / INP_NUM); inner++)
        {
#pragma HLS UNROLL
            context_layer_new(sfm_out[outer][inner], v_stream[outer * (HEAD_LEN / INP_NUM) + inner], ctx_acc_drainer[outer * (HEAD_LEN / INP_NUM) + inner], layer_id, seq_id);
        }
    }

    for(int PE = 0; PE < ATTENTION_CHANNELS; PE++)
    {
#pragma HLS UNROLL
        requant_ctx(ctx_acc_drainer[PE], ctx_requant_result[PE], seq_id, layer_id, ctx_alpha);
    }

    packet_merger(ctx_requant_result, ctx_requant_merged);
{WEATHER_USE_ROUTER_ATTENTION}
}