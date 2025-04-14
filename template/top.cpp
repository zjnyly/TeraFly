void loopLynx_{CU_ID}(
    int i_seq,
    float *host_addr,
    int load_weight_layer,
{MEM_PORT_REGION}
{RING_CONNECTION})
{
#pragma HLS interface m_axi port = host_addr offset = slave bundle = gmem8 
{MEM_PRAGMA_REGION}

    const int device_id = {CU_ID};

    float input_buffer[FULL_INP_LEN];
#pragma HLS array_partition variable = input_buffer type = cyclic factor = INP_PARALLEL
    int8_t int_buffer[FULL_INP_LEN * 4];
#pragma HLS array_partition variable = int_buffer type = cyclic factor = INP_NUM
    float float_buffer[FULL_INP_LEN * 4];
#pragma HLS array_partition variable = float_buffer type = cyclic factor = INP_NUM
    float res_buffer[FULL_INP_LEN];
#pragma HLS array_partition variable = res_buffer type = cyclic factor = INP_PARALLEL
    int8_t KV_buffer[2 * ATTENTION_CHANNELS][INP_LEN / ATTENTION_CHANNELS];
#pragma HLS array_partition variable = KV_buffer type = cyclic factor = INP_NUM dim = 0
    int8_t q_buffer[INP_LEN];
#pragma HLS ARRAY_PARTITION variable = q_buffer type = cyclic factor = HEAD_LEN

    static float bias_onboard[BIAS_SIZE_TOTAL];
#pragma HLS bind_storage variable=bias_onboard type=ram_1p impl=URAM 
    static float ln_weight_onboard[LN_SIZE_TOTAL + INP_LEN / INP_PARALLEL * 2][INP_PARALLEL];
#pragma HLS bind_storage variable=ln_weight_onboard type=ram_1p impl=URAM 
    static float ln_bias_onboard[LN_SIZE_TOTAL + INP_LEN / INP_PARALLEL * 2][INP_PARALLEL];
#pragma HLS bind_storage variable=ln_bias_onboard type=ram_1p impl=URAM 
    static float linear_alpha[9 * NUM_LAYER];
#pragma HLS bind_storage variable=linear_alpha type=ram_1p impl=BRAM

    if(load_weight_layer >= 0)
    {
        int BIAS_FOR_BIAS = load_weight_layer * BIAS_SIZE_LAYER;
        int BIAS_FOR_LN = load_weight_layer * LN_SIZE_LAYER;
        
        // first for bias
        if (load_weight_layer < NUM_LAYER){
            for(int idx = 0; idx < BIAS_SIZE_LAYER; idx++)
            {
                #pragma HLS PIPELINE II=1
                bias_onboard[BIAS_FOR_BIAS + idx] = host_addr[idx + 4 * INP_LEN];
            }
            for(int idx = 0; idx < LN_SIZE_LAYER; idx++)
            {
                for(int serial = 0; serial < INP_PARALLEL; serial++)
                {
                    #pragma HLS PIPELINE II=1
                    int inner_idx = idx * INP_PARALLEL + serial;
                    ln_weight_onboard[BIAS_FOR_LN + idx][serial] = host_addr[inner_idx];
                    ln_bias_onboard[BIAS_FOR_LN + idx][serial] = host_addr[inner_idx +  2 * INP_LEN];
                }
            }
        }else{
            for(int idx = 0; idx < 9 * NUM_LAYER; idx++)
            {
                #pragma HLS PIPELINE II=1
                linear_alpha[idx] = host_addr[idx];
            }
        }
        return;
    }

    input_loader_memory(host_addr, input_buffer);    

    for (int i_layer = 0; i_layer < NUM_LAYER; i_layer++)
    {   
        int weight_layer_bias = WEIGHT_SIZE_LAYER * i_layer; 
        int scale_layer_bias = SCALING_FACTOR_SIZE_LAYER * i_layer;
        int bias_layer = BIAS_SIZE_LAYER * i_layer;
        int attn_ln_bias = LN_SIZE_LAYER * i_layer;
        int fn_ln_bias = attn_ln_bias + LN_BIAS_HALF;
        int memory_bias = MEMORY_SIZE + i_layer * KV_CACHE_SIZE;

        set_buffer_zero(float_buffer);

        Fused_Res_LN_copy(input_buffer, float_buffer, res_buffer, attn_ln_bias, stream_previous, stream_next, int_buffer, device_id, ln_weight_onboard, ln_bias_onboard);

        // compute QKV
        GEMM_QUANT({MEM_PORT_CALL}
                     stream_previous, stream_next, int_buffer, float_buffer, weight_layer_bias, scale_layer_bias, bias_layer, QKV_ROWS, QKV_COLS, device_id, false, bias_onboard, linear_alpha);
       
{WRITE_KV_ONCHIP}
{WRITE_KV_OFFCHIP}
        
        q_writer(float_buffer, q_buffer, device_id);
        
        // compute attention
        attention_wrapper_new(q_buffer, {MEM_PORT_CALL_ATTN} stream_previous, stream_next,
                              int_buffer, i_layer, i_seq, device_id);
        
        // compute O
        GEMM_QUANT({MEM_PORT_CALL}
                     stream_previous, stream_next, int_buffer, float_buffer, weight_layer_bias + WEIGHT_O_BIAS, scale_layer_bias + SCALE_O_BIAS, bias_layer + BIAS_O, O_ROWS, O_COLS, device_id, true, bias_onboard, linear_alpha);

        // Res
        Fused_Res_LN_copy(input_buffer, float_buffer, res_buffer, fn_ln_bias, stream_previous, stream_next, int_buffer, device_id, ln_weight_onboard, ln_bias_onboard);

        GEMM_QUANT({MEM_PORT_CALL}
                     stream_previous, stream_next, int_buffer, float_buffer, weight_layer_bias + WEIGHT_MLP1_BIAS, scale_layer_bias + SCALE_MLP1_BIAS, bias_layer + BIAS_MLP1, MLP1_ROWS, MLP1_COLS, device_id, false, bias_onboard, linear_alpha);
        Gelu_layer(float_buffer, int_buffer, i_seq);
       
        GEMM_QUANT({MEM_PORT_CALL}
                     stream_previous, stream_next, int_buffer, float_buffer, weight_layer_bias + WEIGHT_MLP2_BIAS, scale_layer_bias + SCALE_MLP2_BIAS, bias_layer + BIAS_MLP2, MLP2_ROWS, MLP2_COLS, device_id, true, bias_onboard, linear_alpha);
        Acc_layer(float_buffer, res_buffer);

        for (int i = 0; i < FULL_INP_LEN; i++)
        {
#pragma HLS UNROLL factor = INP_PARALLEL
            input_buffer[i] = res_buffer[i];
        }
    }

    output_writer_memory(res_buffer, host_addr);
}