
void layerNorm(
    hls::stream<parallel_pack_float> &inp,
    hls::stream<router_pack_float> &outp,
    int LN_BIAS, float ln_weight[LN_SIZE_TOTAL][INP_PARALLEL], float ln_bias[LN_SIZE_TOTAL][INP_PARALLEL], const int device_id)
{

    int LN_INNER_BIAS = device_id * INP_LEN;
    float buf[FULL_INP_LEN];
    float mean_arr[INP_PARALLEL] = {0.0};
    float var_arr[INP_PARALLEL] = {0.0};

    for (int i_emb = 0; i_emb < FULL_INP_LEN / INP_PARALLEL; i_emb++)
    {
#pragma HLS pipeline II = 1
        parallel_pack_float data_pack = inp.read();
        for (int parallel = 0; parallel < INP_PARALLEL; parallel++)
        {
#pragma HLS UNROLL
            converter_t converter;
            converter.i = data_pack.range(parallel * 32 + 31, parallel * 32);
            float data = converter.f;
            buf[i_emb * INP_PARALLEL + parallel] = data;
            mean_arr[parallel] += data;
            float square = data * data;
            var_arr[parallel] += square;
        }
    }

    float mean = 0.0;
    float var = 0.0;

    for(int serial = 0; serial < INP_PARALLEL; serial++)
    {
#pragma HLS pipeline II=1
        mean += mean_arr[serial];
        var += var_arr[serial];
    }
    
    mean *= INV_FULL_INP_LEN; 
    float mean_square = (float)mean * (float)mean;
    var *= INV_FULL_INP_LEN;
    var -= mean_square;
    const float eps = 0.000010;
    float denomenator_f = var + eps;
    float sqrt_denomenator_rev = 1 / sqrt(denomenator_f);

    for (int i_emb = 0; i_emb < INP_LEN / INP_PARALLEL; i_emb++)
    {
        router_pack_float data_pack;
        for (int parallel = 0; parallel < INP_PARALLEL; parallel++)
        {
// #pragma HLS UNROLL factor=INP_PARALLEL/2
#pragma HLS UNROLL
            int BIAS = i_emb * INP_PARALLEL + parallel;
            int BIAS_LN = i_emb + LN_BIAS;
            float data_temp_1 = (buf[BIAS + LN_INNER_BIAS] - mean) * sqrt_denomenator_rev;
            float data_temp_2 = data_temp_1 * ln_weight[BIAS_LN][parallel] + ln_bias[BIAS_LN][parallel];
            converter_t converter;
            converter.f = data_temp_2;
            data_pack.range(32 * parallel + 31, 32 * parallel) = converter.i;
        }
        outp.write(data_pack);
    }
}

void Fused_Res_LN_copy(
    float *inp_shortcut,
    float *inp_direct,
    float *res_buffer,
    int LN_BIAS,
    hls::stream<router_pack_float> &stream_previous,
    hls::stream<router_pack_float> &stream_next,
    int8_t * output, 
    const int device, 
    float ln_weight[LN_SIZE_TOTAL][INP_PARALLEL], 
    float ln_bias[LN_SIZE_TOTAL][INP_PARALLEL])
{
#pragma HLS DATAFLOW
    hls::stream<parallel_pack_float> res_out;
#pragma HLS STREAM variable = res_out depth = 16
#pragma HLS BIND_STORAGE variable = res_out type = fifo impl = srl
    hls::stream<parallel_pack_float> for_router;
#pragma HLS STREAM variable = for_router depth = 16
#pragma HLS BIND_STORAGE variable = for_router type = fifo impl = srl
    hls::stream<parallel_pack_float> from_router;
#pragma HLS STREAM variable = from_router depth = 16
#pragma HLS BIND_STORAGE variable = from_router type = fifo impl = srl
    Res_layer(inp_direct, inp_shortcut, res_buffer, res_out);
    layerNorm(res_out, for_router, LN_BIAS, ln_weight, ln_bias, device);
{WEATHER_USE_ROUTER}
}