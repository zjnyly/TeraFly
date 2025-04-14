void accumulate_manager(
    hls::stream<int> & inp,
    hls::stream<int> & outp,
    int ROWS, int COLS)
{
    for(int row = 0; row < ROWS; row++)
    {
        int acc_data = 0;
        for(int col = 0; col < COLS; col++)
        {
#pragma HLS PIPELINE II=1
            acc_data += inp.read();
        }
        outp.write(acc_data);
    }
}

void acc_result_merger(
    hls::stream<int> inp[PROCESSOR], 
    hls::stream<proc_pack_int> & outp,
    int ROWS)
{
    for(int row = 0; row < ROWS; row++)
    {
#pragma HLS PIPELINE II=1
        proc_pack_int data_pack;
        for(int PE = 0; PE < PROCESSOR; PE++)
        {
#pragma HLS UNROLL
            int data = inp[PE].read();
            data_pack.range(32 * PE + 31, 32 * PE) = data;
        }
        outp.write(data_pack);
    }
}

void weight_loader(
    io_pack_int8 * w_addr,
    hls::stream<io_pack_int8> & w_loader, 
    int bias, int ITERS)
{
loop_over_seq:
    for (int iter = 0; iter < ITERS ; iter++)
    {
#pragma HLS PIPELINE II = 1
        io_pack_int8 data_pack = w_addr[bias + iter];
        w_loader.write(data_pack);
    }
}

void stream_copy(
    int8_t input_buffer_int8[FULL_INP_LEN],
    hls::stream<io_pack_int8> dispacher[PROCESSOR],
    int ROWS, int COLS)
{
    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
#pragma HLS PIPELINE II = 1
            io_pack_int8 data_pack;
            for (int parallel = 0; parallel < INP_NUM; parallel++)
            {
#pragma HLS UNROLL
                data_pack.range(parallel * 8 + 7, parallel * 8) = input_buffer_int8[col * INP_NUM + parallel];
            }
            for (int PE = 0; PE < PROCESSOR; PE++)
            {
#pragma HLS UNROLL
                dispacher[PE].write(data_pack);
            }
        }
    }
}


void requant(
    hls::stream<router_pack_int> & output,
    hls::stream<router_pack_float> & for_router,
    int BIAS, int SCALE_BIAS, int ROWS, bool isFloat, float bias_onboard[BIAS_SIZE_TOTAL], float linear_alpha[108])
{
    for (int row = 0; row < ROWS / ROUNDS_FROM; row++)
    {
        router_pack_int acc_pack = output.read();
        router_pack_float data_pack;
        for (int parallel = 0; parallel < ROUTE_NUM; parallel++)
        {
#pragma HLS PIPELINE II = 1
            int bias_idx = row * ROUTE_NUM + parallel;
            int scale_bias = SCALE_BIAS + bias_idx / (INP_LEN);
            int acc = acc_pack.range(32 * parallel + 31, 32 * parallel);
            float res = acc * linear_alpha[scale_bias];
            float plus_bias = res + bias_onboard[BIAS + bias_idx];
            float result;
            if(isFloat)
            {
                result = plus_bias;
            }
            else if(plus_bias <= -128)
            {
            
                result = 0;
            }
            else if (plus_bias >= 127)
            {
                result = 127;
            }
            else
            {
                result = plus_bias;
            }
            converter_t converter;
            converter.f = result;
            data_pack.range(parallel * 32 + 31, parallel * 32) = converter.i;
        }
        for_router.write(data_pack);
    }
}


void GEMM_QUANT(
{MEM_PORT_REGION}
    hls::stream<router_pack_float> &stream_previous,
    hls::stream<router_pack_float> &stream_next,
    int8_t * input_buffer_int8,
    float * output_buffer_float,
    int WEIGHT_BIAS,
    int SCALE_BIAS,
    int BIAS,
    int ROWS, int COLS,
    const int device,
    bool isFloat, float bias_onboard[BIAS_SIZE_TOTAL], float linear_alpha[9 * NUM_LAYER])
{

#pragma HLS DATAFLOW
    hls::stream<io_pack_int8> block_w_loader[PROCESSOR];
#pragma HLS STREAM variable = block_w_loader depth = 2
#pragma HLS BIND_STORAGE variable = block_w_loader type = fifo impl = srl
    hls::stream<io_pack_int8> dispacher[PROCESSOR];
#pragma HLS STREAM variable = dispacher depth = 2
#pragma HLS BIND_STORAGE variable = dispacher type = fifo impl = srl
    hls::stream<int> adder_tree_output[PROCESSOR];
#pragma HLS STREAM variable = adder_tree_output depth = 2
#pragma HLS BIND_STORAGE variable = adder_tree_output type = fifo impl = srl
    hls::stream<int> output[PROCESSOR];
#pragma HLS STREAM variable = output depth = 128
#pragma HLS BIND_STORAGE variable = output type = fifo impl = bram
    hls::stream<proc_pack_int> packed_output;
#pragma HLS STREAM variable = packed_output depth = 128
#pragma HLS BIND_STORAGE variable = packed_output type = fifo impl = bram
    hls::stream<router_pack_int> repacked_output;
#pragma HLS STREAM variable = repacked_output depth = 128
#pragma HLS BIND_STORAGE variable = repacked_output type = fifo impl = bram
    hls::stream<router_pack_float> for_router;
#pragma HLS STREAM variable = for_router depth = 32
#pragma HLS BIND_STORAGE variable = for_router type = fifo impl = srl
    hls::stream<router_pack_float> from_router;
#pragma HLS STREAM variable = from_router depth = 32
#pragma HLS BIND_STORAGE variable = from_router type = fifo impl = srl
    
{WEIGHT_LOADER_REGION}

    stream_copy(input_buffer_int8, dispacher, ROWS, COLS);
    for(int PE = 0; PE < PROCESSOR; PE++)
    {
#pragma HLS UNROLL
        {Mul_Adder_Tree}(dispacher[PE], block_w_loader[PE], adder_tree_output[PE], ROWS * COLS);
        accumulate_manager(adder_tree_output[PE], output[PE], ROWS, COLS);
    }

{WEATHER_USE_ADAPTER}
    requant(repacked_output, for_router, BIAS, SCALE_BIAS, ROWS, isFloat, bias_onboard, linear_alpha);
{WEATHER_USE_ROUTER}
}

