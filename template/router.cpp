
void router_ln(
    hls::stream<router_pack_float> &in,
    hls::stream<router_pack_float> &stream_previous,
    hls::stream<router_pack_float> &stream_next,
    hls::stream<router_pack_float> &out)
{
    router_pack_float buffer[8];
    hls::stream<int> delay;
#pragma HLS STREAM variable = delay depth = 2
    for (int round = 0; round < INP_LEN / 64; round++)
    {
    loop_over_block:
        for (int j = 0; j < 8; j++)
        {
            router_pack_float from_previous = in.read();
            buffer[j] = from_previous;
        }

        for (int i = 0; i < CU; i++)
        {
#pragma HLS PIPELINE off
            for (int j = 0; j < 8; j++)
            {
#pragma HLS PIPELINE
                router_pack_float data_pack;
                data_pack = buffer[j];
                stream_next.write(data_pack);
            }

            delay.write(1);
            delay.read();

            for (int j = 0; j < 8; j++)
            {
#pragma HLS PIPELINE
                router_pack_float data_pack;
                data_pack = stream_previous.read();
                buffer[j] = data_pack;
                out.write(data_pack);
            }
        }
    }
}

void write_buffer_ln(
    hls::stream<router_pack_float> &from_router,
    int8_t output_buffer_int[FULL_INP_LEN], 
    // float output_buffer_float[FULL_INP_LEN], 
    const int device)
{
    const int device_bias = (device + 1) * INP_LEN;

    for(int iter = 0; iter < (INP_LEN / 64); iter++)
    {
        for(int cu = 0; cu < CU; cu++)
        {
            int bias = (device_bias + cu * INP_LEN) % FULL_INP_LEN + iter * 64;

            for(int data_transfer = 0; data_transfer < 8; data_transfer++) 
            {
                router_pack_float data_pack = from_router.read();
                for(int idx = 0; idx < ROUTE_NUM; idx++)
                {
#pragma HLS UNROLL
                    converter_t converter;
                    converter.i = data_pack.range(32 * idx + 31, 32 * idx); 
                    int IDX = bias + data_transfer * ROUTE_NUM + idx;
                    float result;
                    if(converter.f >= 127){ result = 127;}
                    else if(converter.f <= -128){ result = -128;}
                    else{result = converter.f;}

                    output_buffer_int[IDX] = round(result);
                    // output_buffer_float[IDX] = converter.f;
                }
            }
        }
    }
}



void write_buffer(
    hls::stream<router_pack_float> &from_router,
    float output_buffer_float[FULL_INP_LEN],
    const int device, int ROWS)
{   
    int STRIDE = ROWS * (PROCESSOR);
    int FULL_LEN = STRIDE * CU;
    int DEVICE_IDX = (device + 1) * STRIDE;
    int CU_IDX[CU];
    for(int cu = 0; cu < CU; cu++)
    {
#pragma HLS PIPELINE II=1
        CU_IDX[cu] = cu * STRIDE;
    }
    
    // (ROWS * 3) / (4 * 8)
    int iters = (ROWS * ROUNDS_TO) / (ROUNDS_FROM * 8);
    for(int row = 0; row < iters; row++)
    {
        for(int cu = 0; cu < CU; cu++)
        {
            int BASE_IDX = (DEVICE_IDX + CU_IDX[cu]) % FULL_LEN;
            for(int data_transfer = 0; data_transfer < 8; data_transfer++) 
            {
                router_pack_float data_pack = from_router.read();
                for(int idx = 0; idx < ROUTE_NUM; idx++)
                {
#pragma HLS PIPELINE II=1
                    converter_t converter;
                    converter.i = data_pack.range(32 * idx + 31, 32 * idx); 
                    output_buffer_float[BASE_IDX + row * 64 + data_transfer * ROUTE_NUM + idx] = converter.f;
                }
            }
        }
    }


}

void router(
    hls::stream<router_pack_float> &in,
    hls::stream<router_pack_float> &stream_previous,
    hls::stream<router_pack_float> &stream_next,
    hls::stream<router_pack_float> &out,
    int ROWS)
{
    router_pack_float buffer[8];
    hls::stream<int> delay;
#pragma HLS STREAM variable = delay depth = 2
    // int rounds = {ROUNDS_FOR_WRITE_BUFFER};
    int rounds = (ROWS * ROUNDS_TO) / (ROUNDS_FROM * 8);
    for (int round = 0; round < rounds; round++)
    {
    loop_over_block:
        for (int j = 0; j < 8; j++)
        {
            router_pack_float from_previous = in.read();
            buffer[j] = from_previous;
        }

        for (int i = 0; i < CU; i++)
        {
#pragma HLS PIPELINE off
            for (int j = 0; j < 8; j++)
            {
#pragma HLS PIPELINE
                router_pack_float data_pack;
                data_pack = buffer[j];
                stream_next.write(data_pack);
            }

            delay.write(1);
            delay.read();

            for (int j = 0; j < 8; j++)
            {
#pragma HLS PIPELINE
                router_pack_float data_pack;
                data_pack = stream_previous.read();
                buffer[j] = data_pack;
                out.write(data_pack);
            }
        }
    }
}


void write_buffer_int(
    hls::stream<router_pack_float> &from_router,
    int8_t output_buffer_int[FULL_INP_LEN],
    const int device)
{
    const int device_bias = (device + 1) * INP_LEN;
    constexpr int flits = ATTENTION_CHANNELS * INP_NUM / ROUTE_NUM;
    
    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
        for (int cu = 0; cu < CU; cu++)
        {
            int bias = (device_bias + cu * INP_LEN) % FULL_INP_LEN + head * (INP_NUM * ATTENTION_CHANNELS);
            for (int i = 0; i < flits /*num_packets*/; i++)
            {
                router_pack_float data_pack = from_router.read();
                for (int j = 0; j < ROUTE_NUM; j++)
                {
#pragma HLS UNROLL factor = ROUTE_NUM
                    int idx = i * ROUTE_NUM + j;
                    converter_t converter;
                    converter.i = data_pack.range(j * 32 + 31, j * 32);
                    output_buffer_int[bias + idx] = int8_t(round(converter.f));
                }
            }
        }
    }
}

void router_attention(
    hls::stream<router_pack_float> &in,
    hls::stream<router_pack_float> &stream_previous,
    hls::stream<router_pack_float> &stream_next,
    hls::stream<router_pack_float> &out)
{
    constexpr int ROUNDS = (HEAD_NUM / CU / HEAD_PARALLEL);
    constexpr int flits = ATTENTION_CHANNELS * INP_NUM / ROUTE_NUM;
    router_pack_float buffer[flits];
    hls::stream<int> delay;
#pragma HLS STREAM variable = delay depth = 2

    for (int round = 0; round < ROUNDS; round++)
    {
    loop_over_block:
        for (int j = 0; j < flits; j++)
        {
            router_pack_float from_previous = in.read();
            buffer[j] = from_previous;
        }

        for (int i = 0; i < CU; i++)
        {
#pragma HLS PIPELINE off
            for (int j = 0; j < flits; j++)
            {
#pragma HLS PIPELINE
                router_pack_float data_pack;
                data_pack = buffer[j];
                stream_next.write(data_pack);
            }

            delay.write(1);
            delay.read();

            for (int j = 0; j < flits; j++)
            {
#pragma HLS PIPELINE
                router_pack_float data_pack;
                data_pack = stream_previous.read();
                buffer[j] = data_pack;
                out.write(data_pack);
            }
        }
    }
}