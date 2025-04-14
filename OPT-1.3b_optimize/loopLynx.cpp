
#include "loopLynx.h"

void write_kv_buffer_onchip(
    float float_buffer[FULL_INP_LEN * 4],
    int8_t KV_buffer[INP_LEN / ATTENTION_CHANNELS],
    const int bias)
{
    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
        int local_buffer_bias = bias + head * (HEAD_LEN * HEAD_PARALLEL);
        for (int i_emb = 0; i_emb < INP_NUM; i_emb++)
        {
#pragma HLS UNROLL factor = 32
            KV_buffer[head * INP_NUM + i_emb] = int8_t(round(float_buffer[local_buffer_bias + i_emb]));
        }
    }
}

void kv_writer_parallel(
    int8_t KV_buffer[INP_LEN / 2],
    io_pack_int8 *addr,
    int BIAS,
    int seq_id)
{
    for (int head = 0; head < (HEAD_NUM / CU / HEAD_PARALLEL); head++)
    {
#pragma HLS PIPELINE II = 1
        io_pack_int8 data_pack;
        int local_buffer_bias = head * INP_NUM;
        int idx = BIAS + head * FULL_SEQ_NUM + seq_id;
        for (int i_emb = 0; i_emb < INP_NUM; i_emb++)
        {
#pragma HLS UNROLL factor = INP_NUM
            data_pack.range(i_emb * 8 + 7, i_emb * 8) = KV_buffer[local_buffer_bias + i_emb];
        }
        // std::cout<<idx<<std::endl;
        addr[idx] = data_pack;
    }
}

void Gelu_layer(
    float *inp,
    int8_t *outp,
    int seq_id)
{
    for (int serial = 0; serial < FULL_INP_LEN * 4; serial++)
    {
#pragma HLS UNROLL factor=32
        float raw = inp[serial];
        int8_t data = round(raw);
        if (data < 0)
        {
            data = 0;
        }
        outp[serial] = data;
    }
}

void Res_layer(
    float *inp_direct,
    float *inp_shortcut,
    float *res_buffer,
    hls::stream<parallel_pack_float> &outp)
{
    for (int i_dim = 0; i_dim < FULL_INP_LEN / INP_PARALLEL; i_dim++)
    {
#pragma HLS pipeline II = 1
        parallel_pack_float data_pack;
        for (int parallel = 0; parallel < INP_PARALLEL; parallel++)
        {
#pragma HLS UNROLL
            converter_t converter;
            int idx = i_dim * INP_PARALLEL + parallel;
            float direct = inp_direct[idx];
            float shortcut = inp_shortcut[idx];
            converter.f = direct + shortcut;
            res_buffer[idx] = converter.f;
            data_pack.range(parallel * 32 + 31, parallel * 32) = converter.i;
        }
        outp.write(data_pack);
    }
}

void Acc_layer(
    float *float_buffer,
    float *acc_buffer)
{
    for (int i_dim = 0; i_dim < FULL_INP_LEN; i_dim++)
    {
#pragma HLS UNROLL factor = INP_PARALLEL
        acc_buffer[i_dim] += float_buffer[i_dim];
    }
}


void input_loader_memory(
    float *inp_addr,
    float *inp_buffer_fp)
{
    for (int i_len = 0; i_len < FULL_INP_LEN; i_len++)
    {
#pragma HLS pipeline II = 1
        inp_buffer_fp[i_len] = inp_addr[i_len];
    }
}

void output_writer_memory(
    float *inp,
    float *outp_addr)
{
    for (int i_dim = 0; i_dim < FULL_INP_LEN; i_dim++)
    {
#pragma HLS pipeline II = 1
        outp_addr[i_dim] = inp[i_dim];
    }
}

void q_writer(
    float *input_buffer,
    int8_t *output_buffer,
    const int device_id)
{
    int bias = device_id * INP_LEN;
    for (int i_seq = 0; i_seq < INP_LEN; i_seq++)
    {
#pragma HLS UNROLL factor=INP_PARALLEL
        output_buffer[i_seq] = int8_t(round(input_buffer[bias + i_seq]));
    }
}


void set_buffer_zero(
    float * float_buffer)
{
    for (int i = 0; i < FULL_INP_LEN; i++)
    {
#pragma HLS UNROLL factor = INP_NUM
        float_buffer[i] = 0;
    }
}

void Mul_Adder_Tree_32(hls::stream<io_pack_int8> & in_0, hls::stream<io_pack_int8> & in_1, hls::stream<int> & out, int ROUNDS)
{
#pragma HLS DATAFLOW
	hls::stream<datapack_32> pass_32;
#pragma HLS STREAM variable = pass_32 depth = 2
#pragma HLS BIND_STORAGE variable = pass_32 type = fifo impl = srl
	hls::stream<datapack_16> pass_16;
#pragma HLS STREAM variable = pass_16 depth = 2
#pragma HLS BIND_STORAGE variable = pass_16 type = fifo impl = srl
	hls::stream<datapack_8> pass_8;
#pragma HLS STREAM variable = pass_8 depth = 2
#pragma HLS BIND_STORAGE variable = pass_8 type = fifo impl = srl
	hls::stream<datapack_4> pass_4;
#pragma HLS STREAM variable = pass_4 depth = 2
#pragma HLS BIND_STORAGE variable = pass_4 type = fifo impl = srl
	hls::stream<datapack_2> pass_2;
#pragma HLS STREAM variable = pass_2 depth = 2
#pragma HLS BIND_STORAGE variable = pass_2 type = fifo impl = srl

	for(int round = 0; round < ROUNDS; round++)
	{
#pragma HLS PIPELINE II=1
		auto data_pack_0 = in_0.read();
		auto data_pack_1 = in_1.read();
		datapack_32 to_pass;
		for (int parallel = 0; parallel < 32; parallel++)
		{
	#pragma HLS UNROLL
            int8_t a = data_pack_0.range(8 * parallel + 7, 8 * parallel);
            int8_t b = data_pack_1.range(8 * parallel + 7, 8 * parallel);
            ap_int<16> c = a * b;
            #pragma HLS BIND_OP variable=c op=mul impl=dsp
			to_pass.range(16 * parallel + 15, 16 * parallel) = c;
		}
		pass_32.write(to_pass);
	}

	for(int round = 0; round < ROUNDS; round++)
	{
#pragma HLS PIPELINE II=1
		auto data_pack = pass_32.read();
		datapack_16 to_pass;
		for (int parallel = 0; parallel < 16; parallel++)
		{
	#pragma HLS UNROLL
			int bias_0 = parallel * 2;
			int bias_1 = parallel * 2 + 1;
            ap_int<16> a = data_pack.range(bias_0 * 16 + 15, bias_0 * 16);
            ap_int<16> b = data_pack.range(bias_1 * 16 + 15, bias_1 * 16);
			to_pass.range(17 * parallel + 16, 17 * parallel) = a + b;
		}
		pass_16.write(to_pass);
	}

	for(int round = 0; round < ROUNDS; round++)
	{
#pragma HLS PIPELINE II=1
		auto data_pack = pass_16.read();
		datapack_8 to_pass;
		for (int parallel = 0; parallel < 8; parallel++)
		{
	#pragma HLS UNROLL
			int bias_0 = parallel * 2;
			int bias_1 = parallel * 2 + 1;
            ap_int<17> a = data_pack.range(bias_0 * 17 + 16, bias_0 * 17);
            ap_int<17> b = data_pack.range(bias_1 * 17 + 16, bias_1 * 17);
			to_pass.range(18 * parallel + 17, 18 * parallel) = a + b;
		}
		pass_8.write(to_pass);
	}

	for(int round = 0; round < ROUNDS; round++)
	{
#pragma HLS PIPELINE II=1
		auto data_pack = pass_8.read();
		datapack_4 to_pass;
		for (int parallel = 0; parallel < 4; parallel++)
		{
	#pragma HLS UNROLL
			int bias_0 = parallel * 2;
			int bias_1 = parallel * 2 + 1;
            ap_int<18> a = data_pack.range(bias_0 * 18 + 17, bias_0 * 18);
            ap_int<18> b = data_pack.range(bias_1 * 18 + 17, bias_1 * 18);
			to_pass.range(19 * parallel + 18, 19 * parallel) = a + b;
		}
		pass_4.write(to_pass);
	}

	for(int round = 0; round < ROUNDS; round++)
	{
#pragma HLS PIPELINE II=1
		auto data_pack = pass_4.read();
		datapack_2 to_pass;
		for (int parallel = 0; parallel < 2; parallel++)
		{
	#pragma HLS UNROLL
			int bias_0 = parallel * 2;
			int bias_1 = parallel * 2 + 1;
            ap_int<19> a = data_pack.range(bias_0 * 19 + 18, bias_0 * 19);
            ap_int<19> b = data_pack.range(bias_1 * 19 + 18, bias_1 * 19);
			to_pass.range(20 * parallel + 19, 20 * parallel) = a + b;
		}
		pass_2.write(to_pass);
	}

	for(int round = 0; round < ROUNDS; round++)
	{
#pragma HLS PIPELINE II=1
		auto data_pack = pass_2.read();
        ap_int<20> a = data_pack.range(39, 20);
        ap_int<20> b = data_pack.range(19, 0);
		int result = a + b;
		out.write(result);
	}
}


void Mul_Adder_Tree_64(hls::stream<io_pack_int8> & in_0, hls::stream<io_pack_int8> & in_1, hls::stream<int> & out, int ROUNDS)
{
#pragma HLS DATAFLOW
	hls::stream<datapack_64> pass_64;
#pragma HLS STREAM variable = pass_64 depth = 2
#pragma HLS BIND_STORAGE variable = pass_64 type = fifo impl = srl
	hls::stream<datapack_32> pass_32;
#pragma HLS STREAM variable = pass_32 depth = 2
#pragma HLS BIND_STORAGE variable = pass_32 type = fifo impl = srl
	hls::stream<datapack_16> pass_16;
#pragma HLS STREAM variable = pass_16 depth = 2
#pragma HLS BIND_STORAGE variable = pass_16 type = fifo impl = srl
	hls::stream<datapack_8> pass_8;
#pragma HLS STREAM variable = pass_8 depth = 2
#pragma HLS BIND_STORAGE variable = pass_8 type = fifo impl = srl
	hls::stream<datapack_4> pass_4;
#pragma HLS STREAM variable = pass_4 depth = 2
#pragma HLS BIND_STORAGE variable = pass_4 type = fifo impl = srl
	hls::stream<datapack_2> pass_2;
#pragma HLS STREAM variable = pass_2 depth = 2
#pragma HLS BIND_STORAGE variable = pass_2 type = fifo impl = srl

	for(int round = 0; round < ROUNDS; round++)
	{
#pragma HLS PIPELINE II=1
		auto data_pack_0 = in_0.read();
		auto data_pack_1 = in_1.read();
		datapack_64 to_pass;
		for (int parallel = 0; parallel < 64; parallel++)
		{
	#pragma HLS UNROLL
            int8_t a = data_pack_0.range(8 * parallel + 7, 8 * parallel);
            int8_t b = data_pack_1.range(8 * parallel + 7, 8 * parallel);
            ap_int<16> c = a * b;
            #pragma HLS BIND_OP variable=c op=mul impl=dsp
			to_pass.range(16 * parallel + 15, 16 * parallel) = c;
		}
		pass_64.write(to_pass);
	}

	for(int round = 0; round < ROUNDS; round++)
	{
#pragma HLS PIPELINE II=1
		auto data_pack = pass_64.read();
		datapack_32 to_pass;
		for (int parallel = 0; parallel < 32; parallel++)
		{
	#pragma HLS UNROLL
			int bias_0 = parallel * 2;
			int bias_1 = parallel * 2 + 1;
            ap_int<16> a = data_pack.range(bias_0 * 16 + 15, bias_0 * 16);
            ap_int<16> b = data_pack.range(bias_1 * 16 + 15, bias_1 * 16);
			to_pass.range(17 * parallel + 16, 17 * parallel) = a + b;
		}
		pass_32.write(to_pass);
	}

	for(int round = 0; round < ROUNDS; round++)
	{
#pragma HLS PIPELINE II=1
		auto data_pack = pass_32.read();
		datapack_16 to_pass;
		for (int parallel = 0; parallel < 16; parallel++)
		{
	#pragma HLS UNROLL
			int bias_0 = parallel * 2;
			int bias_1 = parallel * 2 + 1;
            ap_int<17> a = data_pack.range(bias_0 * 17 + 16, bias_0 * 17);
            ap_int<17> b = data_pack.range(bias_1 * 17 + 16, bias_1 * 17);
			to_pass.range(18 * parallel + 17, 18 * parallel) = a + b;
		}
		pass_16.write(to_pass);
	}

	for(int round = 0; round < ROUNDS; round++)
	{
#pragma HLS PIPELINE II=1
		auto data_pack = pass_16.read();
		datapack_8 to_pass;
		for (int parallel = 0; parallel < 8; parallel++)
		{
	#pragma HLS UNROLL
			int bias_0 = parallel * 2;
			int bias_1 = parallel * 2 + 1;
            ap_int<18> a = data_pack.range(bias_0 * 18 + 17, bias_0 * 18);
            ap_int<18> b = data_pack.range(bias_1 * 18 + 17, bias_1 * 18);
			to_pass.range(19 * parallel + 18, 19 * parallel) = a + b;
		}
		pass_8.write(to_pass);
	}

	for(int round = 0; round < ROUNDS; round++)
	{
#pragma HLS PIPELINE II=1
		auto data_pack = pass_8.read();
		datapack_4 to_pass;
		for (int parallel = 0; parallel < 4; parallel++)
		{
	#pragma HLS UNROLL
			int bias_0 = parallel * 2;
			int bias_1 = parallel * 2 + 1;
            ap_int<19> a = data_pack.range(bias_0 * 19 + 18, bias_0 * 19);
            ap_int<19> b = data_pack.range(bias_1 * 19 + 18, bias_1 * 19);
			to_pass.range(20 * parallel + 19, 20 * parallel) = a + b;
		}
		pass_4.write(to_pass);
	}

	for(int round = 0; round < ROUNDS; round++)
	{
#pragma HLS PIPELINE II=1
		auto data_pack = pass_4.read();
		datapack_2 to_pass;
		for (int parallel = 0; parallel < 2; parallel++)
		{
	#pragma HLS UNROLL
			int bias_0 = parallel * 2;
			int bias_1 = parallel * 2 + 1;
            ap_int<20> a = data_pack.range(bias_0 * 20 + 19, bias_0 * 20);
            ap_int<20> b = data_pack.range(bias_1 * 20 + 19, bias_1 * 20);
			to_pass.range(21 * parallel + 20, 21 * parallel) = a + b;
		}
		pass_2.write(to_pass);
	}

	for(int round = 0; round < ROUNDS; round++)
	{
#pragma HLS PIPELINE II=1
		auto data_pack = pass_2.read();
        ap_int<21> a = data_pack.range(41, 21);
        ap_int<21> b = data_pack.range(20, 0);
		int result = a + b;
		out.write(result);
	}
}


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
    Mul_Adder_Tree_64(head_loader, weight_loader, sub_acc_result, seq_id + 1);
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
                int acc_result = sub_result[parallel];

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
    typedef ap_fixed<32 + 10, 10> sfm_fixed;
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
#pragma HLS STREAM variable = inp_value_bypass depth = 6144
#pragma HLS BIND_STORAGE variable = inp_value_bypass type = fifo impl = uram
    hls::stream<float> inp_value_normalized_bypass;
#pragma HLS STREAM variable = inp_value_normalized_bypass depth = 6144
#pragma HLS BIND_STORAGE variable = inp_value_normalized_bypass type = fifo impl = uram

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
	io_pack_int8 *w_addr_0,
	io_pack_int8 *w_addr_1,
	io_pack_int8 *w_addr_2,
	io_pack_int8 *w_addr_3,

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

#include "const_data/attn_alpha.txt"
#include "const_data/ctx_alpha.txt"

	loader_new(w_addr_0, k_stream[0], layer_id, seq_id);
	loader_new(w_addr_1, k_stream[1], layer_id, seq_id);
	loader_new(w_addr_2, v_stream[0], layer_id, seq_id);
	loader_new(w_addr_3, v_stream[1], layer_id, seq_id);

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
	router_attention(ctx_requant_merged, stream_previous, stream_next, router_out);
	write_buffer_int(router_out, output_buffer, device_id);

}














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
	router_ln(for_router, stream_previous, stream_next, from_router);
	write_buffer_ln(from_router, output, device);
}



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
	io_pack_int8 *w_addr_0,
	io_pack_int8 *w_addr_1,
	io_pack_int8 *w_addr_2,
	io_pack_int8 *w_addr_3,
	io_pack_int8 *w_addr_4,
	io_pack_int8 *w_addr_5,
	io_pack_int8 *w_addr_6,
	io_pack_int8 *w_addr_7,

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
    
	weight_loader(w_addr_0, block_w_loader[0], WEIGHT_BIAS, ROWS * COLS);
	weight_loader(w_addr_1, block_w_loader[1], WEIGHT_BIAS, ROWS * COLS);
	weight_loader(w_addr_2, block_w_loader[2], WEIGHT_BIAS, ROWS * COLS);
	weight_loader(w_addr_3, block_w_loader[3], WEIGHT_BIAS, ROWS * COLS);
	weight_loader(w_addr_4, block_w_loader[4], WEIGHT_BIAS, ROWS * COLS);
	weight_loader(w_addr_5, block_w_loader[5], WEIGHT_BIAS, ROWS * COLS);
	weight_loader(w_addr_6, block_w_loader[6], WEIGHT_BIAS, ROWS * COLS);
	weight_loader(w_addr_7, block_w_loader[7], WEIGHT_BIAS, ROWS * COLS);


    stream_copy(input_buffer_int8, dispacher, ROWS, COLS);
    for(int PE = 0; PE < PROCESSOR; PE++)
    {
#pragma HLS UNROLL
        Mul_Adder_Tree_64(dispacher[PE], block_w_loader[PE], adder_tree_output[PE], ROWS * COLS);
        accumulate_manager(adder_tree_output[PE], output[PE], ROWS, COLS);
    }

	acc_result_merger(output, repacked_output, ROWS);

    requant(repacked_output, for_router, BIAS, SCALE_BIAS, ROWS, isFloat, bias_onboard, linear_alpha);
	router(for_router, stream_previous, stream_next, from_router, ROWS);
	write_buffer(from_router, output_buffer_float, device, ROWS);
}







void loopLynx_0(
    int i_seq,
    float *host_addr,
    int load_weight_layer,
	io_pack_int8 *w_addr_0,
	io_pack_int8 *w_addr_1,
	io_pack_int8 *w_addr_2,
	io_pack_int8 *w_addr_3,
	io_pack_int8 *w_addr_4,
	io_pack_int8 *w_addr_5,
	io_pack_int8 *w_addr_6,
	io_pack_int8 *w_addr_7,

	hls::stream<router_pack_float> &stream_previous,
	hls::stream<router_pack_float> &stream_next)
{
#pragma HLS interface m_axi port = host_addr offset = slave bundle = gmem8 
#pragma HLS interface m_axi port = w_addr_0 offset = slave bundle = gmem0
#pragma HLS interface m_axi port = w_addr_1 offset = slave bundle = gmem1
#pragma HLS interface m_axi port = w_addr_2 offset = slave bundle = gmem2
#pragma HLS interface m_axi port = w_addr_3 offset = slave bundle = gmem3
#pragma HLS interface m_axi port = w_addr_4 offset = slave bundle = gmem4
#pragma HLS interface m_axi port = w_addr_5 offset = slave bundle = gmem5
#pragma HLS interface m_axi port = w_addr_6 offset = slave bundle = gmem6
#pragma HLS interface m_axi port = w_addr_7 offset = slave bundle = gmem7


    const int device_id = 0;

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
        GEMM_QUANT(w_addr_0, w_addr_1, w_addr_2, w_addr_3, w_addr_4, w_addr_5, w_addr_6, w_addr_7, 
                     stream_previous, stream_next, int_buffer, float_buffer, weight_layer_bias, scale_layer_bias, bias_layer, QKV_ROWS, QKV_COLS, device_id, false, bias_onboard, linear_alpha);
       
		write_kv_buffer_onchip(float_buffer, KV_buffer[0], KV_LOCAL_BIAS_0[0]);
		write_kv_buffer_onchip(float_buffer, KV_buffer[1], KV_LOCAL_BIAS_0[1]);
		write_kv_buffer_onchip(float_buffer, KV_buffer[2], KV_LOCAL_BIAS_0[2]);
		write_kv_buffer_onchip(float_buffer, KV_buffer[3], KV_LOCAL_BIAS_0[3]);

		kv_writer_parallel(KV_buffer[0], w_addr_0, memory_bias, i_seq);
		kv_writer_parallel(KV_buffer[1], w_addr_1, memory_bias, i_seq);
		kv_writer_parallel(KV_buffer[2], w_addr_2, memory_bias, i_seq);
		kv_writer_parallel(KV_buffer[3], w_addr_3, memory_bias, i_seq);

        
        q_writer(float_buffer, q_buffer, device_id);
        
        // compute attention
        attention_wrapper_new(q_buffer, w_addr_0, w_addr_1, w_addr_2, w_addr_3,  stream_previous, stream_next,
                              int_buffer, i_layer, i_seq, device_id);
        
        // compute O
        GEMM_QUANT(w_addr_0, w_addr_1, w_addr_2, w_addr_3, w_addr_4, w_addr_5, w_addr_6, w_addr_7, 
                     stream_previous, stream_next, int_buffer, float_buffer, weight_layer_bias + WEIGHT_O_BIAS, scale_layer_bias + SCALE_O_BIAS, bias_layer + BIAS_O, O_ROWS, O_COLS, device_id, true, bias_onboard, linear_alpha);

        // Res
        Fused_Res_LN_copy(input_buffer, float_buffer, res_buffer, fn_ln_bias, stream_previous, stream_next, int_buffer, device_id, ln_weight_onboard, ln_bias_onboard);

        GEMM_QUANT(w_addr_0, w_addr_1, w_addr_2, w_addr_3, w_addr_4, w_addr_5, w_addr_6, w_addr_7, 
                     stream_previous, stream_next, int_buffer, float_buffer, weight_layer_bias + WEIGHT_MLP1_BIAS, scale_layer_bias + SCALE_MLP1_BIAS, bias_layer + BIAS_MLP1, MLP1_ROWS, MLP1_COLS, device_id, false, bias_onboard, linear_alpha);
        Gelu_layer(float_buffer, int_buffer, i_seq);
       
        GEMM_QUANT(w_addr_0, w_addr_1, w_addr_2, w_addr_3, w_addr_4, w_addr_5, w_addr_6, w_addr_7, 
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

void loopLynx_1(
    int i_seq,
    float *host_addr,
    int load_weight_layer,
	io_pack_int8 *w_addr_0,
	io_pack_int8 *w_addr_1,
	io_pack_int8 *w_addr_2,
	io_pack_int8 *w_addr_3,
	io_pack_int8 *w_addr_4,
	io_pack_int8 *w_addr_5,
	io_pack_int8 *w_addr_6,
	io_pack_int8 *w_addr_7,

	hls::stream<router_pack_float> &stream_previous,
	hls::stream<router_pack_float> &stream_next)
{
#pragma HLS interface m_axi port = host_addr offset = slave bundle = gmem8 
#pragma HLS interface m_axi port = w_addr_0 offset = slave bundle = gmem0
#pragma HLS interface m_axi port = w_addr_1 offset = slave bundle = gmem1
#pragma HLS interface m_axi port = w_addr_2 offset = slave bundle = gmem2
#pragma HLS interface m_axi port = w_addr_3 offset = slave bundle = gmem3
#pragma HLS interface m_axi port = w_addr_4 offset = slave bundle = gmem4
#pragma HLS interface m_axi port = w_addr_5 offset = slave bundle = gmem5
#pragma HLS interface m_axi port = w_addr_6 offset = slave bundle = gmem6
#pragma HLS interface m_axi port = w_addr_7 offset = slave bundle = gmem7


    const int device_id = 1;

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
        GEMM_QUANT(w_addr_0, w_addr_1, w_addr_2, w_addr_3, w_addr_4, w_addr_5, w_addr_6, w_addr_7, 
                     stream_previous, stream_next, int_buffer, float_buffer, weight_layer_bias, scale_layer_bias, bias_layer, QKV_ROWS, QKV_COLS, device_id, false, bias_onboard, linear_alpha);
       
		write_kv_buffer_onchip(float_buffer, KV_buffer[0], KV_LOCAL_BIAS_1[0]);
		write_kv_buffer_onchip(float_buffer, KV_buffer[1], KV_LOCAL_BIAS_1[1]);
		write_kv_buffer_onchip(float_buffer, KV_buffer[2], KV_LOCAL_BIAS_1[2]);
		write_kv_buffer_onchip(float_buffer, KV_buffer[3], KV_LOCAL_BIAS_1[3]);

		kv_writer_parallel(KV_buffer[0], w_addr_0, memory_bias, i_seq);
		kv_writer_parallel(KV_buffer[1], w_addr_1, memory_bias, i_seq);
		kv_writer_parallel(KV_buffer[2], w_addr_2, memory_bias, i_seq);
		kv_writer_parallel(KV_buffer[3], w_addr_3, memory_bias, i_seq);

        
        q_writer(float_buffer, q_buffer, device_id);
        
        // compute attention
        attention_wrapper_new(q_buffer, w_addr_0, w_addr_1, w_addr_2, w_addr_3,  stream_previous, stream_next,
                              int_buffer, i_layer, i_seq, device_id);
        
        // compute O
        GEMM_QUANT(w_addr_0, w_addr_1, w_addr_2, w_addr_3, w_addr_4, w_addr_5, w_addr_6, w_addr_7, 
                     stream_previous, stream_next, int_buffer, float_buffer, weight_layer_bias + WEIGHT_O_BIAS, scale_layer_bias + SCALE_O_BIAS, bias_layer + BIAS_O, O_ROWS, O_COLS, device_id, true, bias_onboard, linear_alpha);

        // Res
        Fused_Res_LN_copy(input_buffer, float_buffer, res_buffer, fn_ln_bias, stream_previous, stream_next, int_buffer, device_id, ln_weight_onboard, ln_bias_onboard);

        GEMM_QUANT(w_addr_0, w_addr_1, w_addr_2, w_addr_3, w_addr_4, w_addr_5, w_addr_6, w_addr_7, 
                     stream_previous, stream_next, int_buffer, float_buffer, weight_layer_bias + WEIGHT_MLP1_BIAS, scale_layer_bias + SCALE_MLP1_BIAS, bias_layer + BIAS_MLP1, MLP1_ROWS, MLP1_COLS, device_id, false, bias_onboard, linear_alpha);
        Gelu_layer(float_buffer, int_buffer, i_seq);
       
        GEMM_QUANT(w_addr_0, w_addr_1, w_addr_2, w_addr_3, w_addr_4, w_addr_5, w_addr_6, w_addr_7, 
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

