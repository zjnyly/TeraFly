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

