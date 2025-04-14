#pragma once

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
#include <hls_half.h>

const int NUM_LAYER = 24;
const int FULL_SEQ_NUM = 1024;
const int CU = 2;
const int INP_NUM = 64;
const int PROCESSOR = 8;
const int ROUTE_NUM = 8;
const int INP_PARALLEL = 8;
const int HEAD_PARALLEL = 2;
const int HEAD_NUM = 32;
const int HEAD_LEN = 64;
const int FULL_INP_LEN = 2048;
const int ROUNDS_FROM = 1;
const int ROUNDS_TO = 1;
const int ATTENTION_CHANNELS = 2;
const float INV_FULL_INP_LEN = 0.00048828125;
const int INP_LEN = 1024;
const int DECODE_ITERS = 201088;
const int WEIGHT_SIZE = 4096;
const int WEIGHT_SIZE_LAYER = 49152;
const int WEIGHT_SIZE_TOTAL = 1179648;
const int SCALING_FACTOR_SIZE_LAYER = 9;
const int SCALING_FACTOR_SIZE_TOTAL = 216;
const int BIAS_SIZE_LAYER = 9216;
const int BIAS_SIZE_TOTAL = 221184;
const int LN_BIAS_HALF = 128;
const int LN_BIAS_FULL = 2048;
const int LN_SIZE_TOTAL = 6144;
const int LN_SIZE_LAYER = 256;
const int MEMORY_SIZE = 1179648;
const int KV_CACHE_SIZE = 8192;
const int QKV_ROWS = 384;
const int QKV_COLS = 32;
const int O_ROWS = 128;
const int O_COLS = 32;
const int MLP1_ROWS = 512;
const int MLP1_COLS = 32;
const int MLP2_ROWS = 128;
const int MLP2_COLS = 128;
const int WEIGHT_O_BIAS = 12288;
const int WEIGHT_MLP1_BIAS = 16384;
const int WEIGHT_MLP2_BIAS = 32768;
const int SCALE_O_BIAS = 3;
const int SCALE_MLP1_BIAS = 4;
const int SCALE_MLP2_BIAS = 8;
const int BIAS_O = 3072;
const int BIAS_MLP1 = 4096;
const int BIAS_MLP2 = 8192;
const int KV_LOCAL_BIAS_0[4] = {2048, 2112, 4096, 4160};
const int KV_LOCAL_BIAS_1[4] = {3072, 3136, 5120, 5184};
typedef ap_uint<32 * PROCESSOR> proc_pack_float;
typedef ap_uint<32 * PROCESSOR> proc_pack_int;
typedef ap_uint<32 * INP_PARALLEL> parallel_pack_float;
typedef ap_uint<8 * INP_PARALLEL> parallel_pack_int8;
typedef ap_uint<32 * ROUTE_NUM> router_pack_float;
typedef ap_uint<32 * ROUTE_NUM> router_pack_int;
typedef ap_uint<32 * INP_NUM> io_pack_float;
typedef ap_uint<8 * INP_NUM> io_pack_int8;
typedef ap_uint<32 * INP_NUM> io_pack_int32;
typedef ap_uint<16 * 64> datapack_64;
typedef ap_uint<17 * 32> datapack_32;
typedef ap_uint<18 * 16> datapack_16;
typedef ap_uint<19 * 8> datapack_8;
typedef ap_uint<20 * 4> datapack_4;
typedef ap_uint<21 * 2> datapack_2;
typedef union {
	float f;
	uint32_t i;
} converter_t;
