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

#define NUM_LAYER  24
#define FULL_SEQ_NUM 1024
#define CU 2
#define seq_num FULL_SEQ_NUM/CU
#define INP_NUM 32
#define PROCESSOR 8
#define route_num 8

#define FULL_INP_LEN 2048
#define INP_LEN FULL_INP_LEN / CU

#define HEAD_NUM 32
#define HEAD_LEN 64

#define INP_PARALLEL 8


typedef ap_int<24> int24_t;


// typedef ap_uint<8 * block_size> block_pack_int8;
// typedef ap_uint<24 * block_size> block_pack_int24;
// typedef ap_uint<32 * block_size> block_pack_float;


typedef ap_uint<32 * INP_PARALLEL> parallel_pack_float;
typedef ap_uint<8 * INP_PARALLEL> parallel_pack_int8;
typedef ap_uint<32 * INP_NUM> io_pack_float;
typedef ap_uint<32 * route_num> router_pack_float;
typedef ap_uint<24 * route_num> router_pack_int24;
typedef ap_uint<8 * INP_NUM> io_pack_int8;
typedef ap_uint<8 * INP_NUM * 2> double_io_pack_int8;
typedef ap_uint<24 * INP_NUM> io_pack_int24;
typedef ap_uint<32 * INP_NUM> io_pack_int32;
typedef ap_uint<24 * INP_NUM * 2> double_io_pack_int24;


typedef union {
  float f;
  uint32_t i;
} converter_t;


