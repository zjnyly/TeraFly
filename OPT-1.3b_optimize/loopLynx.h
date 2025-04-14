#include "params.h"
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
	hls::stream<router_pack_float> &stream_next);void loopLynx_1(
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
	hls::stream<router_pack_float> &stream_next);