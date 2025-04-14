
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
#pragma HLS UNROLL factor = INP_NUM
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
#pragma HLS UNROLL factor=INP_NUM
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