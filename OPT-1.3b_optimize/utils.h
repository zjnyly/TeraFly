#pragma once
#include "loopLynx.h"
#include <iostream>
#include <fstream>

using namespace std;
void load_io_pack_float_data(io_pack_float * buffer, string fileName, int buffer_size){
    std::ifstream file(fileName, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
    }

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint32_t> data(fileSize / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    file.close();

    for(int i = 0; i < buffer_size; i++){
        for(int j = 0; j < INP_NUM; j++){
            buffer[i].range(j * 32 + 31, j * 32) = data[i * INP_NUM + j];
        }

    }
}


void load_float_data(float * buffer, string fileName, int buffer_size, int bias = 0){
    std::ifstream file(fileName, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
    }

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> data(fileSize / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    file.close();

    // std::cout<<fileSize<<std::endl;
    // std::cout<<buffer_size<<std::endl;

    for(int i = 0; i < buffer_size; i++){
        buffer[i] = data[i + bias];
        // std::cout<<data[i]<<std::endl;
    }
    
}

void read_float_data(float * buffer, int buffer_size){
    for(int i = 0; i < buffer_size; i++){
            std::cout<< buffer[i] <<", ";
        
    }
    std::cout<<std::endl;
}

void read_io_pack_int8_data(io_pack_int8 * buffer, int buffer_size, int bias){
    for(int i = bias; i < buffer_size; i++){
        std::cout<<"line "<<i<<"    ";
        for(int j = 0; j < INP_NUM; j++){
            cout<< (int)int8_t(buffer[i].range(j * 8 + 7, j * 8)) <<" ";
        }
        std::cout<<std::endl;
    }
    cout<<endl;
}


void load_io_pack_int8_data(io_pack_int8 * buffer, string fileName, int buffer_size){
    std::ifstream file(fileName, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
    }

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);


    std::vector<int64_t> data(fileSize / sizeof(int64_t));
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    file.close();

    // std::cout<<"fileSize " << fileSize<<std::endl;
    // std::cout<<buffer_size * INP_NUM <<std::endl;


    for(int i = 0; i < buffer_size; i++){
        for(int j = 0; j < INP_NUM / 8; j++){
            buffer[i].range(j * 64 + 63, j * 64) = data[i * (INP_NUM / 8) + j];
           
        }

    }
}


void load_io_pack_int8_data_mcpy(io_pack_int8 * buffer, string fileName, int buffer_size){
    std::ifstream file(fileName, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
    }

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);


    std::vector<int64_t> data(fileSize / sizeof(int64_t));
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    file.close();

    // std::cout<<"fileSize " << fileSize<<std::endl;
    // std::cout<<buffer_size * INP_NUM <<std::endl;


    for(int i = 0; i < buffer_size; i++){
        for(int j = 0; j < INP_NUM / 8; j++){
            buffer[i].range(j * 64 + 63, j * 64) = data[i * (INP_NUM / 8) + j];
           
        }

    }
}

void read_binary_weights(const std::string &filename, std::vector<float> &weights, size_t size)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    weights.resize(size);
    file.read(reinterpret_cast<char *>(weights.data()), size * sizeof(float));
    if (!file)
    {
        std::cerr << "Error reading from file: " << filename << std::endl;
    }
}

