/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <cstdint>
#include <fstream>
#include <vector>
#include <unistd.h>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include "../gpu_memory.hpp"

#include <cuda_runtime.h>

void write_csv(std::ofstream& csv_file, std::vector<int>& nb_values, uint32_t* latencies, int num_searches, bool write_all) {
    if (csv_file.is_open()) {
        for (size_t nb_idx = 0; nb_idx < nb_values.size(); nb_idx++) {
            csv_file << nb_values[nb_idx];  // Write the nq value
            uint32_t sum = 0;
            for (int iter = 0; iter < num_searches; iter++) {
                uint32_t latency = *(latencies + nb_idx * num_searches + iter);
                if (write_all) {
                    csv_file << ", " << latency;
                }
                sum += latency;
            }
            double avg_latency = sum / static_cast<double>(num_searches);
            csv_file << ", " << avg_latency << "\n";
        }
        csv_file.close();
    } else {
        std::cerr << "Unable to open file for writing\n";
    }
}

int main(int argc, char* argv[]) {
    std::string data_path = "";
    int d = 128;      // dimension
    int nq = 50;      // batch size

    int opt;
    while ((opt = getopt(argc, argv, "p:d:n:")) != -1) {
        switch (opt) {
            case 'p':
                data_path = optarg;
                break;
            case 'd':
                d = std::atoi(optarg);
                break;
            case 'n':
                nq = std::atoi(optarg);
                break;
            default:
                break;
        }
    }
    if (data_path.empty()) {
        std::cerr << "./exec -p <data_path> -d <dimension> -n <nq>" << std::endl;
    }

    std::vector<int> nb_values = {50000, 100000, 500000, 1000000};
    int k = 4;

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    int num_searches = 10;  // Number of search iterations per nb configuration
    int num_warmups = 0;   // Number of warmup iterations per nb configuration

    // Latency storage: 2D array where rows are nb configurations and columns are iterations
    uint32_t latencies[nb_values.size()][num_searches];
    uint32_t memories[nb_values.size()][num_searches];

    // Allocate memory for xq based on current nq
    float* xq = new float[d * nq];
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    // Move xq to GPU
    float *xq_gpu;
    int xq_gpu_size = nq*d*sizeof(float);
    cudaError_t err = cudaMalloc((void **)&xq_gpu, (xq_gpu_size));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device vector (error code " << cudaGetErrorString(err) << ")!\n";
        return 1;
    }
    err = cudaMemcpy(xq_gpu, xq, xq_gpu_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy vector from host to device (error code " << cudaGetErrorString(err) << ")!\n";
        return 1;
    }

    // Allocate I and D on GPU
    long* I_gpu;
    int   I_gpu_size = k*nq*sizeof(long);
    err = cudaMalloc((void **)&I_gpu, (I_gpu_size));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device vector (error code " << cudaGetErrorString(err) << ")!\n";
        return 1;
    }

    float* D_gpu;
    int D_gpu_size = k*nq*sizeof(float);
    err = cudaMalloc((void **)&D_gpu, (D_gpu_size));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device vector (error code " << cudaGetErrorString(err) << ")!\n";
        return 1;
    }

    for (size_t nb_idx = 0; nb_idx < nb_values.size(); nb_idx++) {
        int nb = nb_values[nb_idx];  // Get current nb configuration
        std::cout << "nb: " << nb << std::endl;

        float* xb = new float[d * nb];
        for (int i = 0; i < nb; i++) {
            for (int j = 0; j < d; j++)
                xb[d * i + j] = distrib(rng);
            xb[d * i] += i / 1000.;
        }

        // Move xb to GPU
        float *xb_gpu;
        int xb_gpu_size = nb*d*sizeof(float);
        err = cudaMalloc((void **)&xb_gpu, (xb_gpu_size));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate device vector (error code " << cudaGetErrorString(err) << ")!\n";
            return 1;
        }
        err = cudaMemcpy(xb_gpu, xb, xb_gpu_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy vector from host to device (error code " << cudaGetErrorString(err) << ")!\n";
            return 1;
        }

        faiss::gpu::StandardGpuResources res;
        faiss::gpu::GpuIndexFlatL2 index_flat(&res, d);
        std::this_thread::sleep_for(std::chrono::seconds(10));
        index_flat.add(nb, xb_gpu); // add vectors to the index
        std::this_thread::sleep_for(std::chrono::seconds(10));

        for (int iter = 0; iter < num_warmups + num_searches; iter++) {
            auto start = std::chrono::high_resolution_clock::now();
            index_flat.search(nq, xq_gpu, k, D_gpu, I_gpu);
            auto end = std::chrono::high_resolution_clock::now();

            // Calculate duration in nanoseconds and convert to microseconds
            auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            uint32_t duration_us = static_cast<uint32_t>(duration_ns / 1000);  // Convert ns to µs

            // Store the duration in the 2D latencies array
            if (iter >= num_warmups) {
                latencies[nb_idx][iter - num_warmups] = duration_us;
            }

            if (iter >= num_warmups) {
                int total_memory = 0;
                int used_memory = 0;
                int free_memory = 0;
                printGPUMemoryUsage(total_memory, used_memory, free_memory);
                memories[nb_idx][iter - num_warmups] = used_memory;
            }
        }            
        cudaFree(xb_gpu);
        delete[] xb;
    }

    std::string file_name = "4-GPU_FlatL2_vary_nb_dim_" + std::to_string(d) + "_nq_" + std::to_string(nq) + "_k_" + std::to_string(k) + "_iter_" + std::to_string(num_searches) + "_latencies.csv";
    std::filesystem::path csv_file_path = std::filesystem::path(data_path) / file_name;
    std::cout << "csv_file_path: " << csv_file_path << std::endl;
    std::ofstream csv_file(csv_file_path);
    write_csv(csv_file, nb_values, &latencies[0][0], num_searches, true);

    // CSV for tracking GPU memory usage
    std::string mem_file_name = "mem_4-GPU_FlatL2_vary_nb_dim_" + std::to_string(d) + "_nq_" + std::to_string(nq) + "_k_" + std::to_string(k) + "_iter_" + std::to_string(num_searches) + "_latencies.csv";
    std::filesystem::path mem_csv_file_path = std::filesystem::path(data_path) / mem_file_name;
    std::ofstream mem_csv_file(mem_csv_file_path);
    write_csv(mem_csv_file, nb_values, &memories[0][0], num_searches, false);

    cudaFree(xq_gpu);
    cudaFree(I_gpu);
    cudaFree(D_gpu);
    delete[] xq;

    return 0;
}