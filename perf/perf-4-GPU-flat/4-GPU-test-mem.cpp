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

void write_csv(std::ofstream& csv_file, std::vector<int>& nq_values, uint32_t* latencies, int num_searches, bool write_all) {
    if (csv_file.is_open()) {
        for (size_t nq_idx = 0; nq_idx < nq_values.size(); nq_idx++) {
            csv_file << nq_values[nq_idx];  // Write the nq value
            uint32_t sum = 0;
            for (int iter = 0; iter < num_searches; iter++) {
                uint32_t latency = *(latencies + nq_idx * num_searches + iter);
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
    int nb = 0;  // database size

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
                nb = std::atoi(optarg);
                break;
            default:
                break;
        }
    }
    if (data_path.empty()) {
        std::cerr << "./exec -p <data_path> -d <dimension> -n <nb>" << std::endl;
    }

    std::vector<int> nq_values;
    for (int nq = 200; nq <= 5000; nq += 500) {
        nq_values.push_back(nq);
    }
    int k = 4;

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    // Move xb to GPU
    float *xb_gpu;
    int xb_gpu_size = nb*d*sizeof(float);
    cudaError_t err = cudaMalloc((void **)&xb_gpu, (xb_gpu_size));
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

    int num_searches = 50;  // Number of search iterations per nq configuration
    int num_warmups = 50;  // Number of warmup iterations per nq configuration

    // Latency storage: 2D array where rows are nq configurations and columns are iterations
    uint32_t latencies[nq_values.size()][num_searches];
    uint32_t memories[nq_values.size()][num_searches];

    for (size_t nq_idx = 0; nq_idx < nq_values.size(); nq_idx++) {
        int nq = nq_values[nq_idx];  // Get current nq configuration
        std::cout << "nq: " << nq << std::endl;
        for (int iter = 0; iter < num_warmups + num_searches; iter++) {
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
            err = cudaMalloc((void **)&xq_gpu, (xq_gpu_size));
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
            int I_gpu_size = k*nq*sizeof(long);
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

            auto start = std::chrono::high_resolution_clock::now();
            index_flat.search(nq, xq_gpu, k, D_gpu, I_gpu);
            auto end = std::chrono::high_resolution_clock::now();

            // Calculate duration in nanoseconds and convert to microseconds
            auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            uint32_t duration_us = static_cast<uint32_t>(duration_ns / 1000);  // Convert ns to µs

            // Store the duration in the 2D latencies array
            if (iter >= num_warmups) {
                latencies[nq_idx][iter - num_warmups] = duration_us;
            }

            if (iter >= num_warmups) {
                int total_memory = 0;
                int used_memory = 0;
                int free_memory = 0;
                printGPUMemoryUsage(total_memory, used_memory, free_memory);
                memories[nq_idx][iter - num_warmups] = used_memory;
            }

            cudaFree(xq_gpu);
            cudaFree(I_gpu);
            cudaFree(D_gpu);
            delete[] xq;
        }
    }

    std::string file_name = "4-GPU_FlatL2_dim_" + std::to_string(d) + "_nb_" + std::to_string(nb) + "_k_" + std::to_string(k) + "_iter_" + std::to_string(num_searches) + "_latencies.csv";
    std::filesystem::path csv_file_path = std::filesystem::path(data_path) / file_name;
    std::cout << "csv_file_path: " << csv_file_path << std::endl;
    std::ofstream csv_file(csv_file_path);
    write_csv(csv_file, nq_values, &latencies[0][0], num_searches, true);

    // CSV for tracking GPU memory usage
    std::string mem_file_name = "mem_4-GPU_FlatL2_dim_" + std::to_string(d) + "_nb_" + std::to_string(nb) + "_k_" + std::to_string(k) + "_iter_" + std::to_string(num_searches) + "_latencies.csv";
    std::filesystem::path mem_csv_file_path = std::filesystem::path(data_path) / mem_file_name;
    std::ofstream mem_csv_file(mem_csv_file_path);
    write_csv(mem_csv_file, nq_values, &memories[0][0], num_searches, false);

    cudaFree(xb_gpu);
    delete[] xb;
    return 0;
}