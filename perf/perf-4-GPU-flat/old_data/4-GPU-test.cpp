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
#include <cstdint>
#include <fstream>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include "gpu_memory.hpp"

#include <cuda_runtime.h>

int main(int argc, char* argv[]) {
    int d = 128;      // dimension
    int nb = 10000;  // database size
    std::vector<int> nq_values;
    for (int nq = 200; nq <= 5000; nq += 200) {
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

    for (size_t nq_idx = 0; nq_idx < nq_values.size(); nq_idx++) {
        int nq = nq_values[nq_idx];  // Get current nq configuration
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

            cudaFree(xq_gpu);
            cudaFree(I_gpu);
            cudaFree(D_gpu);
            delete[] xq;
        }
    }

    std::ofstream csv_file("4-GPU-flat_200_5000_dim_128_batch_10000_k_4_latencies.csv");
    if (csv_file.is_open()) {
        for (size_t nq_idx = 0; nq_idx < nq_values.size(); nq_idx++) {
            csv_file << nq_values[nq_idx];  // Write the nq value
            uint32_t sum = 0;
            for (int iter = 0; iter < num_searches; iter++) {
                csv_file << ", " << latencies[nq_idx][iter];
                sum += latencies[nq_idx][iter];
            }
            double avg_latency = sum / static_cast<double>(num_searches);
            csv_file << ", " << avg_latency << "\n";
        }
        csv_file.close();
    } else {
        std::cerr << "Unable to open file for writing\n";
    }

    cudaFree(xb_gpu);
    delete[] xb;
    return 0;
}