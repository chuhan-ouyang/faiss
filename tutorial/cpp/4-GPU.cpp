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

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include "gpu_memory.hpp"

#include <cuda_runtime.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <number of runs>\n";
        return 1;
    }
    int num_runs = std::atoi(argv[1]);

    int d = 1024;      // dimension
    int nb = 100000; // database size
    int nq = 10;  // nb of queries
    int k = 4;
    

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    // Move xp to GPU
    // Device vector
    float *d_Data;
    int size = nq*d*sizeof(float);
    cudaError_t err = cudaMalloc((void **)&d_Data, (size));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device vector (error code " << cudaGetErrorString(err) << ")!\n";
        return 1;
    }

    // Copy host vector to device vector
    err = cudaMemcpy(d_Data, xq, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy vector from host to device (error code " << cudaGetErrorString(err) << ")!\n";
        return 1;
    }



    faiss::gpu::StandardGpuResources res;

    // 1. Using a flat index
    std::cout << "Using a flat index GPU" << std::endl;
    faiss::gpu::GpuIndexFlatL2 index_flat(&res, d);

    printf("is_trained = %s\n", index_flat.is_trained ? "true" : "false");
    std::cout << "declared index_flat" << std::endl;
    printGPUMemoryUsage();
    std::this_thread::sleep_for(std::chrono::seconds(10));

    std::cout << "Adding vectors to the index" << std::endl;
    index_flat.add(nb, xb); // add vectors to the index
    printf("ntotal = %ld\n", index_flat.ntotal);
    std::cout << "added vectors to the index" << std::endl;
    printGPUMemoryUsage();
    std::this_thread::sleep_for(std::chrono::seconds(10));


    std::cout << "Searching vectors in the index" << std::endl;

    for (int i = 0; i < num_runs; i++){ // search xq
        long* I = new long[k * nq];
        float* D = new float[k * nq];

        // index_flat.search(nq, xq, k, D, I);
        index_flat.search(nq, d_Data, k, D, I);
        if(i == num_runs -1){
            std::cout << "last run" << std::endl;
            printGPUMemoryUsage();
        }
            

        delete[] I;
        delete[] D;
    }
    cudaFree(d_Data);
    // cudaDeviceReset();


    // // 2. Using an IVF index
    // std::cout << "Using an IVF index GPU" << std::endl;
    // int nlist = 100;
    // faiss::gpu::GpuIndexIVFFlat index_ivf(&res, d, nlist, faiss::METRIC_L2);
    
    // std::cout << "before training" << std::endl;
    // assert(!index_ivf.is_trained);
    // index_ivf.train(nb, xb);
    // assert(index_ivf.is_trained);
    // std::cout << "finish training" << std::endl;
    // printGPUMemoryUsage();
    // std::this_thread::sleep_for(std::chrono::seconds(10));

    // std::cout << "Before adding vectors to the index" << std::endl;
    // index_ivf.add(nb, xb); // add vectors to the index
    // std::cout << "After adding vectors to the index" << std::endl;
    // printGPUMemoryUsage();
    // std::this_thread::sleep_for(std::chrono::seconds(10));

    // printf("is_trained = %s\n", index_ivf.is_trained ? "true" : "false");
    // printf("ntotal = %ld\n", index_ivf.ntotal);

    // for (int i = 0; i < num_runs; i++){// search xq
    //     long* I = new long[k * nq];
    //     float* D = new float[k * nq];

    //     index_ivf.search(nq, xq, k, D, I);
    //     if (i == num_runs - 1){
    //         std::cout << "last run" << std::endl;
    //         printGPUMemoryUsage();
    //     }

    //     delete[] I;
    //     delete[] D;
    // }

    delete[] xb;
    delete[] xq;

    return 0;
}