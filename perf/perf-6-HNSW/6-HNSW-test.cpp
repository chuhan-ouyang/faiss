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
#include <chrono>
#include <iostream>
#include <cstdint>
#include <fstream>
#include <vector>

#include <faiss/IndexHNSW.h>
#include <cuda_runtime.h>

using idx_t = faiss::idx_t;

int main() {
    int d = 64;      // dimension
    int nb = 100000; // database size
    int nq = 10000;  // nb of queries

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

    int k = 4;

    faiss::IndexHNSWFlat index(d, 32);
    // index.add(nb, xb_gpu);

    // { 
    //     // Allocate memory for xq based on current nq
    //     float* xq = new float[d * nq];
    //     for (int i = 0; i < nq; i++) {
    //         for (int j = 0; j < d; j++)
    //             xq[d * i + j] = distrib(rng);
    //         xq[d * i] += i / 1000.;
    //     }

    //     // Move xq to GPU
    //     float *xq_gpu;
    //     int xq_gpu_size = nq*d*sizeof(float);
    //     err = cudaMalloc((void **)&xq_gpu, (xq_gpu_size));
    //     if (err != cudaSuccess) {
    //         std::cerr << "Failed to allocate device vector (error code " << cudaGetErrorString(err) << ")!\n";
    //         return 1;
    //     }
    //     err = cudaMemcpy(xq_gpu, xq, xq_gpu_size, cudaMemcpyHostToDevice);
    //     if (err != cudaSuccess) {
    //         std::cerr << "Failed to copy vector from host to device (error code " << cudaGetErrorString(err) << ")!\n";
    //         return 1;
    //     }

    //     // idx_t* I = new idx_t[k * nq];
    //     // float* D = new float[k * nq];

    //     // Allocate I and D on GPU
    //     idx_t* I_gpu;
    //     int I_gpu_size = k*nq*sizeof(idx_t);
    //     err = cudaMalloc((void **)&I_gpu, (I_gpu_size));
    //     if (err != cudaSuccess) {
    //         std::cerr << "Failed to allocate device vector (error code " << cudaGetErrorString(err) << ")!\n";
    //         return 1;
    //     }

    //     float* D_gpu;
    //     int D_gpu_size = k*nq*sizeof(float);
    //     err = cudaMalloc((void **)&D_gpu, (D_gpu_size));
    //     if (err != cudaSuccess) {
    //         std::cerr << "Failed to allocate device vector (error code " << cudaGetErrorString(err) << ")!\n";
    //         return 1;
    //     }

    //     // search xq
    //     index.search(nq, xq_gpu, k, D_gpu, I_gpu);

    //     // printf("I=\n");
    //     // for (int i = nq - 5; i < nq; i++) {
    //     //     for (int j = 0; j < k; j++)
    //     //         printf("%5zd ", I_gpu[i * k + j]);
    //     //     printf("\n");
    //     // }

    //     // printf("D=\n");
    //     // for (int i = nq - 5; i < nq; i++) {
    //     //     for (int j = 0; j < k; j++)
    //     //         printf("%5f ", D_gpu[i * k + j]);
    //     //     printf("\n");
    //     // }

    //     cudaFree(xq_gpu);
    //     cudaFree(I_gpu);
    //     cudaFree(D_gpu);
    //     delete[] xq;
    // }

    cudaFree(xb_gpu);
    delete[] xb;

    return 0;
}
