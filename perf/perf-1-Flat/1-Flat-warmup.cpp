/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <faiss/IndexFlat.h>

// 64-bit int
using idx_t = faiss::idx_t;

int main() {
    int d = 960;      // dimension
    int nb = 100000;  // database size

    std::vector<int> nq_values;
    for (int nq = 50; nq <= 50; nq += (nq == 1 ? 4 : 5)) {
        nq_values.push_back(nq);
    }

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    // Allocate memory for xb
    float* xb = new float[d * nb];

    // Populate xb
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    // Initialize the index
    faiss::IndexFlatL2 index(d);  // call constructor
    index.add(nb, xb);            // add vectors to the index

    int k = 4;  // Top-k results
    int num_searches = 100;  // Number of iterations per nq configuration

    // Latency storage: 2D array where rows are nq configurations and columns are iterations
    uint32_t latencies[nq_values.size()][num_searches];

    // Iterate over different nq configurations
    for (size_t nq_idx = 0; nq_idx < nq_values.size(); nq_idx++) {
        int nq = nq_values[nq_idx];  // Get current nq configuration

        // Allocate memory for xq based on current nq
        float* xq = new float[d * nq];

        // Populate xq
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < d; j++)
                xq[d * i + j] = distrib(rng);
            xq[d * i] += i / 1000.;
        }

        // Perform search for each iteration
        for (int iter = 0; iter < num_searches; iter++) {
            idx_t* I = new idx_t[k * nq];
            float* D = new float[k * nq];

            auto start = std::chrono::high_resolution_clock::now();
            index.search(nq, xq, k, D, I);
            auto end = std::chrono::high_resolution_clock::now();

            // Calculate duration in nanoseconds and convert to microseconds
            auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            uint32_t duration_us = static_cast<uint32_t>(duration_ns / 1000);  // Convert ns to µs

            // Store the duration in the 2D latencies array
            latencies[nq_idx][iter] = duration_us;

            delete[] I;
            delete[] D;
        }
        delete[] xq;  // Free memory for xq after each nq configuration
    }

    std::ofstream csv_file("1-Flat_warmup_dim_960_batch_100000_k_4_latencies.csv");
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

    delete[] xb;

    return 0;
}