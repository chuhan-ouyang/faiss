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
#include <unistd.h>
#include <filesystem>

#include <faiss/IndexFlat.h>

// 64-bit int
using idx_t = faiss::idx_t;

int main(int argc, char** argv) {
    std::string data_path = "";
    int d = 960;      // dimension
    int nb = 10000;  //  database size

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

    int nq = 1;
    std::vector<int> nq_values;
    nq_values.push_back(nq);

    std::mt19937 rng(1234);
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
    int num_searches = 50;  // Number of iterations per nq configuration
    int num_warmups = 50;  // Number of iterations per nq configuration

    // Latency storage: 2D array where rows are nq configurations and columns are iterations
    uint32_t latencies[nq_values.size()][num_searches];

    // Iterate over different nq configurations
    for (size_t nq_idx = 0; nq_idx < nq_values.size(); nq_idx++) {
        int nq = nq_values[nq_idx];  // Get current nq configuration
        std::cout << "nq: " << nq << std::endl;

        // Allocate memory for xq based on current nq
        float* xq = new float[d * nq];

        // Populate xq
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < d; j++)
                xq[d * i + j] = distrib(rng);
            xq[d * i] += i / 1000.;
        }

        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        // Perform search for each iteration
        for (int iter = 0; iter < num_warmups + num_searches; iter++) {
            auto start = std::chrono::high_resolution_clock::now();
            index.search(nq, xq, k, D, I);
            auto end = std::chrono::high_resolution_clock::now();

            // Calculate duration in nanoseconds and convert to microseconds
            auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            uint32_t duration_us = static_cast<uint32_t>(duration_ns / 1000);  // Convert ns to µs

            // Store the duration in the 2D latencies array
            if (iter >= num_warmups) {
                latencies[nq_idx][iter - num_warmups] = duration_us;
            }
        }
        delete[] I;
        delete[] D;
        delete[] xq;  // Free memory for xq after each nq configuration
    }

    std::string file_name = "1-FlatL2-CPU-repeated-same-data_dim_" + std::to_string(d) + "_nb_" + std::to_string(nb) + "_k_" + std::to_string(k) + "_nq_" + std::to_string(nq) + "_iter_" + std::to_string(num_searches) + "_latencies.csv";
    std::filesystem::path csv_file_path = std::filesystem::path(data_path) / file_name;
    std::cout << "csv_file_path: " << csv_file_path << std::endl;
    std::ofstream csv_file(csv_file_path);
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