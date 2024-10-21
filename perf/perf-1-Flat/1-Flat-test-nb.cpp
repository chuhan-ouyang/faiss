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
    int d = 128;      // dimension
    int nq = 50;    //  batch size

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

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    int k = 4;  // Top-k results
    int num_searches = 50;  // Number of iterations per nq configuration
    int num_warmups = 50;  // Number of iterations per nq configuration

    // Latency storage: 2D array where rows are nb configurations and columns are iterations
    uint32_t latencies[nb_values.size()][num_searches];

    // Iterate over different nq configurations
    for (size_t nb_idx = 0; nb_idx < nb_values.size(); nb_idx++) {
        int nb = nb_values[nb_idx];  // Get current nq configuration
        std::cout << "nb: " << nb << std::endl;

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

            auto start = std::chrono::high_resolution_clock::now();
            index.search(nq, xq, k, D, I);
            auto end = std::chrono::high_resolution_clock::now();

            // Calculate duration in nanoseconds and convert to microseconds
            auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            uint32_t duration_us = static_cast<uint32_t>(duration_ns / 1000);  // Convert ns to µs

            // Store the duration in the 2D latencies array
            if (iter >= num_warmups) {
                latencies[nb_idx][iter - num_warmups] = duration_us;
            }
            delete[] xb;
        }
        delete[] I;
        delete[] D;
        delete[] xq;  // Free memory for xq after each nq configuration
    }

    std::string file_name = "1-FlatL2-CPU-vary-nb_dim_" + std::to_string(d) + "_nq_" + std::to_string(nq) + "_k_" + std::to_string(k) + "_iter_" + std::to_string(num_searches) + "_latencies.csv";
    std::filesystem::path csv_file_path = std::filesystem::path(data_path) / file_name;
    std::cout << "csv_file_path: " << csv_file_path << std::endl;
    std::ofstream csv_file(csv_file_path);
    if (csv_file.is_open()) {
        for (size_t nb_idx = 0; nb_idx < nb_values.size(); nb_idx++) {
            csv_file << nb_values[nb_idx];  // Write the nb value
            uint32_t sum = 0;
            for (int iter = 0; iter < num_searches; iter++) {
                csv_file << ", " << latencies[nb_idx][iter];
                sum += latencies[nb_idx][iter];
            }
            double avg_latency = sum / static_cast<double>(num_searches);
            csv_file << ", " << avg_latency << "\n";
        }
        csv_file.close();
    } else {
        std::cerr << "Unable to open file for writing\n";
    }

    return 0;
}