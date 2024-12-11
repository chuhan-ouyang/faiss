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
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <unistd.h>

#include <faiss/IndexFlat.h>

using idx_t = faiss::idx_t;

void readCSVtoFloatArray(const std::string& pathname, float* arr, int n, int d) {
    std::ifstream file(pathname);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + pathname);
    }

    std::string line;
    int row = 0;
    while (std::getline(file, line)) {
        if (row >= n) {
            throw std::runtime_error("CSV file has more rows than expected.");
        }

        std::istringstream lineStream(line);
        std::string cell;
        int col = 0;

        while (std::getline(lineStream, cell, ',')) {
            if (col >= d) {
                throw std::runtime_error("CSV file has more columns than expected.");
            }

            try {
                arr[row * d + col] = std::stof(cell);
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid data at row " + std::to_string(row) + " column " + std::to_string(col));
            }

            ++col;
        }

        if (col != d) {
            throw std::runtime_error("Row " + std::to_string(row) + " has fewer columns than expected.");
        }

        // print row content
        // std::cout << "First row content: " << std::endl;
        // if (row == 0) {
        //     for (int j = 0; j < col; j++) {
        //         std::cout << arr[j] << " ";
        //     }
        //     std::cout << std::endl;
        // }
        ++row;
    }
    if (row != n) {
        throw std::runtime_error("CSV file has fewer rows than expected.");
    }
    file.close();
}

int main(int argc, char** argv) {
    // read path/...csv and store in this matrix float* xb = new float[d * nb]
    // read path /users/co232/vortex/benchmark/perf_data/gist/query_emb.csv and store in this matrix float* xb = new float[d * nq]

    std::string base_data_path = "";
    std::string query_data_path = "";
    int d = 128;      // dimension
    int nb = 10000;   // database size
    int nq = 0;

    int opt;
    while ((opt = getopt(argc, argv, "p:e:d:b:q:")) != -1) {
        switch (opt) {
            case 'p':
                base_data_path = optarg;
                std::cout << "base_csv_data_path: " << base_data_path << std::endl;
                break;
            case 'e':
                query_data_path = optarg;
                std::cout << "query_csv_data_path: " << query_data_path << std::endl;
                break;
            case 'd':
                d = std::atoi(optarg);
                std::cout << "d: " << d << std::endl;
                break;
            case 'b':
                nb = std::atoi(optarg);
                std::cout << "nb: " << nb << std::endl;
                break;
            case 'q':
                nq = std::atoi(optarg);
                std::cout << "nq: " << nq << std::endl;
                break;
            default:
                break;
        }
    }
    if (base_data_path.empty() || query_data_path.empty()) {
        std::cerr << "./exec -p <base_data_path> -e <query_data_path> -d <dimension> -b <nb> -q <nq>" << std::endl;
    }

    float* xq = new float[d * nq];
    float* xb = new float[d * nb];
   
    try {
        readCSVtoFloatArray(query_data_path, xq, nq, d);
        std::cout << "xq loaded successfully, nq (rows): " << nb << ", d (cols): " << d << std::endl;
        for (int j = 0; j < d; ++j) {
            std::cout << xq[j] << " ";
        }
        std::cout << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        delete[] xq;
        return 1;
    }

    try {
        readCSVtoFloatArray(base_data_path, xb, nb, d);
        std::cout << "xb loaded successfull, nb (rows): " << nb << ", d (cols): " << d << std::endl;
        std::cout << "Firs row content: " << std::endl;
        // Optionally, print part of the data for verification
        for (int j = 0; j < d; ++j) {
            std::cout << xb[j] << " ";
        }
        std::cout << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        delete[] xb;
        return 1;
    }

    int nlist = 100;
    int k = 5;

    faiss::IndexFlatL2 index(d);  // call constructor
    index.add(nb, xb);            // add vectors to the index

    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        index.search(nq, xq, k, D, I);

        printf("I first 5 rows=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("D first 5 rows=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5f ", D[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    delete[] xb;
    delete[] xq;

    return 0;
}