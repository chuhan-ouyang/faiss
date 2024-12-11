#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

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
        if (row == 0) {
            for (int j = 0; j < col; j++) {
                std::cout << arr[row * d + j] << " ";
            }
            std::cout << std::endl;
        }

        ++row;
    }

    if (row != n) {
        throw std::runtime_error("CSV file has fewer rows than expected.");
    }

    file.close();
}

int main() {
    const std::string path = "/users/co232/vortex/benchmark/perf_data/gist/base_emb.csv";
    int n = 1000000; // Number of rows
    int d = 960; // Number of columns

    float* xb = new float[d * n];

    try {
        readCSVtoFloatArray(path, xb, n, d);
        std::cout << "Data loaded successfull, n (rows): " << n << ", d (cols): " << d << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        delete[] xb;
        return 1;
    }

    std::cout << "Firs row content: " << std::endl;
    // Optionally, print part of the data for verification
    for (int i = 0; i <= 0; ++i) {
        for (int j = 0; j < d; ++j) {
            std::cout << xb[i * d + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] xb;
    return 0;
}