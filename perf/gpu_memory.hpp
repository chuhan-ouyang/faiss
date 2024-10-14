// gpu_memory.cpp
#pragma once
#include <iostream>
#include <nvml.h>

void printGPUMemoryUsage(int& total_memory, int& used_memory, int& free_memory) {
    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return;
    }

    // Get the first GPU device handle
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device); // 0 is the index of the first GPU
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get GPU handle: " << nvmlErrorString(result) << std::endl;
        nvmlShutdown();
        return;
    }

    // Get memory information
    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get memory information: " << nvmlErrorString(result) << std::endl;
        nvmlShutdown();
        return;
    }

    // Print memory usage
    // std::cout << "Total memory: " << memory.total / (1024 * 1024) << " MB" << std::endl;
    // std::cout << "Used memory: " << memory.used / (1024 * 1024) << " MB" << std::endl;
    // std::cout << "Free memory: " << memory.free / (1024 * 1024) << " MB" << std::endl;

    total_memory = memory.total / (1024 * 1024);
    used_memory = memory.used / (1024 * 1024);
    free_memory = memory.free / (1024 * 1024);

    // Shutdown NVML
    nvmlShutdown();
}