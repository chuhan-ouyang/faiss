cmake_minimum_required(VERSION 3.10)
project(MyFaissProject)

# Find Faiss package
find_package(Faiss REQUIRED)

# Find CUDA Toolkit if using Faiss with GPU support
find_package(CUDAToolkit REQUIRED)

# Add an executable for your project
add_executable(my_faiss_app 4-GPU.cpp)

# Include directories
target_include_directories(my_faiss_app PRIVATE ${FAISS_INCLUDE_DIRS})

# Link against Faiss and CUDA libraries
target_link_libraries(my_faiss_app PRIVATE faiss CUDA::cudart)