g++ 4-GPU-test.cpp -L/usr/local/lib/ -lfaiss  -lcublas_static -lnvidia-ml  -lculibos -lcudart_static -lpthread -ldl -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -o myCublasApp
