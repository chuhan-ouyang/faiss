g++ 4-GPU-ivf-test.cpp -L/usr/local/lib/ -lfaiss  -lcublas_static -lnvidia-ml  -lculibos -lcudart_static -lpthread -ldl -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -O3 -o 4-GPU-ivf-test
g++ 4-GPU-ivf-repeated.cpp -L/usr/local/lib/ -lfaiss  -lcublas_static -lnvidia-ml  -lculibos -lcudart_static -lpthread -ldl -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -O3 -o 4-GPU-rep-test
