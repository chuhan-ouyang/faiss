source ../../env.sh
g++ 4-GPU-test-mem.cpp -L/usr/local/lib/ -lfaiss  -lcublas_static -lnvidia-ml  -lculibos -lcudart_static -lpthread -ldl -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -o build/4-GPU-test-mem
g++ 4-GPU-test-mem-vary-nb.cpp -L/usr/local/lib/ -lfaiss  -lcublas_static -lnvidia-ml  -lculibos -lcudart_static -lpthread -ldl -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -o build/4-GPU-test-mem-vary-nb
g++ 4-GPU-repeated.cpp -L/usr/local/lib/ -lfaiss  -lcublas_static -lnvidia-ml  -lculibos -lcudart_static -lpthread -ldl -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -o build/4-GPU-rep
g++ 4-GPU-repeated-same.cpp -L/usr/local/lib/ -lfaiss  -lcublas_static -lnvidia-ml  -lculibos -lcudart_static -lpthread -ldl -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -o build/4-GPU-rep-same
