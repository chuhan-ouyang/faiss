source ../../env.sh
g++ -o build/1-Flat-test 1-Flat-test.cpp -lfaiss
g++ -o build/1-Flat-same 1-Flat-test-same.cpp -lfaiss
g++ -o build/1-Flat-repeated 1-Flat-repeated.cpp -lfaiss
g++ -o build/1-Flat-test-same 1-Flat-repeated-same.cpp -lfaiss
g++ -o build/1-Flat-test-nb 1-Flat-test-nb.cpp -lfaiss