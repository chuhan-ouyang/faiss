source ../../env.sh
g++ -o build/gist_test_ivfflat gist_test_ivfflat.cpp -g -lfaiss
g++ -o build/csv_read csv_read.cpp -lfaiss