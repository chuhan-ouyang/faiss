source ../../env.sh
g++ -o build/gist_test_indexflatl2 gist_test_indexflatl2.cpp -g -lfaiss
g++ -o build/csv_read csv_read.cpp -lfaiss