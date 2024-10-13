#include "hnswlib.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

const int DIMENSIONS = 100;
const int MAX_ELEMENTS = 100000;
const int M = 10;
const int EF_CONSTRUCTION = 200;

int main() {
    hnswlib::L2Space l2space(DIMENSIONS);
    hnswlib::HierarchicalNSW<float> hnsw_alg(&l2space, MAX_ELEMENTS, M, EF_CONSTRUCTION);

    std::ifstream file("../../../SPACEV1B/vectors.bin/vectors_1.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file." << std::endl;
        return -1;
    }

    std::vector<float> vectors(DIMENSIONS * MAX_ELEMENTS);

    file.read(reinterpret_cast<char*>(vectors.data()), DIMENSIONS * MAX_ELEMENTS * sizeof(float));

    if (file.gcount() < DIMENSIONS * MAX_ELEMENTS * sizeof(float)) {
        std::cerr << "Not enough data read from file." << std::endl;
        return -1;
    }

    for (size_t i = 0; i < MAX_ELEMENTS; ++i) {
        hnsw_alg.addPoint(vectors.data() + i * DIMENSIONS, i);
    }

    hnsw_alg.saveIndex("../../../HNSW_SPACEV1B/index.bin");

    file.close();

    std::cout << "Successfully added " << MAX_ELEMENTS << " vectors to the HNSW index." << std::endl;

    return 0;
}