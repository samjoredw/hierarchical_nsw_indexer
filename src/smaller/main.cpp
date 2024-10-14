#include "../resources/hnswlib.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cassert>
#include <chrono>


const int DIMENSIONS = 100;
const int MAX_ELEMENTS = 100000;
const int M = 30;
const int EF_CONSTRUCTION = 50;

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

    std::cout << "Environment Parameters:" << std::endl;
    std::cout << "Dimensions: " << DIMENSIONS << std::endl;
    std::cout << "Number of Vectors: " << MAX_ELEMENTS << std::endl;
    std::cout << "Number of Neighbors (M): " << M << std::endl;
    std::cout << "EF Construction: " << EF_CONSTRUCTION << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < MAX_ELEMENTS; ++i) {
        hnsw_alg.addPoint(vectors.data() + i * DIMENSIONS, i);
    }

    hnsw_alg.saveIndex("../../../HNSW_SPACEV1B/index.bin");

    file.close();

    std::cout << "Successfully added " << MAX_ELEMENTS << " vectors to the HNSW index." << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Time taken to add points: " << elapsed_time.count() << " seconds." << std::endl;

    return 0;
}