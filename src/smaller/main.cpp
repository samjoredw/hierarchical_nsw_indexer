#include "../resources/hnswlib.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cassert>
#include <chrono>

const int DIMENSIONS = 100;           // Dimension of the vectors
const int MAX_ELEMENTS = 100000;      // Maximum number of elements in the index
const int M = 16;                     // Number of connections
const int EF_CONSTRUCTION = 100;       // Controls index search speed/build speed tradeoff

int main() {
    // Create an HNSW index with L2 distance
    hnswlib::L2Space l2space(DIMENSIONS);
    hnswlib::HierarchicalNSW<float> hnsw_alg(&l2space, MAX_ELEMENTS, M, EF_CONSTRUCTION);

    // Open the binary file containing int8_t vectors
    std::ifstream file("../../../SPACEV1B/vectors.bin/vectors_1.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file." << std::endl;
        return -1;
    }

    // Prepare to read int8_t vectors
    std::vector<int8_t> vectors(DIMENSIONS * MAX_ELEMENTS); // Use int8_t instead of float

    // Read vectors from the file
    file.read(reinterpret_cast<char*>(vectors.data()), DIMENSIONS * MAX_ELEMENTS * sizeof(int8_t));

    if (file.gcount() < DIMENSIONS * MAX_ELEMENTS * sizeof(int8_t)) {
        std::cerr << "Not enough data read from file." << std::endl;
        return -1;
    }

    std::cout << "Environment Parameters:" << std::endl;
    std::cout << "Dimensions: " << DIMENSIONS << std::endl;
    std::cout << "Number of Vectors: " << MAX_ELEMENTS << std::endl;
    std::cout << "Number of Neighbors (M): " << M << std::endl;
    std::cout << "EF Construction: " << EF_CONSTRUCTION << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    // Add each vector to the HNSW index
    for (size_t i = 0; i < MAX_ELEMENTS; ++i) {
        // Cast the int8_t pointer to float pointer for the addPoint function
        hnsw_alg.addPoint(reinterpret_cast<float*>(vectors.data() + i * DIMENSIONS), i);
    }

    // Save the index to disk
    hnsw_alg.saveIndex("../../../HNSW_SPACEV1B/index.bin");

    file.close();

    std::cout << "Successfully added " << MAX_ELEMENTS << " vectors to the HNSW index." << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Time taken to add points: " << elapsed_time.count() << " seconds." << std::endl;

    return 0;
}