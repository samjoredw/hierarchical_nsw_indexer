#include "../resources/hnswlib.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <queue>
#include <cmath>

int main() {
    const std::string index_path = "../../../HNSW_SPACEV1B/index.bin";
    const int DIMENSIONS = 100;
    
    hnswlib::L2Space space(DIMENSIONS);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_path);

    const std::string query_path = "../../../SPTAG/SPTAG/datasets/SPACEV1B/query.bin";
    std::ifstream query_file(query_path, std::ios::binary);
    if (!query_file.is_open()) {
        std::cerr << "Could not open query file." << std::endl;
        delete alg_hnsw;
        return -1;
    }

    float query_vector[DIMENSIONS];
    query_file.read(reinterpret_cast<char*>(query_vector), DIMENSIONS * sizeof(float));
    if (query_file.gcount() < DIMENSIONS * sizeof(float)) {
        std::cerr << "Not enough data read from query file." << std::endl;
        query_file.close(); // Close the file
        delete alg_hnsw; // Clean up memory before exiting
        return -1;
    }

    // Check the validity of the query vector before using it
    for (int i = 0; i < DIMENSIONS; ++i) {
        if (std::isnan(query_vector[i])) {
            std::cerr << "Query vector contains NaN values." << std::endl;
            query_file.close();
            delete alg_hnsw;
            return -1;
        }
    }

    // Number of neighbors to search
    const int k = 3;

    // Measure query time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform the search for the nearest neighbors using the constant query vector
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(query_vector, k);

    // Output the results
    std::cout << "Query Results:" << std::endl;
    while (!result.empty()) {
        auto [dist, label] = result.top();
        result.pop();

        // Check for NaN distance
        if (std::isnan(dist)) {
            std::cerr << "Distance calculation resulted in NaN for label: " << label << std::endl;
            continue; // Skip this result
        }

        std::cout << "Label: " << label << ", Distance: " << dist << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time; // Measure in milliseconds
    std::cout << "Time taken for query: " << elapsed_time.count() << " milliseconds." << std::endl;

    // Clean up
    query_file.close(); // Close the query file
    delete alg_hnsw; // Clean up the HNSW object
    return 0;
}