#include "../resources/hnswlib.h"
#include <iostream>
#include <vector>
#include <chrono>


int main() {
    // Load the existing HNSW index from disk
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
        query_file.close();
        delete alg_hnsw;
        return -1;
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
        std::cout << "Label: " << label << ", Distance: " << dist << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Time taken for query: " << elapsed_time.count() << " seconds." << std::endl;

    // Clean up memory
    delete alg_hnsw;
    return 0;
}