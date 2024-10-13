#include "../resources/hnswlib.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cassert>

const int DIMENSIONS = 100;
const int MAX_ELEMS = 1402020720;
const int M = 16;
const int EF_CONSTRUCTION = 200;
const int QUERIES = 29316;
const size_t MAX_BATCH = 5000;

int main () {

      hnswlib::L2SpaceI l2space(DIMENSIONS); 
      hnswlib::HierarchicalNSW<int> hnsw_alg(&l2space, MAX_ELEMS, M, EF_CONSTRUCTION);

      const std::string vector_dir = "~/users/samjo/nvme1n1/SPTAG/SPTAG/datasets/SPACEV1B/vectors.bin/";

      // This is going to be the outer loop for each binary file in the directory
      for (const auto & binary_file : std::filesystem::directory_iterator(vector_dir)) {
            if (binary_file.path().extension() != ".bin") {
                  continue;
            }

            std::ifstream file(binary_file.path(), std::ios::binary);
            if (!file.is_open()) {
                  std::cerr << "Could not open file: " << binary_file.path() << std::endl;
                  continue;
            }

            // Okay now at this point, we have to decide how we are going to process the vectors.
            // Trying to store all vectors within a binary file in memory is probably not a good idea.
            // So we can either use streaming or batch processing here. I think batch processing is the
            // correct move because we can directly optimize diskIO.

            // Lets start with 5,000 vectors per batch
            int8_t* vectors = (int8_t*)malloc(MAX_BATCH * DIMENSIONS * sizeof(int8_t));
                  if (vectors == nullptr) {
                  std::cerr << "ALLOC WAY TOO BIG\n" << std::endl;
                  return 1;
            }

            size_t total_vectors_read = 0; 

            // Then we read and process the vectors in their batches
            while (true) {
                  file.read((char*)vectors, MAX_BATCH * DIMENSIONS * sizeof(int8_t));
                  std::streamsize bytes_read = file.gcount();

                  // If we have read 0 bytes, then we have reached the end of the file
                  if (bytes_read == 0) {
                        break;
                  }
                  // If we have read less than a full vector, then we have reached the end of the file
                  size_t num_vectors = bytes_read / DIMENSIONS / sizeof(int8_t);
                  if (num_vectors == 0) {
                        break;
                  }

                  // And this is the inner most loop where we add the vectors to the index
                  for (size_t i = 0; i < num_vectors; ++i) {
                        hnsw_alg.addPoint(vectors + i * DIMENSIONS, total_vectors_read + i);
                  }

                  total_vectors_read += num_vectors; 
            }

            hnsw_alg.saveIndex("HNSW_SPACEV1B/index.bin"); 

            free(vectors);
            file.close();
      }

      return 0;
}