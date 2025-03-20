#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>

#include <sys/time.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// added these
#include <faiss/Index.h>
#include <stdlib.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <pthread.h>
#include <iostream>
#include <sstream>      // for ostringstream
#include <fstream>  
#include <iosfwd>
#include <faiss/impl/platform_macros.h>
#include <assert.h>     /* assert */
#include <thread>
#include <set>
#include <math.h>  
#include <numeric> // for std::accumulate
#include <cmath>   // for std::mean and std::stdev
#include <nlohmann/json.hpp>
#include "utils.cpp"

void peak_memory_footprint()
{
    unsigned iPid = (unsigned)getpid();
    std::string status_file = "/proc/" + std::to_string(iPid) + "/status";
    std::ifstream info(status_file);
    if (!info.is_open())
    {
        std::cout << "memory information open error!" << std::endl;
    }
    std::string tmp;
    while (getline(info, tmp))
    {
        if (tmp.find("Name:") != std::string::npos || tmp.find("VmPeak:") != std::string::npos || tmp.find("VmHWM:") != std::string::npos)
            std::cout << tmp << std::endl;
    }
    info.close();
}


// Create index, write it to file and collect statistics
int main(int argc, char *argv[]) {
	// Get number of threads
    unsigned int nthreads = std::thread::hardware_concurrency();
	std::cout << "Number of threads: " << nthreads << std::endl;

	// Parameters 
    std::string path_database_vectors;
	std::string path_index;
    size_t d; 	
    int M; 				
    int gamma;
    int M_beta; 		

	// Parse arguments
	if (argc != 7) {
		fprintf(stderr, "Syntax: <path_database_vectors> <path_index> <d> <M> <gamma> <M_beta>\n");
		exit(1);
	}

	// Store parameters
	path_database_vectors = argv[1];
	path_index = argv[2];
	d = atoi(argv[3]);
	M = atoi(argv[4]);
	gamma = atoi(argv[5]);
	M_beta = atoi(argv[6]);

	// Read vectors from file
	size_t num_vecs, dim;
	float* database_vectors = fvecs_read(path_database_vectors.c_str(), &dim, &num_vecs);
	assert(d == d2 && "dimensions of vectors in file do not match the dimensions passed as argument");

	// According to the following GitHub issue, the metadata does not influence the performance of ACORN:
	// https://github.com/TAG-Research/ACORN/issues/2
	// Therefore, we leave the metadata empty
	std::vector<int> metadata(num_vecs,0);

    // Initialize the ACORN index
    faiss::IndexACORNFlat acorn_index(d, M, gamma, metadata, M_beta);

	// Add vectors to index: This is the part that is timed for the index construction time
	double t0 = elapsed();
	acorn_index.add(num_vecs, database_vectors);

	// Print statistics
	printf("Index construction time: %.3f s\n", elapsed() - t0);
	peak_memory_footprint();
	delete[] database_vectors;

	// Write index to file
	write_index(&acorn_index, path_index.c_str());
}
