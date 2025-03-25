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


// Execute the queries, compute and report the recall
int main(int argc, char *argv[]) {
	// Get number of threads
    unsigned int nthreads = std::thread::hardware_concurrency();
	std::cout << "Number of threads: " << nthreads << std::endl;
    
	// Parameters
    std::string path_database_attributes; 		
	std::string path_query_vectors;
	std::string path_query_attributes;
	std::string path_groundtruth;
	std::string path_index;
	std::string filter_type;
    int k; 		
    int efs; 								//  default is 16

	// Check if the number of arguments is correct
	if (argc < 9) {
		fprintf(stderr, "Syntax: <path_database_attributes> <path_query_vectors> <path_query_attributes> <path_groundtruth> <path_index> <filter_type> <k> <efs>\n");
		exit(1);	
	}

	// Read command line arguments
	path_database_attributes = argv[1];
	path_query_vectors = argv[2];
	path_query_attributes = argv[3];
	path_groundtruth = argv[4];
	path_index = argv[5];
	filter_type = argv[6];
	k = atoi(argv[7]);
	efs = atoi(argv[8]);

	// Read database attributes
	size_t n_items, n_attributes;
	std::vector<int> database_attributes;	// n_items x n_attributes
	database_attributes = read_int_attributes(path_database_attributes.c_str(), &n_items, &n_attributes);

	// Read query vectors
    size_t n_queries, d;
    float* query_vectors;					// n_queries x d
	query_vectors = fvecs_read(path_query_vectors.c_str(), &d, &n_queries);

	// Read query attributes
	size_t n_queries_2, n_attributes_2;
	std::vector<int> query_attributes;		// n_queries x n_attributes
	if (filter_type == "EM" or filter_type == "EMIS"){
		query_attributes = read_int_attributes(path_query_attributes.c_str(), &n_queries_2, &n_attributes_2);
	}
	else if (filter_type == "R"){
		query_attributes = read_int_pair_attributes(path_query_attributes.c_str(), &n_queries_2, &n_attributes_2);
	}
	assert(n_queries == n_queries_2 && "Number of queries in query vectors and query attributes do not match");
	assert(n_attributes == n_attributes_2 && "Number of attributes in database and query attributes do not match");

	// Read ground-truth and truncate to at most k
	size_t n_queries_3;
	std::vector<std::vector<int>> groundtruth;
	groundtruth = read_ivecs(path_groundtruth.c_str());
	n_queries_3 = groundtruth.size();
	assert(n_queries == n_queries_3 && "Number of queries in query vectors and groundtruth do not match");
	for (std::size_t i = 0; i < groundtruth.size(); ++i) {
        if (groundtruth[i].size() > static_cast<std::size_t>(k)) {
            groundtruth[i].resize(k);
        }
    }
	

	// Load index from file
	auto& acorn_index = *dynamic_cast<faiss::IndexACORNFlat*>(faiss::read_index(path_index.c_str()));

	// Set search parameter
	acorn_index.acorn.efSearch = efs;

	double t0 = elapsed();
	// Compute bitmap for filtering
	std::vector<char> filter_bitmap(n_queries * n_items);
	// EM: We support multiple attributes for EM-filtering
	if (filter_type == "EM"){
		for (size_t q = 0; q < n_queries; q++) {
			for (size_t i = 0; i < n_items; i++) {
				bool match = true;
				for (size_t a = 0; a < n_attributes; a++) {
					if ( database_attributes[i * n_attributes + a] != query_attributes[q * n_attributes + a]) {
						match = false;
						break;
					}
				}
				filter_bitmap[q * n_items + i] = match;
			}
		}
	}
	// R: We support multiple attributes for range-filtering
	else if (filter_type == "R"){
		for (size_t q = 0; q < n_queries; q++) {
			for (size_t i = 0; i < n_items; i++) {
				bool match = true;
				for (size_t a = 0; a < n_attributes; a++) {
					if (( database_attributes[i * n_attributes + a] >= query_attributes[q * n_attributes * 2 + a * 2]) &&
					   	( database_attributes[i * n_attributes + a] <= query_attributes[q * n_attributes * 2 + a * 2 + 1])) { 
						match = false;
						break;
					}
				}
				filter_bitmap[q * n_items + i] = match;
			}
		}
	}
	// EMIS: TODO
	else if (filter_type == "EMIS"){
		fprintf(stderr, "EMIS filtering not implemented yet\n");
	} else {
		fprintf(stderr, "Unknown filter type: %s\n", filter_type.c_str());
	}

	// Execute queries on ACORN index
	// TODO: How does ACORN behave if there are less than k matching items?
    std::vector<faiss::idx_t> nearest_neighbors(k * n_queries);
	std::vector<float> distances(k * n_queries);
	acorn_index.search(n_queries, query_vectors, k, distances.data(), nearest_neighbors.data(), filter_bitmap.data());
	double query_execution_time = elapsed() - t0;		
    peak_memory_footprint();

	// Compute recall 
	// TODO: Check what happens (what ACORN returns) if there are less than k matching items
	size_t match_count = 0;
	size_t total_count = 0;
	faiss::idx_t* nearest_neighbors_ptr = nearest_neighbors.data();
	for (size_t q = 0; q < n_queries; q++) {
		int n_valid_neighbors = std::min(k, (int)groundtruth[q].size());
		std::vector<int> groundtruth_q = groundtruth[q];
		std::vector<int> nearest_neighbors_q(nearest_neighbors_ptr + q * k, nearest_neighbors_ptr + (q+1) * k);
		std::sort(groundtruth_q.begin(), groundtruth_q.end());
		std::sort(nearest_neighbors_q.begin(), nearest_neighbors_q.end());
		std::vector<int> intersection;
		std::set_intersection(groundtruth_q.begin(), groundtruth_q.begin() + n_valid_neighbors, nearest_neighbors_q.begin(), nearest_neighbors_q.begin() + k, std::back_inserter(intersection));
		match_count += intersection.size();
		total_count += n_valid_neighbors;
	}
	double recall = (double)match_count / total_count;
	double qps = (double)n_queries / query_execution_time;
	printf("Queries per second: %.3f\n", qps);
	printf("Recall: %.3f\n", recall);
}
