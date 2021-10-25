#include <stdio.h>
#include <stdlib.h>
// #include <stdint.h>
#include <float.h>
// #include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

using namespace std;


float CalculateDistance(ArffInstance *a, float *b) {
    float sum = 0;
    
    for(int i = 0; i < a->size(); i++) {
        // printf("%f\n", b[i]);
        float diff = a->get(i)->operator float() - b[i];
        sum += diff * diff;
    }
    
    return sum;
}

int* AssignClusters(ArffData *dataset, float *centroids, int k) {
    int *clusters = (int *) malloc(dataset->num_instances() * sizeof(int));

    for(int i = 0; i < 3; i++) {
        float min_distance = FLT_MAX;

        for(int j = 0; j < k; j++) {
            float dist = CalculateDistance(dataset->get_instance(i), &centroids[j * dataset->num_attributes()]);

            if(dist < min_distance) {
                min_distance = dist;
                clusters[i] = j;
            }
        }
    }

    return clusters;
}

int* KMeans(ArffData *dataset, int k) {
    int *clusters;

    return clusters;
}

int main(int argc, char *argv[]) {
    if(argc != 3)
    {
        cout << "Usage: ./kmeans datasets/dataset.arff k" << endl;
        exit(0);
    }

    // Number of clusters
    int k = strtol(argv[2], NULL, 10);

    srand(7);

    // Parse dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();

    float *centroids = (float *) malloc(k * 2 * sizeof(float));
    int* clusters = NULL;

    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    // Initialize centroids as random datapoints
    for(int i = 0; i < k * 2; i += 2) {
        int rand_point = rand() % (dataset->num_instances() + 1);

        centroids[i] = dataset->get_instance(rand_point)->get(i % 2)->operator float();
        centroids[i + 1] = dataset->get_instance(rand_point)->get((i + 1) % 2)->operator float();
        // printf("%d: %f, %f\n", i/2, centroids[i], centroids[i + 1]);
    }

    AssignClusters(dataset, centroids, k);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    free(centroids);
}
