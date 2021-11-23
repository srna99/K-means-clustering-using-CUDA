#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
// #include <bits/stdc++.h>
#include <cuda_runtime.h>
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __shared__
#endif

using namespace std;


float CalculateDistance(ArffInstance *a, float *b) {
    float sum = 0;
    
    for(int i = 0; i < a->size() - 1; i++) {
        float diff = a->get(i)->operator float() - b[i];
        sum += diff * diff;
    }
    
    return sum;
}

void AssignClusters(ArffData *dataset, float *centroids, int k, int *clusters) {
    for(int i = 0; i < dataset->num_instances(); i++) {
        float min_distance = FLT_MAX;

        for(int j = 0; j < k; j++) {
            float dist = CalculateDistance(dataset->get_instance(i), &centroids[j * (dataset->num_attributes() - 1)]);

            if(dist < min_distance) {
                min_distance = dist;
                clusters[i] = j;
            }
        }
    }
}

void CalculateClusterMeans(ArffData *dataset, int *clusters, float *centroids, int k) {
    int num_attr = dataset->num_attributes() - 1;

    float *sumOfValues = (float *) calloc(k * num_attr, sizeof(float));
    float *centroidDiffs = (float *) malloc(k * num_attr * sizeof(float));
    int *clusterSize = (int *) calloc(k, sizeof(int));

    for(int i = 0; i < dataset->num_instances(); i++) {
        for(int j = 0; j < num_attr; j++) {
            sumOfValues[clusters[i] * num_attr + j] += dataset->get_instance(i)->get(j)->operator float();
        }

        clusterSize[clusters[i]]++;
    }

    // bool finished = true;

    for(int i = 0; i < k * num_attr; i++) {
        float newCentroid = sumOfValues[i] / clusterSize[(int) (i / num_attr)];

        centroidDiffs[i] = abs(centroids[i] - newCentroid);
        // finished = finished && (centroidDiffs[i] <= 1e-6);

        centroids[i] = newCentroid;
        printf("Centroid #%d: Attr%d = %f, Diff = %f, Cluster Size = %d\n", (int) (i / num_attr) + 1, i % num_attr, centroids[i], centroidDiffs[i], clusterSize[(int) (i / num_attr)]);
    }

    free(sumOfValues);
    free(centroidDiffs);
    free(clusterSize);
}

int main(int argc, char *argv[]) {
    if(argc != 3)
    {
        cout << "Usage: ./kmeans datasets/dataset.arff k" << endl;
        exit(0);
    }

    // Number of clusters
    int k = strtol(argv[2], NULL, 10);

    srand(13);

    // Parse dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();

    int num_attr = dataset->num_attributes() - 1;
    int num_inst = dataset->num_instances();

    float *centroids = (float *) malloc(k * num_attr * sizeof(float));
    int *clusters = (int *) malloc(num_inst * sizeof(int));

    // Initialize centroids as random datapoints
    for(int i = 0; i < k; i++) {
        int rand_point = rand() % (num_inst + 1);

        for(int j = 0; j < num_attr; j++) {
            centroids[i * num_attr + j] = dataset->get_instance(rand_point)->get(j)->operator float();
        }
    }

    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    int iteration = 0;

    // Kmeans
    while(iteration < 150) {
        printf("----------------------- ITERATION %d ---------------------------\n", iteration);

        AssignClusters(dataset, centroids, k, clusters);

        CalculateClusterMeans(dataset, clusters, centroids, k);

        iteration++;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("It took %llu ms to process %d datapoints into %d clusters.\n", diff, num_inst, k);

    free(centroids);
    free(clusters);
}
