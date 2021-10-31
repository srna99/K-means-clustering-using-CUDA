#include <stdio.h>
#include <stdlib.h>
// #include <stdint.h>
#include <float.h>
#include <math.h>
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

void AssignClusters(ArffData *dataset, float *centroids, int k, int *clusters) {
    for(int i = 0; i < dataset->num_instances(); i++) {
        float min_distance = FLT_MAX;

        for(int j = 0; j < k; j++) {
            float dist = CalculateDistance(dataset->get_instance(i), &centroids[j * dataset->num_attributes()]);

            if(dist < min_distance) {
                min_distance = dist;
                clusters[i] = j;
            }
        }
    }
}

bool CalculateClusterMeans(ArffData *dataset, int *clusters, float *centroids, int k) {
    float *sumOfValues = (float *) calloc(k * 2, sizeof(float));
    float *centroidDiffs = (float *) malloc(k * 2 * sizeof(float));
    int *clusterSize = (int *) calloc(k, sizeof(int));

    for(int i = 0; i < dataset->num_instances(); i++) {
        sumOfValues[clusters[i] * 2] += dataset->get_instance(i)->get(0)->operator float();
        sumOfValues[clusters[i] * 2 + 1] += dataset->get_instance(i)->get(1)->operator float();

        clusterSize[clusters[i]]++;
        // printf("%f, %f, %d, %d, %d\n", sumOfValues[clusters[i] * 2], sumOfValues[clusters[i] * 2 + 1], clusterSize[clusters[i]], clusters[i], i);
    }

    bool finished = true;

    for(int i = 0; i < k * 2; i++) {
        float newCentroid = sumOfValues[i] / clusterSize[(int) (i / 2)];

        centroidDiffs[i] = abs(centroids[i] - newCentroid);
        finished = finished && (centroidDiffs[i] <= 1e-6);

        centroids[i] = newCentroid;
        printf("%f, %f, %d\n", centroids[i], centroidDiffs[i], clusterSize[(int) (i / 2)]);
    }

    printf("%s\n", finished ? "true" : "false");

    free(sumOfValues);
    free(centroidDiffs);
    free(clusterSize);

    return finished;
}

int main(int argc, char *argv[]) {
    if(argc != 3)
    {
        cout << "Usage: ./kmeans datasets/dataset.arff k" << endl;
        exit(0);
    }

    // Number of clusters
    int k = strtol(argv[2], NULL, 10);

    srand(1);

    // Parse dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();

    float *centroids = (float *) malloc(k * 2 * sizeof(float));
    int *clusters = (int *) malloc(dataset->num_instances() * sizeof(int));

    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    // Initialize centroids as random datapoints
    for(int i = 0; i < k; i++) {
        int rand_point = rand() % (dataset->num_instances() + 1);

        centroids[i] = dataset->get_instance(rand_point)->get(0)->operator float();
        centroids[i + 1] = dataset->get_instance(rand_point)->get(1)->operator float();
        // printf("%d: %f, %f\n", i/2, centroids[i], centroids[i + 1]);
    }

    int iteration = 0;
    bool finished = false;

    while(iteration < 1000 && !finished) {
        printf("----------------------- ITERATION %d ---------------------------\n", iteration);
        AssignClusters(dataset, centroids, k, clusters);

        finished = CalculateClusterMeans(dataset, clusters, centroids, k);

        iteration++;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("It took %llu ms to process %lu datapoints into %d clusters.\n", diff, dataset->num_instances(), k);

    free(centroids);
    free(clusters);
}
