#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

using namespace std;


float CalculateDistance(ArffInstance *a, float *b) {
    float sum = 0;
    float diff;
    
    for(int i = 0; i < a->size() - 1; i++) {
        diff = a->get(i)->operator float() - b[i];
        sum += diff * diff;
    }
    
    return sum;
}

void AssignClusters(ArffData *dataset, float *centroids, int k, int *clusters) {
    for(int i = 0; i < dataset->num_instances(); i++) {
        float minDistance = FLT_MAX;

        for(int j = 0; j < k; j++) {
            float dist = CalculateDistance(dataset->get_instance(i), &centroids[j * (dataset->num_attributes() - 1)]);

            if(dist < minDistance) {
                minDistance = dist;
                clusters[i] = j;
            }
        }
    }
}

void CalculateClusterMeans(ArffData *dataset, int *clusters, float *centroids, int k) {
    int numAttr = dataset->num_attributes() - 1;

    float *sumOfCentroids = (float *) calloc(k * numAttr, sizeof(float));
    float *centroidDiffs = (float *) malloc(k * numAttr * sizeof(float));
    int *clusterSizes = (int *) calloc(k, sizeof(int));

    for(int i = 0; i < dataset->num_instances(); i++) {
        for(int j = 0; j < numAttr; j++) {
            sumOfCentroids[clusters[i] * numAttr + j] += dataset->get_instance(i)->get(j)->operator float();
        }

        clusterSizes[clusters[i]]++;
    }

    // bool finished = true;

    for(int i = 0; i < k * numAttr; i++) {
        float newCentroid = sumOfCentroids[i] / clusterSizes[(int) (i / numAttr)];

        centroidDiffs[i] = abs(centroids[i] - newCentroid);
        // finished = finished && (centroidDiffs[i] <= 1e-6);

        centroids[i] = newCentroid;

        // printf("Centroid #%d: Attr%d = %f, Diff = %f, Cluster Size = %d\n", (int) (i / numAttr) + 1, i % numAttr, centroids[i], centroidDiffs[i], clusterSizes[(int) (i / numAttr)]);
    }

    free(sumOfCentroids);
    free(centroidDiffs);
    free(clusterSizes);

    // return finished;
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

    int numAttr = dataset->num_attributes() - 1;
    int numInst = dataset->num_instances();

    float *centroids = (float *) malloc(k * numAttr * sizeof(float));
    int *clusters = (int *) malloc(numInst * sizeof(int));

    // Initialize centroids as random datapoints
    for(int i = 0; i < k; i++) {
        int randPoint = rand() % (numInst + 1);

        for(int j = 0; j < numAttr; j++) {
            centroids[i * numAttr + j] = dataset->get_instance(randPoint)->get(j)->operator float();
        }
    }

    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    int iteration = 0;
    // bool finished = false;

    // Kmeans
    while(iteration < 120) {
        // printf("----------------------- ITERATION %d ---------------------------\n", iteration);

        AssignClusters(dataset, centroids, k, clusters);

        CalculateClusterMeans(dataset, clusters, centroids, k);

        iteration++;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("It took %llu ms to process %d datapoints into %d clusters in %d iterations.\n", diff, numInst, k, iteration);

    free(centroids);
    free(clusters);
}
