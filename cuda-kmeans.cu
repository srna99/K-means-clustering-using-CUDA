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


__device__ float CalculateDistance(float *a, float *b, int size) {
    float sum = 0;
    float diff;
    
    for(int i = 0; i < size - 1; i++) {
        diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return sum;
}

__global__ void AssignClusters(float *dataset, float *centroids, int *clusters, int k, int num_inst, int num_attr) {
    for(int i = 0; i < num_inst; i++) {
        float min_distance = FLT_MAX;

        for(int j = 0; j < k; j++) {
            float dist = CalculateDistance(dataset[i], &centroids[j * (num_attr - 1)]);

            if(dist < min_distance) {
                min_distance = dist;
                clusters[i] = j;
            }
        }
    }
}

__global__ void CalculateClusterMeans(float *dataset, float *centroids, int *clusters, int k, int num_inst, int num_attr) {
    for(int i = 0; i < num_inst; i++) {
        for(int j = 0; j < num_attr; j++) {
            sumOfValues[clusters[i] * num_attr + j] += dataset[i * num_attr + j];
        }

        clusterSize[clusters[i]]++;
    }

    for(int i = 0; i < k * num_attr; i++) {
        float newCentroid = sumOfValues[i] / clusterSize[(int) (i / num_attr)];

        centroidDiffs[i] = abs(centroids[i] - newCentroid);

        centroids[i] = newCentroid;
        printf("Centroid #%d: Attr%d = %f, Diff = %f, Cluster Size = %d\n", (int) (i / num_attr) + 1, i % num_attr, centroids[i], centroidDiffs[i], clusterSize[(int) (i / num_attr)]);
    }
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

    // Allocate host memory
    float *h_datapoints = (float *) malloc(num_inst * num_attr * sizeof(float));
    float *h_centroids = (float *) malloc(k * num_attr * sizeof(float));
    int *h_clusters = (int *) malloc(num_inst * sizeof(int));

    for(int i = 0; i < num_inst; i++) {
        for(int j = 0; j < num_attr; j++) {
            h_datapoints[i * num_attr + j] = dataset->get_instance(i)->get(j)->operator float();
        }
    }

    // Initialize centroids as random datapoints
    for(int i = 0; i < k; i++) {
        int rand_point = rand() % (num_inst + 1);

        for(int j = 0; j < num_attr; j++) {
            h_centroids[i * num_attr + j] = h_datapoints[rand_point * num_attr + j];
        }
    }

    // Allocate device memory
    float *d_datapoints
    float *d_centroids;
    int *d_clusters;
    float *d_sumOfValues
    // float *d_centroidDiffs;
    int *d_clusterSizes;

    cudaMalloc(&d_datapoints, num_inst * num_attr * sizeof(float));
    cudaMalloc(&d_centroids, k * num_attr * sizeof(float));
    cudaMalloc(&d_clusters, num_inst * sizeof(int));
    cudaMalloc(&d_sumOfValues, k * num_attr * sizeof(float));
    // cudaMalloc(&d_centroidDiffs, k * num_attr, sizeof(float));
    cudaMalloc(&d_clusterSizes, k * sizeof(int));

    cudaMemset(d_sumOfValues, 0, k * num_attr * sizeof(float));
    cudaMemset(d_clusterSizes, 0, k * sizeof(int));

    // Transfer host memory to device memory
    cudaMemcpy(d_datapoints, h_datapoints, num_inst * num_attr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, k * num_attr * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_clusters, h_clusters, num_inst * sizeof(int), cudaMemcpyHostToDevice);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int threadsPerBlock = 256;
    int gridSize = (num_inst + threadsPerBlock - 1) / threadsPerBlock;

    int iteration = 0;

    // Kmeans
    while(iteration < 150) {
        printf("----------------------- ITERATION %d ---------------------------\n", iteration);

        AssignClusters<<<gridSize, threadsPerBlock>>>(d_datapoints, d_centroids, d_clusters, k, num_inst, num_attr);

        CalculateClusterMeans<<<gridSize, threadsPerBlock>>>(d_datapoints, d_centroids, d_clusters, d_sumOfValues, d_clusterSizes, k, num_inst, num_attr);

        iteration++;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Transfer device memory to host memory
    cudaMemcpy(h_clusters, d_clusters, num_inst * sizeof(int), cudaMemcpyDeviceToHost);

    cudaError_t cudaError = cudaGetLastError();
  
    if(cudaError != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    printf("It took %llu ms to process %d datapoints into %d clusters.\n", milliseconds, num_inst, k);

    cudaFree(d_datapoints);
    cudaFree(d_centroids);
    cudaFree(d_clusters);
    cudaFree(d_sumOfValues);
    cudaFree(d_centroidDiffs);
    cudaFree(d_clusterSizes);
    free(h_datapoints);
    free(h_centroids);
    free(h_clusters);
}
