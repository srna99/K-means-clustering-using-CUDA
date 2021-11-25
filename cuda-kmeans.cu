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

__global__ void AssignClusters(float *dataset, float *centroids, int *clusters, int k, int numInst, int numAttr) {
    for(int i = 0; i < num_inst; i++) {
        float minDistance = FLT_MAX;

        for(int j = 0; j < k; j++) {
            float dist = CalculateDistance(dataset[i], &centroids[j * (numAttr - 1)]);

            if(dist < minDistance) {
                minDistance = dist;
                clusters[i] = j;
            }
        }
    }
}

__global__ void CalculateClusterMeans(float *dataset, float *centroids, int *clusters, int k, int numInst, int numAttr) {
    for(int i = 0; i < numInst; i++) {
        for(int j = 0; j < numAttr; j++) {
            sumOfValues[clusters[i] * numAttr + j] += dataset[i * numAttr + j];
        }

        clusterSize[clusters[i]]++;
    }

    for(int i = 0; i < k * numAttr; i++) {
        float newCentroid = sumOfValues[i] / clusterSize[(int) (i / numAttr)];

        centroidDiffs[i] = abs(centroids[i] - newCentroid);

        centroids[i] = newCentroid;
        printf("Centroid #%d: Attr%d = %f, Diff = %f, Cluster Size = %d\n", (int) (i / numAttr) + 1, i % numAttr, centroids[i], centroidDiffs[i], clusterSize[(int) (i / numAttr)]);
    }
}

int main(int argc, char *argv[]) {
    if(argc != 3)
    {
        cout << "Usage: ./cuda-kmeans datasets/dataset.arff k" << endl;
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

    // Allocate host memory
    float *h_dataset = (float *) malloc(numInst * numAttr * sizeof(float));
    float *h_centroids = (float *) malloc(k * numAttr * sizeof(float));
    int *h_clusters = (int *) malloc(numInst * sizeof(int));

    for(int i = 0; i < numInst; i++) {
        for(int j = 0; j < numAttr; j++) {
            h_dataset[i * numAttr + j] = dataset->get_instance(i)->get(j)->operator float();
        }
    }

    // Initialize centroids as random datapoints
    for(int i = 0; i < k; i++) {
        int randPoint = rand() % (numInst + 1);

        for(int j = 0; j < numAttr; j++) {
            h_centroids[i * numAttr + j] = h_dataset[randPoint * numAttr + j];
        }
    }

    // Allocate device memory
    float *d_dataset;
    float *d_centroids;
    int *d_clusters;
    float *d_sumOfValues;
    // float *d_centroidDiffs;
    int *d_clusterSizes;

    cudaMalloc(&d_dataset, numInst * numAttr * sizeof(float));
    cudaMalloc(&d_centroids, k * numAttr * sizeof(float));
    cudaMalloc(&d_clusters, numInst * sizeof(int));
    cudaMalloc(&d_sumOfValues, k * numAttr * sizeof(float));
    // cudaMalloc(&d_centroidDiffs, k * num_attr, sizeof(float));
    cudaMalloc(&d_clusterSizes, k * sizeof(int));

    cudaMemset(d_sumOfValues, 0, k * numAttr * sizeof(float));
    cudaMemset(d_clusterSizes, 0, k * sizeof(int));

    // Transfer host memory to device memory
    cudaMemcpy(d_dataset, h_dataset, numInst * numAttr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, k * numAttr * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_clusters, h_clusters, num_inst * sizeof(int), cudaMemcpyHostToDevice);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int threadsPerBlock = 256;
    int gridSize = (numInst + threadsPerBlock - 1) / threadsPerBlock;

    int iteration = 0;

    // Kmeans
    while(iteration < 150) {
        printf("----------------------- ITERATION %d ---------------------------\n", iteration);

        AssignClusters<<<gridSize, threadsPerBlock>>>(d_dataset, d_centroids, d_clusters, k, numInst, numAttr);

        cudaMemset(d_sumOfValues, 0, k * numAttr * sizeof(float));
        cudaMemset(d_clusterSizes, 0, k * sizeof(int));

        CalculateClusterMeans<<<gridSize, threadsPerBlock>>>(d_dataset, d_centroids, d_clusters, d_sumOfValues, d_clusterSizes, k, numInst, numAttr);

        iteration++;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Transfer device memory to host memory
    cudaMemcpy(h_clusters, d_clusters, numInst * sizeof(int), cudaMemcpyDeviceToHost);

    cudaError_t cudaError = cudaGetLastError();
  
    if(cudaError != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    printf("It took %llu ms to process %d datapoints into %d clusters.\n", milliseconds, numInst, k);

    cudaFree(d_dataset);
    cudaFree(d_centroids);
    cudaFree(d_clusters);
    cudaFree(d_sumOfValues);
    // cudaFree(d_centroidDiffs);
    cudaFree(d_clusterSizes);
    free(h_dataset);
    free(h_centroids);
    free(h_clusters);
}
