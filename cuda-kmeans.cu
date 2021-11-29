#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <cuda_runtime.h>

using namespace std;


#define THREADS_PER_BLOCK_DATA 256
#define THREADS_PER_BLOCK_CENTROIDS 32

// Calculate distance from one datapoint to another
__device__ float CalculateDistance(float *a, float *b, int size) {
    float sum = 0;
    float diff;
    
    for(int i = 0; i < size; i++) {
        diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return sum;
}

// Assign each datapoint to nearest centroid
__global__ void AssignClusters(float *dataset, float *centroids, int *clusters, int k, int numInst, int numAttr) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < numInst) {
        float minDistance = FLT_MAX;

        for(int i = 0; i < k; i++) {
            float dist = CalculateDistance(&dataset[tid * numAttr], &centroids[i * numAttr], numAttr);

            if(dist < minDistance) {
                minDistance = dist;
                clusters[tid] = i;
            }
        }
    }
}

// Calculate the sums for each cluster and their total sizes
__global__ void CalculateClusterSumsAndSizes(float *dataset, int *clusters, float *sumOfCentroids, int *clusterSizes, int k, int numInst, int numAttr) {
    extern __shared__ int sharedMemory[];

    int *sharedClusters = sharedMemory;
    float *sharedDataset = (float *) &sharedMemory[THREADS_PER_BLOCK_DATA];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int th_idx = threadIdx.x;
    
    if(tid < numInst) {
        for(int i = 0; i < numAttr; i++) {
            sharedDataset[th_idx * numAttr + i] = dataset[tid * numAttr + i];
        }

        sharedClusters[th_idx] = clusters[tid];
    } else {
        for(int i = 0; i < numAttr; i++) {
            sharedDataset[th_idx * numAttr + i] = 0;
        }

        sharedClusters[th_idx] = -1;
    }

    __syncthreads();

    if(th_idx < numAttr) {
        float attrSum[THREADS_PER_BLOCK_CENTROIDS] = {0};

        for(int i = 0; i < THREADS_PER_BLOCK_DATA; i++) {
            int cluster = sharedClusters[i];

            if(cluster != -1) {
                attrSum[cluster] += sharedDataset[i * numAttr + th_idx];
            }
        }

        for(int i = 0; i < k; i++) {
            atomicAdd(&sumOfCentroids[i * numAttr + th_idx], attrSum[i]);
        }
    } else if(th_idx == numAttr + 1) {
        int clusterAmounts[THREADS_PER_BLOCK_CENTROIDS] = {0};

        for(int i = 0; i < THREADS_PER_BLOCK_DATA; i++) {
            int cluster = sharedClusters[i];

            if(cluster != -1) {
                clusterAmounts[cluster] += 1;
            }
        }

        for(int i = 0; i < k; i++) {
            atomicAdd(&clusterSizes[i], clusterAmounts[i]);
        }
    }
}

// Calculate means for each cluster for new centroids
__global__ void CalculateCentroidMeans(float *centroids, float *sumOfCentroids, int *clusterSizes, float *centroidDiffs, int k, int numAttr) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < k * numAttr) {
        float newCentroid = sumOfCentroids[tid] / clusterSizes[(int) (tid / numAttr)];

        centroidDiffs[tid] = abs(centroids[tid] - newCentroid);

        centroids[tid] = newCentroid;
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
    float *h_centroidDiffs = (float *) malloc(k * numAttr * sizeof(float));;
    int *h_clusterSizes = (int *) malloc(k * sizeof(int));

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
    float *d_sumOfCentroids;
    float *d_centroidDiffs;
    int *d_clusterSizes;

    cudaMalloc(&d_dataset, numInst * numAttr * sizeof(float));
    cudaMalloc(&d_centroids, k * numAttr * sizeof(float));
    cudaMalloc(&d_clusters, numInst * sizeof(int));
    cudaMalloc(&d_sumOfCentroids, k * numAttr * sizeof(float));
    cudaMalloc(&d_centroidDiffs, k * numAttr * sizeof(float));
    cudaMalloc(&d_clusterSizes, k * sizeof(int));

    cudaMemset(d_sumOfCentroids, 0, k * numAttr * sizeof(float));
    cudaMemset(d_centroidDiffs, 0, k * numAttr * sizeof(float));
    cudaMemset(d_clusterSizes, 0, k * sizeof(int));

    // Transfer host memory to device memory
    cudaMemcpy(d_dataset, h_dataset, numInst * numAttr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, k * numAttr * sizeof(float), cudaMemcpyHostToDevice);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int gridSizeForDataset = (numInst + THREADS_PER_BLOCK_DATA - 1) / THREADS_PER_BLOCK_DATA;
    int gridSizeForCentroids = (numAttr * k + THREADS_PER_BLOCK_CENTROIDS - 1) / THREADS_PER_BLOCK_CENTROIDS;
    int sharedMemorySize = THREADS_PER_BLOCK_DATA * numAttr * sizeof(float) + THREADS_PER_BLOCK_DATA * sizeof(int);

    cudaError_t cudaError;
    int iteration = 0;

    // Kmeans
    while(iteration < 130) {
        // printf("----------------------- ITERATION %d ---------------------------\n", iteration);

        AssignClusters<<<gridSizeForDataset, THREADS_PER_BLOCK_DATA>>>(d_dataset, d_centroids, d_clusters, k, numInst, numAttr);

        cudaMemset(d_sumOfCentroids, 0, k * numAttr * sizeof(float));
        cudaMemset(d_centroidDiffs, 0, k * numAttr * sizeof(float));
        cudaMemset(d_clusterSizes, 0, k * sizeof(int));

        CalculateClusterSumsAndSizes<<<gridSizeForDataset, THREADS_PER_BLOCK_DATA, sharedMemorySize>>>(d_dataset, d_clusters, d_sumOfCentroids, d_clusterSizes, k, numInst, numAttr);

        CalculateCentroidMeans<<<gridSizeForCentroids, THREADS_PER_BLOCK_CENTROIDS>>>(d_centroids, d_sumOfCentroids, d_clusterSizes, d_centroidDiffs, k, numAttr);

        // cudaMemcpy(h_centroids, d_centroids, k * numAttr * sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_centroidDiffs, d_centroidDiffs, k * numAttr * sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_clusterSizes, d_clusterSizes, k * sizeof(int), cudaMemcpyDeviceToHost);

        cudaError = cudaGetLastError();
  
        if(cudaError != cudaSuccess) {
            fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);
        }

        // for(int i = 0; i < k * numAttr; i++) {
        //     printf("Centroid #%d: Attr%d = %f, Diff = %f, Cluster Size = %d\n", (int) (i / numAttr) + 1, i % numAttr, h_centroids[i], h_centroidDiffs[i], h_clusterSizes[(int) (i / numAttr)]);
        // }

        iteration++;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Transfer device memory to host memory
    cudaMemcpy(h_clusters, d_clusters, numInst * sizeof(int), cudaMemcpyDeviceToHost);

    printf("It took %d ms to process %d datapoints into %d clusters.\n", (int) milliseconds, numInst, k);

    cudaFree(d_dataset);
    cudaFree(d_centroids);
    cudaFree(d_clusters);
    cudaFree(d_sumOfCentroids);
    cudaFree(d_centroidDiffs);
    cudaFree(d_clusterSizes);
    free(h_dataset);
    free(h_centroids);
    free(h_clusters);
    free(h_centroidDiffs);
    free(h_clusterSizes);
}
