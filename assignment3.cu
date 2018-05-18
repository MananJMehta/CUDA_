/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <sys/time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
//#define WARP_SIZE 32 

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
// FILL HERE: translate C-version vectorAdd to CUDA-version kernel code

__global__ void vectorAdd(const float *A, const float *B, float *C) 
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	C[i] = A[i] + B[i];
}

/**
 * Host main routine
 */
int
main(void)
{


    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size

    /*CHOOSE NUMBER OF ELEMENTS*/
    int numElements = 524288; 
    size_t size = numElements * sizeof(float);


    /*SEQUENTIAL VECTOR ADDITION*/
    printf("[Sequential Vector addition of %d elements]\n", numElements);


    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void**) &d_A, numElements * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void**) &d_B, numElements * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void**) &d_C, numElements * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*TIMER START*/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice); 

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice); 

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
	// FILL HERE: call 'vectorAdd' function with 
	//            4 blocks of 256 threads 
    
    int threadsPerBlock = 256;
    int blocksPerGrid = numElements / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vectorAdd <<< blocksPerGrid, threadsPerBlock >>>(d_A, d_B, d_C);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost); 

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*TIMER END*/
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("The elapsed time for Sequential Vector Add is:%fms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("The elapsed time is %.2f ms\n", milliseconds);

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED for sequential.\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*STREAM BASED VECTOR ADDITION*/

    printf("[Stream Based Vector addition of %d elements]\n", numElements);

    int numElementsPerStream = (numElements/4);
    // Allocate the device input vector A0
    float *d_A0 = NULL;
    err = cudaMalloc((void**) &d_A0, numElementsPerStream * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A0 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector A1
    float *d_A1 = NULL;
    err = cudaMalloc((void**) &d_A1, numElementsPerStream * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector A2
    float *d_A2 = NULL;
    err = cudaMalloc((void**) &d_A2, numElementsPerStream * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector A3
    float *d_A3 = NULL;
    err = cudaMalloc((void**) &d_A3, numElementsPerStream * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B0
    float *d_B0 = NULL;
    err = cudaMalloc((void**) &d_B0, numElementsPerStream * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B0 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B1
    float *d_B1 = NULL;
    err = cudaMalloc((void**) &d_B1, numElementsPerStream * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B2
    float *d_B2 = NULL;
    err = cudaMalloc((void**) &d_B2, numElementsPerStream * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B3
    float *d_B3 = NULL;
    err = cudaMalloc((void**) &d_B3, numElementsPerStream * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector C0
    float *d_C0 = NULL;
    err = cudaMalloc((void**) &d_C0, numElementsPerStream * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C0 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector C1
    float *d_C1 = NULL;
    err = cudaMalloc((void**) &d_C1, numElementsPerStream * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector C2
    float *d_C2 = NULL;
    err = cudaMalloc((void**) &d_C2, numElementsPerStream * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector C3
    float *d_C3 = NULL;
    err = cudaMalloc((void**) &d_C3, numElementsPerStream * sizeof(float)); // FILL HERE

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*CREATE STREAMS*/
    cudaStream_t stream0, stream1,stream2,stream3;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    /*TIMER START*/
    cudaEvent_t start_overlapped, stop_overlapped;
    cudaEventCreate(&start_overlapped);
    cudaEventCreate(&stop_overlapped);
    cudaEventRecord(start_overlapped);

    for (int i = 0; i < size; i+=numElements*4)
    {
		err = cudaMemcpyAsync(d_A0, h_A + i, numElementsPerStream*sizeof(float), cudaMemcpyHostToDevice, stream0);

        if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy vector A0 from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

		err = cudaMemcpyAsync(d_B0, h_B + i, numElementsPerStream*sizeof(float), cudaMemcpyHostToDevice, stream0);

        if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy vector B0 from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

		err = cudaMemcpyAsync(d_A1, h_A + i + numElementsPerStream  , numElementsPerStream*sizeof(float), cudaMemcpyHostToDevice, stream1);

        if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy vector A1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

		err = cudaMemcpyAsync(d_B1, h_B + i + numElementsPerStream  , numElementsPerStream*sizeof(float), cudaMemcpyHostToDevice, stream1);

        if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy vector B1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

		err = cudaMemcpyAsync(d_A2, h_A + i + numElementsPerStream*2, numElementsPerStream*sizeof(float), cudaMemcpyHostToDevice, stream2);

        if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy vector A2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

		err = cudaMemcpyAsync(d_B2, h_B + i + numElementsPerStream*2, numElementsPerStream*sizeof(float), cudaMemcpyHostToDevice, stream2);

        if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy vector B2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

		err = cudaMemcpyAsync(d_A3, h_A + i + numElementsPerStream*3, numElementsPerStream*sizeof(float), cudaMemcpyHostToDevice, stream3);

        if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy vector A3 from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

		err = cudaMemcpyAsync(d_B3, h_B + i + numElementsPerStream*3, numElementsPerStream*sizeof(float), cudaMemcpyHostToDevice, stream3);

        if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy vector B3 from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}


	//Launch kernel on stream 0

	vectorAdd <<<blocksPerGrid/4,threadsPerBlock,0,stream0>>> (d_A0, d_B0, d_C0);

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to launch vectorAdd kernel for stream 0 (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
   	}

	//Launch kernel on stream 1
 
	vectorAdd <<<blocksPerGrid/4,threadsPerBlock,0,stream1>>> (d_A1, d_B1, d_C1);

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to launch vectorAdd kernel for stream 1 (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
   	}

	//Launch kernel on stream 2
 
	vectorAdd <<<blocksPerGrid/4,threadsPerBlock,0,stream2>>> (d_A2, d_B2, d_C2);

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to launch vectorAdd kernel for stream 2 (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
   	}

	//Launch kernel on stream 3
 
	vectorAdd <<<blocksPerGrid/4,threadsPerBlock,0,stream3>>> (d_A3, d_B3, d_C3);

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to launch vectorAdd kernel for stream 3 (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
   	}

	//Copy back from Device to Host

    	printf("Copy output data from the CUDA device to the host memory\n");

	err = cudaMemcpyAsync(h_C + i, d_C0, numElementsPerStream*sizeof(float), cudaMemcpyDeviceToHost, stream0);

    	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy vector C0 from device to host (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	err = cudaMemcpyAsync(h_C + i + numElementsPerStream  , d_C1, numElementsPerStream*sizeof(float), cudaMemcpyDeviceToHost, stream1);

    	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy vector C1 from device to host (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	err = cudaMemcpyAsync(h_C + i + numElementsPerStream*2, d_C2, numElementsPerStream*sizeof(float), cudaMemcpyDeviceToHost, stream2);

    	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy vector C2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	err = cudaMemcpyAsync(h_C + i + numElementsPerStream*3, d_C3, numElementsPerStream*sizeof(float), cudaMemcpyDeviceToHost, stream3);

    	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy vector C3 from device to host (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
    }


    /*TIMER END*/
    cudaEventRecord(stop_overlapped);
    cudaEventSynchronize(stop_overlapped);
    float milliseconds_overlapped = 0;
    cudaEventElapsedTime(&milliseconds_overlapped, start_overlapped, stop_overlapped);
    printf("The elapsed time for Overlapped Vector Add is:%fms\n", milliseconds_overlapped);
    cudaEventDestroy(start_overlapped);
    cudaEventDestroy(stop_overlapped);

    // Verify that the result vector is correct for Stream based Vector add
    for (int i = 0; i < numElements; ++i)
    {
	if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
	{
		fprintf(stderr, "Result verification failed at element %d!\n", i);
		exit(EXIT_FAILURE);
	}	
    }
    printf("Test PASSED for Stream based Vector Add \n");

    // Free Device Memory
    err = cudaFree(d_A0);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A0 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_A1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_A2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_A3);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B0);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B0 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B3);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C0);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C0 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C3);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

