// Warp counter in CUDA

#include <stdio.h>
#include <cuda_runtime.h>


// includes, project
////////////////////////////////////////////////////////////////////////////////
// declarations, forward

//extern "C"

// FILL HERE: define lock class
//            USe atomic operation for both lock and unlock functions
//            Fill free to use any atomic operation that correctly works for unlock.

class Lock
{
        int* mutex;
        public:
                Lock()
                {
                        int state =0;
                        cudaMalloc((void**) &mutex, sizeof(int));
                        cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
                }

                ~Lock()
                {
                        cudaFree(mutex);
                }

		__device__ void lock()
                {
                        while (atomicCAS(mutex, 0, 1) != 0);
                }

                __device__ void unlock()
                {
                        *mutex =0;
                }
};

/**
 * CUDA Kernel Device code
 * Computes cooperative additions
 */
// FILL HERE: Implement a kernel code that counts the total number of warps 
//            used in the kernel by using lock.

__global__ void myKernel(Lock lock, int *nwarps_dev)
{
	if(threadIdx.x % 32 == 0)
	{
		lock.lock();
		*nwarps_dev += 1;
		lock.unlock();
	}
}

/**
 * Host main routine
 */
int 
main(void) 
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int nwarps_host, *nwarps_dev;
    Lock lock;

    err = cudaMalloc((void**)&nwarps_dev, sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device nwarps (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    nwarps_host = 0;

    err = cudaMemcpy(nwarps_dev, &nwarps_host, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy nwarps from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Kernel Invocation 
    int blocksPerGrid = 125;
    int threadsPerBlock = 1000;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // FILL HERE: Defind a kernel invocation code that uses the blocksPerGrid blocks of threadsPerBlock threads
    myKernel<<<blocksPerGrid, threadsPerBlock>>> (lock, nwarps_dev);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch blockCounterUnLocked kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();
	
    // Copy the device result to the host 
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(&nwarps_host, nwarps_dev, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy A from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();

    printf("number of warps = %d\n", nwarps_host);
    
    // Free device global memory
    cudaFree(nwarps_dev);

    return 0;
}

