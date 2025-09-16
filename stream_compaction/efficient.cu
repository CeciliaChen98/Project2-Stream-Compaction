#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int* data, int d) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int offset = 1 << (d + 1);                  
            index = (index + 1) * offset - 1;   
            if (index >= n) {
                return;
            }
            int left = index - (offset >> 1);
            data[index] += data[left];
        }

        __global__ void kernDownSweep(int n, int* data, int d) {
            int offset = 1 << (d + 1);
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            index = (index + 1) * offset - 1;
            if (index >= n) {
                return;
            }
            int left = index - (offset >> 1);
            int t = data[left];
            data[left] = data[index];
            data[index] += t;
        }

        __global__ void kernSetZero(int n, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index == 0) {
                data[n - 1] = 0;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {

            int* d_data;

            int sizeInBytes = n * sizeof(int);

            int level = ilog2ceil(n);
            int size = 1 << level;
            int sizeInBytes2 = size * sizeof(int);
            cudaMalloc((void**)&d_data, sizeInBytes2);

            const int BLOCK = 256;

            cudaMemset(d_data, 0, sizeInBytes2);
            cudaMemcpy(d_data, idata, sizeInBytes, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // Up Sweep
            for (int d = 0; d < level; d ++) {
                int nodes = size >> (d + 1);
                int grid = (nodes + BLOCK - 1) / BLOCK;
                kernUpSweep <<<grid, BLOCK >> > (size, d_data, d);
            }
            kernSetZero <<<1, 1>>> (size, d_data);
            // Down Sweep
            for (int d = level - 1; d > -1; --d) {
                int nodes = size >> (d + 1);
                int grid = (nodes + BLOCK - 1) / BLOCK;
                kernDownSweep <<<grid, BLOCK >>> (size, d_data, d);
            }

            cudaDeviceSynchronize();
            timer().endGpuTimer();

            cudaMemcpy(odata, d_data, sizeInBytes, cudaMemcpyDeviceToHost);
            cudaFree(d_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

            int* d_in, * d_out, * d_bools, * d_indices;
            int sizeInBytes = n * sizeof(int);
            cudaMalloc(&d_in, sizeInBytes);
            cudaMalloc(&d_out, sizeInBytes);
            cudaMalloc(&d_bools, sizeInBytes);
            cudaMalloc(&d_indices, sizeInBytes);
            cudaMemcpy(d_in, idata, sizeInBytes, cudaMemcpyHostToDevice);

            int* d_data;
            int level = ilog2ceil(n);
            int size = 1 << level;
            int sizeInBytes2 = size * sizeof(int);
            cudaMalloc((void**)&d_data, sizeInBytes2);
            cudaMemset(d_data, 0, sizeInBytes2);

            const int BLOCK = 256;
            const int GRID = (n + BLOCK - 1) / BLOCK;

            timer().startGpuTimer();
            // Step 1: Map to booleans
            StreamCompaction::Common::kernMapToBoolean <<<GRID, BLOCK>>> (n, d_bools, d_in);

            // Step 2: Scan
            cudaMemcpy(d_data, d_bools, sizeInBytes, cudaMemcpyDeviceToDevice);
            // Up Sweep
            for (int d = 0; d < level; d++) {
                int nodes = size >> (d + 1);
                int g = (nodes + BLOCK - 1) / BLOCK;
                kernUpSweep <<<g, BLOCK >> > (size, d_data, d);
            }
            kernSetZero <<<1, 1 >> > (size, d_data);
            // Down Sweep
            for (int d = level - 1; d > -1; --d) {
                int nodes = size >> (d + 1);
                int g = (nodes + BLOCK - 1) / BLOCK;
                kernDownSweep <<<g, BLOCK >> > (size, d_data, d);
            }
            cudaDeviceSynchronize();
            cudaMemcpy(d_indices, d_data, sizeInBytes, cudaMemcpyDeviceToDevice);

            // Step 3: Scatter
            StreamCompaction::Common::kernScatter <<<GRID, BLOCK >>> (n, d_out, d_in, d_bools, d_indices);
            
            // Step 4: Calculate Count
            int lastIdx, lastBool = 0;
            cudaMemcpy(&lastIdx, d_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, d_bools + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIdx + lastBool;

            timer().endGpuTimer();

            cudaMemcpy(odata, d_out, count * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_in); 
            cudaFree(d_out); 
            cudaFree(d_bools); 
            cudaFree(d_indices);
            cudaFree(d_data);

            return count;
        }
    }
}
