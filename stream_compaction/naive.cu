#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernInclusiveToExclusive(int n, int* out, const int* in) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            out[index] = (index == 0) ? 0 : in[index - 1];
        }

        __global__ void kernNaiveScan(int n, int* out, const int* in, int offset) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            int v = in[index];
            if (index >= offset) {
                v += in[index - offset];
            }
            out[index] = v;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int* d_in, * d_out;
            int sizeInBytes = n * sizeof(int);
            cudaMalloc((void**)&d_in, sizeInBytes);
            cudaMalloc((void**)&d_out, sizeInBytes);

            const int BLOCK = 512;
            const int GRID = (n + BLOCK - 1) / BLOCK;

            cudaMemcpy(d_in, idata, sizeInBytes, cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();

            for (int offset = 1; offset < n; offset <<= 1) {
                kernNaiveScan <<<GRID, BLOCK>>> (n, d_out, d_in, offset);
                std::swap(d_in, d_out);
            }

            kernInclusiveToExclusive <<<GRID, BLOCK >>> (n, d_out, d_in);
            timer().endGpuTimer();

            cudaMemcpy(odata, d_out, sizeInBytes, cudaMemcpyDeviceToHost);
            cudaFree(d_in);
            cudaFree(d_out);
        }
    }
}
