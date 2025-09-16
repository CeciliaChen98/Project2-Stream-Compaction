#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i-1];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    count++;
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* b = new int[n];
            for (int i = 0; i < n; i++) {
                if (idata[i] == 0) {
                    b[i] = 0;
                }
                else {
                    b[i] = 1;
                }
            }
            int* s = new int[n];
            s[0] = 0;
            // Implement scan
            for (int i = 1; i < n; i++) {
                s[i] = s[i - 1] + b[i - 1];
            }
            // Scatter
            for (int i = 0; i < n; i++) {
                if (b[i] == 1) {
                    odata[s[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            return s[n-1] + b[n-1];
        }

    }
}
