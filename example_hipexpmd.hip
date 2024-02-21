/* MIT License
 *
 * Copyright (c) 2024 Maximilian Behr
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "hipexpm.h"

int main(void) {
    /*-----------------------------------------------------------------------------
     * variables
     *-----------------------------------------------------------------------------*/
    int ret = 0;      // return value
    int n = 1 << 10;  // size of the matrix

    double *A, *expmA;      // A and expmA on the host
    double *d_A, *d_expmA;  // A and expmA on the device
    void *d_buffer = NULL;  // memory buffer on the device
    void *h_buffer = NULL;  // memory buffer on the host

    /*------------------------  -----------------------------------------------------
     * allocate A and expmA on the host
     *-----------------------------------------------------------------------------*/
    hipHostMalloc((void **)&A, sizeof(*A) * n * n);
    hipHostMalloc((void **)&expmA, sizeof(*expmA) * n * n);

    /*-----------------------------------------------------------------------------
     * fill matrix A
     *-----------------------------------------------------------------------------*/
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (i >= j) {
                A[i + j * n] = -1.;
            } else {
                A[i + j * n] = -2.;
            }
        }
    }

    /*-----------------------------------------------------------------------------
     * copy A to the decive
     *-----------------------------------------------------------------------------*/
    hipMalloc((void **)&d_A, sizeof(*d_A) * n * n);
    hipMemcpy(d_A, A, sizeof(*A) * n * n, hipMemcpyHostToDevice);

    /*-----------------------------------------------------------------------------
     * allocate expmA on the device
     *-----------------------------------------------------------------------------*/
    hipMalloc((void **)&d_expmA, sizeof(*d_expmA) * n * n);

    /*-----------------------------------------------------------------------------
     * workspace query and allocate memory buffer on the device and the host
     *-----------------------------------------------------------------------------*/
    size_t d_bufferSize = 0, h_bufferSize = 0;
    ret = hipexpmd_bufferSize(n, &d_bufferSize, &h_bufferSize);
    if (ret) {
        fprintf(stderr, "hipexpmd_bufferSize failed with error %d\n", ret);
        fflush(stderr);
        return ret;
    }
    if (d_bufferSize > 0) {
        hipMalloc((void **)&d_buffer, d_bufferSize);
    }
    if (h_bufferSize > 0) {
        hipHostMalloc((void **)&h_buffer, h_bufferSize);
    }

    /*-----------------------------------------------------------------------------
     * compute the approximation of the matrix exponential of A and measure the time
     *-----------------------------------------------------------------------------*/
    auto t0 = std::chrono::high_resolution_clock::now();
    ret = hipexpmd(n, d_A, n, d_buffer, h_buffer, d_expmA, n);
    if (ret) {
        fprintf(stderr, "hipexpmd failed with error %d\n", ret);
        fflush(stderr);
        return ret;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double wtime = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    /*-----------------------------------------------------------------------------
     * copy result to host and print the first 5x5 block
     *-----------------------------------------------------------------------------*/
    hipMemcpy(expmA, d_expmA, sizeof(*d_expmA) * n * n, hipMemcpyDeviceToHost);

    printf("expmA(1:5, 1:5) =\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%+e ", expmA[i + j * n]);
        }
        printf("\n");
    }
    printf("WallClockTime = %fs\n", wtime * 1e-9);

    /*-----------------------------------------------------------------------------
     * clear matrices A, expmA, d_A, and d_expmA and the device and host buffer
     *-----------------------------------------------------------------------------*/
    hipHostFree(A);
    hipHostFree(expmA);
    hipFree(d_A);
    hipFree(d_expmA);
    hipFree(d_buffer);
    hipHostFree(h_buffer);
    return 0;
}
