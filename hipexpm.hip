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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "hipexpm.h"

#define CHECK_HIPEXPM(err)                                                 \
    do {                                                                   \
        int error_code = (err);                                            \
        if (error_code) {                                                  \
            fprintf(stderr, "hipexpm Error %d. In file '%s' on line %d\n", \
                    error_code, __FILE__, __LINE__);                       \
            fflush(stderr);                                                \
            return -1;                                                     \
        }                                                                  \
    } while (false)

#define CHECK_HIP(err)                                                              \
    do {                                                                            \
        hipError_t error_code = (err);                                              \
        if (error_code != hipSuccess) {                                             \
            fprintf(stderr, "HIP Error %d: %s. In file '%s' on line %d\n",          \
                    error_code, hipGetErrorString(error_code), __FILE__, __LINE__); \
            fflush(stderr);                                                         \
            return -2;                                                              \
        }                                                                           \
    } while (false)

#define CHECK_HIPBLAS(err)                                                              \
    do {                                                                                \
        hipblasStatus_t error_code = (err);                                             \
        if (error_code != HIPBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "HIPBLAS Error %d - %s. In file '%s' on line %d\n",         \
                    error_code, hipblasStatusToString(error_code), __FILE__, __LINE__); \
            fflush(stderr);                                                             \
            return -3;                                                                  \
        }                                                                               \
    } while (false)

static inline const char *hipexpm_hipsolverGetErrorEnum(hipsolverStatus_t error) {
    switch (error) {
        case HIPSOLVER_STATUS_SUCCESS:
            return "HIPSOLVER_STATUS_SUCCESS";

        case HIPSOLVER_STATUS_ALLOC_FAILED:
            return "HIPSOLVER_STATUS_ALLOC_FAILED";

        case HIPSOLVER_STATUS_INVALID_VALUE:
            return "HIPSOLVER_STATUS_INVALID_VALUE";

        case HIPSOLVER_STATUS_ARCH_MISMATCH:
            return "HIPSOLVER_STATUS_ARCH_MISMATCH";

        case HIPSOLVER_STATUS_EXECUTION_FAILED:
            return "HIPSOLVER_STATUS_EXECUTION_FAILED";

        case HIPSOLVER_STATUS_INTERNAL_ERROR:
            return "HIPSOLVER_STATUS_INTERNAL_ERROR";
        default:
            return "unknown";
    }
}

#define CHECK_HIPSOLVER(err)                                                         \
    do {                                                                             \
        hipsolverStatus_t error_code = (err);                                        \
        if (error_code != HIPSOLVER_STATUS_SUCCESS) {                                \
            fprintf(stderr, "HIPSOLVER Error %d - %s. In file '%s' on line %d\n",    \
                    error_code, hipexpm_hipsolverGetErrorEnum(error_code), __FILE__, \
                    __LINE__);                                                       \
            fflush(stderr);                                                          \
            return -4;                                                               \
        }                                                                            \
    } while (false)

template <typename T>
struct hipexpm_traits;

template <>
struct hipexpm_traits<double> {
    typedef double S;
    typedef double hipBlasType;

    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr double one = 1.;
    static constexpr double mone = -1.;
    static constexpr double zero = 0.;
    static constexpr double ftwo = 2.;

    /*-----------------------------------------------------------------------------
     * Pade coefficients
     *-----------------------------------------------------------------------------*/
    static constexpr double Pade3[] = {120., 60., 12., 1.};
    static constexpr double Pade5[] = {30240., 15120., 3360., 420., 30., 1.};
    static constexpr double Pade7[] = {17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.};
    static constexpr double Pade9[] = {17643225600., 8821612800., 2075673600., 302702400., 30270240., 2162160., 110880., 3960., 90., 1.};
    static constexpr double Pade13[] = {64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800., 129060195264000., 10559470521600., 670442572800., 33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.};

    /*-----------------------------------------------------------------------------
     * absolute value, used for computing the 1-norm of a matrix
     *-----------------------------------------------------------------------------*/
    __device__ inline static double abs(const double x) {
        return fabs(x);
    }

    /*-----------------------------------------------------------------------------
     * hipBLAS
     *-----------------------------------------------------------------------------*/
    inline static hipblasStatus_t hipblasXdscal(hipblasHandle_t handle, int n, const double *alpha, double *x, int incx) {
        return hipblasDscal(handle, n, alpha, x, incx);
    }

    inline static hipblasStatus_t hipblasXgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const double *alpha, const double *AP, int lda, const double *BP, int ldb, const double *beta, double *CP, int ldc) {
        return hipblasDgemm(handle, transA, transB, m, n, k, alpha, AP, lda, BP, ldb, beta, CP, ldc);
    }

    inline static hipblasStatus_t hipblasXgeam(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, const double *alpha, const double *AP, int lda, const double *beta, const double *BP, int ldb, double *CP, int ldc) {
        return hipblasDgeam(handle, transA, transB, m, n, alpha, AP, lda, beta, BP, ldb, CP, ldc);
    }

    /*-----------------------------------------------------------------------------
     * hipSOLVER
     *-----------------------------------------------------------------------------*/
    inline static hipsolverStatus_t hipsolverDnXgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, double *A, int lda, int *Lwork) {
        return hipsolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
    }

    inline static hipsolverStatus_t hipsolverDnXgetrf(hipsolverHandle_t handle, int m, int n, double *A, int lda, void *work, int *devIpiv, int *info) {
        return hipsolverDnDgetrf(handle, m, n, A, lda, reinterpret_cast<double *>(work), devIpiv, info);
    }

    inline static hipsolverStatus_t hipsolverDnXgetrs(hipsolverHandle_t handle, hipblasOperation_t trans, int n, int nrhs, const double *A, int lda, int *devIpiv, double *B, int ldb, int *info) {
        return hipsolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, info);
    }
};

template <>
struct hipexpm_traits<float> {
    typedef float S;
    typedef float hipBlasType;

    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr float one = 1.f;
    static constexpr float mone = -1.f;
    static constexpr float zero = 0.f;
    static constexpr float ftwo = 2.f;

    /*-----------------------------------------------------------------------------
     * Pade coefficients
     *-----------------------------------------------------------------------------*/
    static constexpr float Pade3[] = {120.f, 60.f, 12.f, 1.f};
    static constexpr float Pade5[] = {30240.f, 15120.f, 3360.f, 420.f, 30.f, 1.f};
    static constexpr float Pade7[] = {17297280.f, 8648640.f, 1995840.f, 277200.f, 25200.f, 1512.f, 56.f, 1.f};
    static constexpr float Pade9[] = {17643225600.f, 8821612800.f, 2075673600.f, 302702400.f, 30270240.f, 2162160.f, 110880.f, 3960.f, 90.f, 1.f};
    static constexpr float Pade13[] = {64764752532480000.f, 32382376266240000.f, 7771770303897600.f, 1187353796428800.f, 129060195264000.f, 10559470521600.f, 670442572800.f, 33522128640.f, 1323241920.f, 40840800.f, 960960.f, 16380.f, 182.f, 1.f};

    /*-----------------------------------------------------------------------------
     * absolute value, used for computing the 1-norm of a matrix
     *-----------------------------------------------------------------------------*/
    __device__ inline static double abs(const double x) {
        return fabsf(x);
    }

    /*-----------------------------------------------------------------------------
     * hipBLAS
     *-----------------------------------------------------------------------------*/
    inline static hipblasStatus_t hipblasXdscal(hipblasHandle_t handle, int n, const float *alpha, float *x, int incx) {
        return hipblasSscal(handle, n, alpha, x, incx);
    }

    inline static hipblasStatus_t hipblasXgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const float *alpha, const float *AP, int lda, const float *BP, int ldb, const float *beta, float *CP, int ldc) {
        return hipblasSgemm(handle, transA, transB, m, n, k, alpha, AP, lda, BP, ldb, beta, CP, ldc);
    }

    inline static hipblasStatus_t hipblasXgeam(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, const float *alpha, const float *AP, int lda, const float *beta, const float *BP, int ldb, float *CP, int ldc) {
        return hipblasSgeam(handle, transA, transB, m, n, alpha, AP, lda, beta, BP, ldb, CP, ldc);
    }

    /*-----------------------------------------------------------------------------
     * hipSOLVER
     *-----------------------------------------------------------------------------*/
    inline static hipsolverStatus_t hipsolverDnXgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, float *A, int lda, int *Lwork) {
        return hipsolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
    }

    inline static hipsolverStatus_t hipsolverDnXgetrf(hipsolverHandle_t handle, int m, int n, float *A, int lda, void *work, int *devIpiv, int *info) {
        return hipsolverDnSgetrf(handle, m, n, A, lda, reinterpret_cast<float *>(work), devIpiv, info);
    }

    inline static hipsolverStatus_t hipsolverDnXgetrs(hipsolverHandle_t handle, hipblasOperation_t trans, int n, int nrhs, const float *A, int lda, int *devIpiv, float *B, int ldb, int *info) {
        return hipsolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, info);
    }
};

template <>
struct hipexpm_traits<hipDoubleComplex> {
    typedef double S;
    typedef hipblasDoubleComplex hipBlasType;

    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr hipDoubleComplex one = {1., 0.};
    static constexpr hipDoubleComplex mone = {-1., 0.};
    static constexpr hipDoubleComplex zero = {0., 0.};
    static constexpr double ftwo = 2.;

    /*-----------------------------------------------------------------------------
     * Pade coefficients
     *-----------------------------------------------------------------------------*/
    static constexpr hipDoubleComplex Pade3[] = {{120., 0.}, {60., 0.}, {12., 0.}, {1., 0.}};
    static constexpr hipDoubleComplex Pade5[] = {{30240., 0.}, {15120., 0.}, {3360., 0.}, {420., 0.}, {30., 0.}, {1., 0.}};
    static constexpr hipDoubleComplex Pade7[] = {{17297280., 0.}, {8648640., 0.}, {1995840., 0.}, {277200., 0.}, {25200., 0.}, {1512., 0.}, {56., 0.}, {1., 0.}};
    static constexpr hipDoubleComplex Pade9[] = {{17643225600., 0.}, {8821612800., 0.}, {2075673600., 0.}, {302702400., 0.}, {30270240., 0.}, {2162160., 0.}, {110880., 0.}, {3960., 0.}, {90., 0.}, {1., 0.}};
    static constexpr hipDoubleComplex Pade13[] = {{64764752532480000., 0.}, {32382376266240000., 0.}, {7771770303897600., 0.}, {1187353796428800., 0.}, {129060195264000., 0.}, {10559470521600., 0.}, {670442572800., 0.}, {33522128640., 0.}, {1323241920., 0.}, {40840800., 0.}, {960960., 0.}, {16380., 0.}, {182., 0.}, {1., 0.}};

    /*-----------------------------------------------------------------------------
     * absolute value, used for computing the 1-norm of a matrix
     *-----------------------------------------------------------------------------*/
    __device__ inline static double abs(const hipDoubleComplex x) {
        return hipCabs(x);
    }

    /*-----------------------------------------------------------------------------
     * hipBLAS
     *-----------------------------------------------------------------------------*/
    inline static hipblasStatus_t hipblasXdscal(hipblasHandle_t handle, int n, const double *alpha, hipDoubleComplex *x, int incx) {
        return hipblasZdscal(handle, n, alpha, reinterpret_cast<hipblasDoubleComplex *>(x), incx);
    }

    inline static hipblasStatus_t hipblasXgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipDoubleComplex *alpha, const hipDoubleComplex *AP, int lda, const hipDoubleComplex *BP, int ldb, const hipDoubleComplex *beta, hipDoubleComplex *CP, int ldc) {
        return hipblasZgemm(handle, transA, transB, m, n, k, reinterpret_cast<const hipblasDoubleComplex *>(alpha), reinterpret_cast<const hipblasDoubleComplex *>(AP), lda, reinterpret_cast<const hipblasDoubleComplex *>(BP), ldb, reinterpret_cast<const hipblasDoubleComplex *>(beta), reinterpret_cast<hipblasDoubleComplex *>(CP), ldc);
    }

    inline static hipblasStatus_t hipblasXgeam(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, const hipDoubleComplex *alpha, const hipDoubleComplex *AP, int lda, const hipDoubleComplex *beta, const hipDoubleComplex *BP, int ldb, hipDoubleComplex *CP, int ldc) {
        return hipblasZgeam(handle, transA, transB, m, n, reinterpret_cast<const hipblasDoubleComplex *>(alpha), reinterpret_cast<const hipblasDoubleComplex *>(AP), lda, reinterpret_cast<const hipblasDoubleComplex *>(beta), reinterpret_cast<const hipblasDoubleComplex *>(BP), ldb, reinterpret_cast<hipblasDoubleComplex *>(CP), ldc);
    }

    /*-----------------------------------------------------------------------------
     * hipSOLVER
     *-----------------------------------------------------------------------------*/
    inline static hipsolverStatus_t hipsolverDnXgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, hipDoubleComplex *A, int lda, int *Lwork) {
        return hipsolverDnZgetrf_bufferSize(handle, m, n, A, lda, Lwork);
    }

    inline static hipsolverStatus_t hipsolverDnXgetrf(hipsolverHandle_t handle, int m, int n, hipDoubleComplex *A, int lda, void *work, int *devIpiv, int *info) {
        return hipsolverDnZgetrf(handle, m, n, A, lda, reinterpret_cast<hipDoubleComplex *>(work), devIpiv, info);
    }

    inline static hipsolverStatus_t hipsolverDnXgetrs(hipsolverHandle_t handle, hipblasOperation_t trans, int n, int nrhs, const hipDoubleComplex *A, int lda, int *devIpiv, hipDoubleComplex *B, int ldb, int *info) {
        return hipsolverDnZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, info);
    }
};

template <>
struct hipexpm_traits<hipComplex> {
    typedef float S;
    typedef hipblasComplex hipBlasType;

    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    static constexpr hipComplex one = {1.f, 0.f};
    static constexpr hipComplex mone = {-1.f, 0.f};
    static constexpr hipComplex zero = {0.f, 0.f};
    static constexpr float ftwo = 2.f;

    /*-----------------------------------------------------------------------------
     * Pade coefficients
     *-----------------------------------------------------------------------------*/
    static constexpr hipComplex Pade3[] = {{120.f, 0.f}, {60.f, 0.f}, {12.f, 0.f}, {1.f, 0.f}};
    static constexpr hipComplex Pade5[] = {{30240.f, 0.f}, {15120.f, 0.f}, {3360.f, 0.f}, {420.f, 0.f}, {30.f, 0.f}, {1.f, 0.f}};
    static constexpr hipComplex Pade7[] = {{17297280.f, 0.f}, {8648640.f, 0.f}, {1995840.f, 0.f}, {277200.f, 0.f}, {25200.f, 0.f}, {1512.f, 0.f}, {56.f, 0.f}, {1.f, 0.f}};
    static constexpr hipComplex Pade9[] = {{17643225600.f, 0.f}, {8821612800.f, 0.f}, {2075673600.f, 0.f}, {302702400.f, 0.f}, {30270240.f, 0.f}, {2162160.f, 0.f}, {110880.f, 0.f}, {3960.f, 0.f}, {90.f, 0.f}, {1.f, 0.f}};
    static constexpr hipComplex Pade13[] = {{64764752532480000.f, 0.f}, {32382376266240000.f, 0.f}, {7771770303897600.f, 0.f}, {1187353796428800.f, 0.f}, {129060195264000.f, 0.f}, {10559470521600.f, 0.f}, {670442572800.f, 0.f}, {33522128640.f, 0.f}, {1323241920.f, 0.f}, {40840800.f, 0.f}, {960960.f, 0.f}, {16380.f, 0.f}, {182.f, 0.f}, {1.f, 0.f}};

    /*-----------------------------------------------------------------------------
     * absolute value, used for computing the 1-norm of a matrix
     *-----------------------------------------------------------------------------*/
    __device__ inline static float abs(const hipComplex x) {
        return hipCabsf(x);
    }

    /*-----------------------------------------------------------------------------
     * hipBLAS
     *-----------------------------------------------------------------------------*/
    inline static hipblasStatus_t hipblasXdscal(hipblasHandle_t handle, int n, const float *alpha, hipComplex *x, int incx) {
        return hipblasCsscal(handle, n, alpha, reinterpret_cast<hipblasComplex *>(x), incx);
    }

    inline static hipblasStatus_t hipblasXgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipComplex *alpha, const hipComplex *AP, int lda, const hipComplex *BP, int ldb, const hipComplex *beta, hipComplex *CP, int ldc) {
        return hipblasCgemm(handle, transA, transB, m, n, k, reinterpret_cast<const hipblasComplex *>(alpha), reinterpret_cast<const hipblasComplex *>(AP), lda, reinterpret_cast<const hipblasComplex *>(BP), ldb, reinterpret_cast<const hipblasComplex *>(beta), reinterpret_cast<hipblasComplex *>(CP), ldc);
    }

    inline static hipblasStatus_t hipblasXgeam(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, const hipComplex *alpha, const hipComplex *AP, int lda, const hipComplex *beta, const hipComplex *BP, int ldb, hipComplex *CP, int ldc) {
        return hipblasCgeam(handle, transA, transB, m, n, reinterpret_cast<const hipblasComplex *>(alpha), reinterpret_cast<const hipblasComplex *>(AP), lda, reinterpret_cast<const hipblasComplex *>(beta), reinterpret_cast<const hipblasComplex *>(BP), ldb, reinterpret_cast<hipblasComplex *>(CP), ldc);
    }

    /*-----------------------------------------------------------------------------
     * hipSOLVER
     *-----------------------------------------------------------------------------*/
    inline static hipsolverStatus_t hipsolverDnXgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, hipComplex *A, int lda, int *Lwork) {
        return hipsolverDnCgetrf_bufferSize(handle, m, n, A, lda, Lwork);
    }

    inline static hipsolverStatus_t hipsolverDnXgetrf(hipsolverHandle_t handle, int m, int n, hipComplex *A, int lda, void *work, int *devIpiv, int *info) {
        return hipsolverDnCgetrf(handle, m, n, A, lda, reinterpret_cast<hipComplex *>(work), devIpiv, info);
    }

    inline static hipsolverStatus_t hipsolverDnXgetrs(hipsolverHandle_t handle, hipblasOperation_t trans, int n, int nrhs, const hipComplex *A, int lda, int *devIpiv, hipComplex *B, int ldb, int *info) {
        return hipsolverDnCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, info);
    }
};

template <typename T>
__global__ static void hipexpm_absrowsums(const int n, const T *__restrict__ d_A, const int ldA, double *__restrict__ buffer) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = i; j < n; j += blockDim.x * gridDim.x) {
        double tmp = 0.;
        for (int k = 0; k < n; ++k) {
            tmp += hipexpm_traits<T>::abs(d_A[k + j * ldA]);
        }
        buffer[j] = tmp;
    }
}

template <typename T>
static int hipexpm_matrix1norm(const int n, const T *__restrict__ d_A, const int ldA, void *d_buffer, double *__restrict__ d_nrmA1) {
    *d_nrmA1 = 0.;
    double *buffer = reinterpret_cast<double *>(d_buffer);
    hipexpm_absrowsums<<<(n + 255) / 256, 256>>>(n, d_A, ldA, buffer);
    CHECK_HIP(hipPeekAtLastError());
    CHECK_HIP(hipDeviceSynchronize());
    *d_nrmA1 = *(thrust::max_element(thrust::device_pointer_cast(buffer), thrust::device_pointer_cast(buffer + n)));
    return 0;
}

template <typename T>
static int hipexpm_parameters(const int n, const T *__restrict__ d_A, const int ldA, void *d_buffer, int *m, int *s) {
    double eta1 = 0.;
    CHECK_HIPEXPM(hipexpm_matrix1norm(n, d_A, ldA, d_buffer, &eta1));
    if constexpr (std::is_same<T, double>::value || std::is_same<T, hipDoubleComplex>::value) {
        const double theta[] = {1.495585217958292e-002, 2.539398330063230e-001, 9.504178996162932e-001, 2.97847961257068e+000, 5.371920351148152e+000};
        *s = 0;
        if (eta1 <= theta[0]) {
            *m = 3;
            return 0;
        }
        if (eta1 <= theta[1]) {
            *m = 5;
            return 0;
        }
        if (eta1 <= theta[2]) {
            *m = 7;
            return 0;
        }
        if (eta1 <= theta[3]) {
            *m = 9;
            return 0;
        }
        *s = ceil(log2(eta1 / theta[4]));
        if (*s < 0) {
            *s = 0;
        }
        *m = 13;
    } else {
        const double theta[] = {4.258730016922831e-001, 1.880152677804762e+000, 3.925724783138660e+000};
        *s = 0;
        if (eta1 <= theta[0]) {
            *m = 3;
            return 0;
        }
        if (eta1 <= theta[1]) {
            *m = 5;
            return 0;
        }
        *s = ceil(log2(eta1 / theta[2]));
        if (*s < 0) {
            *s = 0;
        }
        *m = 7;
    }
    return 0;
}

template <typename T>
__global__ static void setDiag(const int n, T *d_A, const int ldA, const T alpha) {
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int j0 = threadIdx.y + blockIdx.y * blockDim.y;
    for (int i = i0; i < n; i += blockDim.x * gridDim.x) {
        for (int j = j0; j < n; j += blockDim.y * gridDim.y) {
            if (i == j) {
                d_A[i + j * ldA] = alpha;
            } else {
                d_A[i + j * ldA] = hipexpm_traits<T>::zero;
            }
        }
    }
}

template <typename T>
__global__ static void addDiag(const int n, T *d_A, const int ldA, const T alpha) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = i; j < n; j += blockDim.x * gridDim.x) {
        if constexpr (std::is_same<T, hipComplex>::value || std::is_same<T, hipDoubleComplex>::value) {
            d_A[j + j * ldA].x += alpha.x;
            d_A[j + j * ldA].y += alpha.y;
        } else {
            d_A[j + j * ldA] += alpha;
        }
    }
}

template <typename T>
static int hipexpm_bufferSize(const int n, size_t *d_bufferSize) {
    /*-----------------------------------------------------------------------------
     * initialize with zero
     *-----------------------------------------------------------------------------*/
    int bufferSize = 0;
    *d_bufferSize = 0;

    /*-----------------------------------------------------------------------------
     * get device and host workspace size for LU factorization
     *-----------------------------------------------------------------------------*/
    // create hipsolver handle
    hipsolverDnHandle_t hipsolverH;
    CHECK_HIPSOLVER(hipsolverDnCreate(&hipsolverH));

    // compute workspace size
    CHECK_HIPSOLVER(hipexpm_traits<T>::hipsolverDnXgetrf_bufferSize(hipsolverH, n, n, nullptr, n, &bufferSize));

    // free workspace
    CHECK_HIPSOLVER(hipsolverDnDestroy(hipsolverH));

    /*-----------------------------------------------------------------------------
     * compute final workspace size
     * matrix T1, T2, T4, T6, T8, U, V -> n * n * 5 +  n * n * 2
     * int64 array ipiv -> n * sizeof(int64_t)
     * int info -> sizeof(int)
     *-----------------------------------------------------------------------------*/
    *d_bufferSize = bufferSize;
    *d_bufferSize = std::max(*d_bufferSize, sizeof(T) * n * n * 5) + sizeof(T) * n * n * 2 + sizeof(int64_t) * n + sizeof(int);

    return 0;
}

int hipexpmd_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    *h_bufferSize = 0;
    return hipexpm_bufferSize<double>(n, d_bufferSize);
}

int hipexpms_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    *h_bufferSize = 0;
    return hipexpm_bufferSize<float>(n, d_bufferSize);
}

int hipexpmc_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    *h_bufferSize = 0;
    return hipexpm_bufferSize<hipComplex>(n, d_bufferSize);
}

int hipexpmz_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize) {
    *h_bufferSize = 0;
    return hipexpm_bufferSize<hipDoubleComplex>(n, d_bufferSize);
}

template <typename T>
static int hipexpm(const int n, const T *d_A, const int ldA, void *d_buffer, void *h_buffer, T *d_F, const int ldF) {
    /*-----------------------------------------------------------------------------
     * kernel launch parameters
     *-----------------------------------------------------------------------------*/
    const size_t threadsPerBlock = 256;
    const size_t blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    /*-----------------------------------------------------------------------------
     * compute the scaling parameter and Pade approximant degree
     *-----------------------------------------------------------------------------*/
    int m, s;
    CHECK_HIPEXPM(hipexpm_parameters(n, d_A, ldA, d_buffer, &m, &s));
    // printf("m = %d, s = %d\n", m, s);

    /*-----------------------------------------------------------------------------
     * split memory buffer
     * memory layout: |U, V, T1, T2, T4, T6, T8| from 0 to (n * n * 7) -1
     *-----------------------------------------------------------------------------*/
    T *T1, *T2, *T4, *T6, *T8, *U, *V;
    U = (T *)d_buffer;
    V = U + n * n * 1;
    T1 = U + n * n * 2;
    T2 = U + n * n * 3;
    T4 = U + n * n * 4;
    T6 = U + n * n * 5;
    T8 = U + n * n * 6;

    /*-----------------------------------------------------------------------------
     * create hipBlas handle
     *-----------------------------------------------------------------------------*/
    hipblasHandle_t hipblasH;
    CHECK_HIPBLAS(hipblasCreate(&hipblasH));

    /*-----------------------------------------------------------------------------
     * rescale T, T = T / 2^s
     *-----------------------------------------------------------------------------*/
    CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, d_A, ldA, &hipexpm_traits<T>::zero, T1, n, T1, n));
    typename hipexpm_traits<T>::S alpha = 1. / (1 << s);
    if (s != 0) {
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXdscal(hipblasH, n * n, &alpha, T1, 1));
    }

    /*-----------------------------------------------------------------------------
     * compute powers of T if needed
     *-----------------------------------------------------------------------------*/
    CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgemm(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &hipexpm_traits<T>::one, T1, n, T1, n, &hipexpm_traits<T>::zero, T2, n));
    if (m >= 5) {
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgemm(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &hipexpm_traits<T>::one, T2, n, T2, n, &hipexpm_traits<T>::zero, T4, n));
    }
    if (m >= 7) {
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgemm(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &hipexpm_traits<T>::one, T2, n, T4, n, &hipexpm_traits<T>::zero, T6, n));
    }
    if (m == 9) {
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgemm(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &hipexpm_traits<T>::one, T4, n, T4, n, &hipexpm_traits<T>::zero, T8, n));
    }

    /*-----------------------------------------------------------------------------
     * compute U and V for the Pade approximant independently on different streams
     *-----------------------------------------------------------------------------*/
    CHECK_HIP(hipDeviceSynchronize());
    hipStream_t streamU, streamV;
    CHECK_HIP(hipStreamCreate(&streamU));
    CHECK_HIP(hipStreamCreate(&streamV));
    if (m == 3) {
        // U = U + c(3)*T2 + c(1)*I
        setDiag<<<blocksPerGrid, threadsPerBlock, 0, streamU>>>(n, U, n, hipexpm_traits<T>::Pade3[1]);
        CHECK_HIP(hipPeekAtLastError());
        CHECK_HIPBLAS(hipblasSetStream(hipblasH, streamU));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, U, n, &hipexpm_traits<T>::Pade3[3], T2, n, U, n));

        // V = V + c(2)*T2 + c(0)*I
        setDiag<<<blocksPerGrid, threadsPerBlock, 0, streamV>>>(n, V, n, hipexpm_traits<T>::Pade3[0]);
        CHECK_HIP(hipPeekAtLastError());
        CHECK_HIPBLAS(hipblasSetStream(hipblasH, streamV));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, V, n, &hipexpm_traits<T>::Pade3[2], T2, n, V, n));
    } else if (m == 5) {
        // U = U + c(5)*T4 + c(3)*T2 + c(1)*I
        setDiag<<<blocksPerGrid, threadsPerBlock, 0, streamU>>>(n, U, n, hipexpm_traits<T>::Pade5[1]);
        CHECK_HIP(hipPeekAtLastError());
        CHECK_HIPBLAS(hipblasSetStream(hipblasH, streamU));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, U, n, &hipexpm_traits<T>::Pade5[3], T2, n, U, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, U, n, &hipexpm_traits<T>::Pade5[5], T4, n, U, n));

        // V = V + c(4)*T4 + c(2)*T2 + c(0)*I
        setDiag<<<blocksPerGrid, threadsPerBlock, 0, streamV>>>(n, V, n, hipexpm_traits<T>::Pade5[0]);
        CHECK_HIP(hipPeekAtLastError());
        CHECK_HIPBLAS(hipblasSetStream(hipblasH, streamV));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, V, n, &hipexpm_traits<T>::Pade5[2], T2, n, V, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, V, n, &hipexpm_traits<T>::Pade5[4], T4, n, V, n));
    } else if (m == 7) {
        // U = U + c(7)*T6 + c(5)*T4 + c(3)*T2 + c(1)*I
        setDiag<<<blocksPerGrid, threadsPerBlock, 0, streamU>>>(n, U, n, hipexpm_traits<T>::Pade7[1]);
        CHECK_HIP(hipPeekAtLastError());
        CHECK_HIPBLAS(hipblasSetStream(hipblasH, streamU));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, U, n, &hipexpm_traits<T>::Pade7[3], T2, n, U, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, U, n, &hipexpm_traits<T>::Pade7[5], T4, n, U, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, U, n, &hipexpm_traits<T>::Pade7[7], T6, n, U, n));

        // V = V + c(6)*T6 + c(4)*T4 + c(2)*T2 + c(0)*I
        setDiag<<<blocksPerGrid, threadsPerBlock, 0, streamV>>>(n, V, n, hipexpm_traits<T>::Pade7[0]);
        CHECK_HIP(hipPeekAtLastError());
        CHECK_HIPBLAS(hipblasSetStream(hipblasH, streamV));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, V, n, &hipexpm_traits<T>::Pade7[2], T2, n, V, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, V, n, &hipexpm_traits<T>::Pade7[4], T4, n, V, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, V, n, &hipexpm_traits<T>::Pade7[6], T6, n, V, n));
    } else if (m == 9) {
        // U = U + c(9)*T8 + c(7)*T6 + c(5)*T4 + c(3)*T2 + c(1)*I
        setDiag<<<blocksPerGrid, threadsPerBlock, 0, streamU>>>(n, U, n, hipexpm_traits<T>::Pade9[1]);
        CHECK_HIP(hipPeekAtLastError());
        CHECK_HIPBLAS(hipblasSetStream(hipblasH, streamU));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, U, n, &hipexpm_traits<T>::Pade9[3], T2, n, U, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, U, n, &hipexpm_traits<T>::Pade9[5], T4, n, U, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, U, n, &hipexpm_traits<T>::Pade9[7], T6, n, U, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, U, n, &hipexpm_traits<T>::Pade9[9], T8, n, U, n));

        // V = V + c(6)*T6 + c(4)*T4 + c(2)*T2 + c(0)*I
        setDiag<<<blocksPerGrid, threadsPerBlock, 0, streamV>>>(n, V, n, hipexpm_traits<T>::Pade9[0]);
        CHECK_HIP(hipPeekAtLastError());
        CHECK_HIPBLAS(hipblasSetStream(hipblasH, streamV));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, V, n, &hipexpm_traits<T>::Pade9[2], T2, n, V, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, V, n, &hipexpm_traits<T>::Pade9[4], T4, n, V, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, V, n, &hipexpm_traits<T>::Pade9[6], T6, n, V, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, V, n, &hipexpm_traits<T>::Pade9[8], T8, n, V, n));
    } else if (m == 13) {
        dim3 grid((n + 15) / 16, (n + 15) / 16);
        //  U = T6*(c(13)*T6 + c(11)*T4 + c(9)*T2) + c(7)*T6 + c(5)*T4 + c(3)*T2 + c(1)*I;
        setDiag<<<grid, dim3(16, 16), 0, streamU>>>(n, U, n, hipexpm_traits<T>::Pade13[1]);
        CHECK_HIP(hipPeekAtLastError());
        CHECK_HIPBLAS(hipblasSetStream(hipblasH, streamU));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, U, n, &hipexpm_traits<T>::Pade13[3], T2, n, U, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, U, n, &hipexpm_traits<T>::Pade13[5], T4, n, U, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, U, n, &hipexpm_traits<T>::Pade13[7], T6, n, U, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::Pade13[9], T2, n, &hipexpm_traits<T>::Pade13[11], T4, n, T8, n));  // overwrite of T8
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::Pade13[13], T6, n, &hipexpm_traits<T>::one, T8, n, T8, n));        // overwrite of T8
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgemm(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &hipexpm_traits<T>::one, T6, n, T8, n, &hipexpm_traits<T>::one, U, n));

        // V = T6*(c(12)*T6 + c(10)*T4 + c(8)*T2) + c(6)*T6 + c(4)*T4 + c(2)*T2 + c(0)*I;
        setDiag<<<grid, dim3(16, 16), 0, streamV>>>(n, V, n, hipexpm_traits<T>::Pade13[0]);
        CHECK_HIP(hipPeekAtLastError());
        CHECK_HIPBLAS(hipblasSetStream(hipblasH, streamV));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, V, n, &hipexpm_traits<T>::Pade13[2], T2, n, V, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, V, n, &hipexpm_traits<T>::Pade13[4], T4, n, V, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::one, V, n, &hipexpm_traits<T>::Pade13[6], T6, n, V, n));
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::Pade13[8], T2, n, &hipexpm_traits<T>::Pade13[10], T4, n, T8, n));  // overwrite of T8
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::Pade13[12], T6, n, &hipexpm_traits<T>::one, T8, n, T8, n));        // overwrite of T8
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgemm(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &hipexpm_traits<T>::one, T6, n, T8, n, &hipexpm_traits<T>::one, V, n));
    } else {
        fprintf(stderr, "m must be 3, 5, 7, 9, or 13\n");
        fflush(stderr);
        return -1;
    }
    CHECK_HIP(hipStreamSynchronize(streamU));
    CHECK_HIP(hipStreamSynchronize(streamV));
    CHECK_HIP(hipStreamDestroy(streamU));
    CHECK_HIP(hipStreamDestroy(streamV));
    CHECK_HIPBLAS(hipblasSetStream(hipblasH, 0));

    // U = T*U
    CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgemm(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &hipexpm_traits<T>::one, T1, n, U, n, &hipexpm_traits<T>::zero, T8, n));
    std::swap(U, T8);

    /*-----------------------------------------------------------------------------
     *  compute F = (V-U)\(U+V) = (V-U)\2*U + I
     *-----------------------------------------------------------------------------*/
    // prepare right-hand side
    CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgeam(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, &hipexpm_traits<T>::mone, U, n, &hipexpm_traits<T>::one, V, n, V, n));

    typename hipexpm_traits<T>::S two = 2.;
    CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXdscal(hipblasH, n * n, &two, U, 1));

    // create hipsolver handle
    hipsolverDnHandle_t hipsolverH;
    CHECK_HIPSOLVER(hipsolverDnCreate(&hipsolverH));

    // split the memory buffer
    // memory layout: |U, V, T1, T2, T4, T6, T8| from 0 to (n * n * 7) -1
    //                |x, x, ipiv, info, dwork|  from n * n * 2         to n * n * 2 + n -1 (ipiv) overwrites T1
    //                                           from n * n * 2 + n     to n * n * 2 + n    (info) overwrites T1, T2
    //                                           from n * n * 2 + n + 1 to XXXXXXXXXXXXX    (dwork) overwrites T1, T2, T4, T6, T8, ...
    int *d_ipiv = (int *)T1;            // use T1 as ipiv
    int *d_info = (int *)(d_ipiv + n);  // put d_info after d_ipiv
    void *d_work = d_info + 1;          // put d_work after d_info

    // compute LU factorization
    CHECK_HIPSOLVER(hipexpm_traits<T>::hipsolverDnXgetrf(hipsolverH, n, n, V, n, d_work, d_ipiv, d_info));

    // solve linear system
    CHECK_HIPSOLVER(hipexpm_traits<T>::hipsolverDnXgetrs(hipsolverH, HIPBLAS_OP_N, n, n, V, n, d_ipiv, U, n, d_info));

    // free workspace
    CHECK_HIPSOLVER(hipsolverDnDestroy(hipsolverH));

    // add identity
    addDiag<<<blocksPerGrid, threadsPerBlock>>>(n, U, n, hipexpm_traits<T>::one);
    CHECK_HIP(hipPeekAtLastError());

    /*-----------------------------------------------------------------------------
     * squaring phase
     *-----------------------------------------------------------------------------*/
    if (((s % 2) == 0) && s > 0) {
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgemm(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &hipexpm_traits<T>::one, U, n, U, n, &hipexpm_traits<T>::zero, V, n));
        std::swap(U, V);
        s--;
    }

    int ldU = n;
    int ldF_ = ldF;
    for (int k = 0; k < s; ++k) {
        CHECK_HIPBLAS(hipexpm_traits<T>::hipblasXgemm(hipblasH, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &hipexpm_traits<T>::one, U, ldU, U, ldU, &hipexpm_traits<T>::zero, d_F, ldF_));
        std::swap(U, d_F);
        std::swap(ldU, ldF_);
    }

    CHECK_HIP(hipDeviceSynchronize());

    /*-----------------------------------------------------------------------------
     * free memory and destroy hipblas handle
     *-----------------------------------------------------------------------------*/
    CHECK_HIPBLAS(hipblasDestroy(hipblasH));
    return 0;
}

int hipexpms(const int n, const float *d_A, const int ldA, void *d_buffer, void *h_buffer, float *d_expmA, const int ldexpmA) {
    return hipexpm(n, d_A, ldA, d_buffer, h_buffer, d_expmA, ldexpmA);
}

int hipexpmd(const int n, const double *d_A, const int ldA, void *d_buffer, void *h_buffer, double *d_expmA, const int ldexpmA) {
    return hipexpm(n, d_A, ldA, d_buffer, h_buffer, d_expmA, ldexpmA);
}

int hipexpmc(const int n, const hipComplex *d_A, const int ldA, void *d_buffer, void *h_buffer, hipComplex *d_expmA, const int ldexpmA) {
    return hipexpm(n, d_A, ldA, d_buffer, h_buffer, d_expmA, ldexpmA);
}

int hipexpmz(const int n, const hipDoubleComplex *d_A, const int ldA, void *d_buffer, void *h_buffer, hipDoubleComplex *d_expmA, const int ldexpmA) {
    return hipexpm(n, d_A, ldA, d_buffer, h_buffer, d_expmA, ldexpmA);
}
