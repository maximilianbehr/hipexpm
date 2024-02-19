# hipexpm - Matrix Exponential Approximation using HIP

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![GitHub Release](https://img.shields.io/github/v/release/maximilianbehr/hipexpm?display_name=release&style=flat)
 ![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/maximilianbehr/hipexpm/total)

**Version:** 1.0.0

**Copyright:** Maximilian Behr

**License:** The software is licensed under under MIT. See [`LICENSE`](LICENSE) for details.

`hipexpm` is a `HIP` library for the numerical approximation of the matrix exponential $e^A$.

`hipexpm` support single and double precision as well as real and complex matrices.


| Functions                          | Data                             |
| -----------------------------------|----------------------------------|
| `hipexpms_bufferSize`, `hipexpms`  | real, single precision matrix    |
| `hipexpmd_bufferSize`, `hipexpmd`  | real, double precision matrix    |
| `hipexpmc_bufferSize`, `hipexpmc`  | complex, single precision matrix |
| `hipexpmz_bufferSize`, `hipexpmz`  | complex, double precision matrix |


Available functions:

```C
int hipexpms_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int hipexpms(const float *d_A, const int n, void *d_buffer, void *h_buffer, float *d_expmA);
```
```C
int hipexpmd_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int hipexpmd(const double *d_A, const int n, void *d_buffer, void *h_buffer, double *d_expmA);
```
```C
int hipexpmc_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int hipexpmc(const hipComplex *d_A, const int n, void *d_buffer, void *h_buffer, hipComplex *d_expmA);
```
```C
int hipexpmz_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int hipexpmz(const hipDoubleComplex *d_A, const int n, void *d_buffer, void *h_buffer, hipDoubleComplex *d_expmA);
```

## Algorithm

`hipexpm` implements the scaling and squaring method for the matrix exponential approximation. 

> The Scaling and Squaring Method for the Matrix Exponential Revisited
Nicholas J. Higham
SIAM Journal on Matrix Analysis and Applications 2005 26:4, 1179-1193 


## Installation

Prerequisites:
 * `CMake >= 3.23`
 * `ROCm >= 6.0.2`

```shell
  mkdir build && cd build
  cmake ..
  make
  make install
```

## Usage and Examples

We provide examples for all supported matrix formats:

  
| File                                           | Data                             |
| -----------------------------------------------|----------------------------------|
| [`example_hipexpms.hip`](example_hipexpms.hip) | real, single precision matrix    |
| [`example_hipexpmd.hip`](example_hipexpmd.hip) | real, double precision matrix    |
| [`example_hipexpmc.hip`](example_hipexpmc.hip) | complex, single precision matrix |
| [`example_hipexpmz.hip`](example_hipexpmz.hip) | complex, double precision matrix |

