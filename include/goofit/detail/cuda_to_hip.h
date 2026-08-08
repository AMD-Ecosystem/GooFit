#pragma once

// CUDA-to-HIP compatibility shim for the ROCm/HIP build of GooFit.
//
// This header is force-included (-include) on every translation unit when
// GooFit is built with GOOFIT_DEVICE=HIP. GooFit's device code is written
// against the CUDA runtime API and the CUDA spelling of __device__ globals
// and device function pointers; HIP provides the same model with a hipXxx
// runtime, so a thin symbol map lets the existing sources compile unchanged.
//
// rocThrust selects THRUST_DEVICE_SYSTEM_HIP automatically under hipcc, so we
// do NOT override THRUST_DEVICE_SYSTEM. Instead GooFit's GPU-path checks use
// GOOFIT_DEVICE_IS_GPU (see CudaCompat.h), which is true for both the CUDA and
// HIP Thrust systems.

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

// Runtime API: 1:1 with the CUDA equivalents.
#define cudaError hipError_t
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaGetErrorString hipGetErrorString

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost

#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaSetDevice hipSetDevice
#define cudaDeviceProp hipDeviceProp_t

#define cudaDeviceSetLimit hipDeviceSetLimit
#define cudaDeviceGetLimit hipDeviceGetLimit
#define cudaLimitStackSize hipLimitStackSize

#define cudaStream_t hipStream_t

#endif // HIP
