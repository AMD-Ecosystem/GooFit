#pragma once

// GooFit's GPU device path was historically gated on
//   THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
// which is only true for the NVIDIA/CUDA Thrust backend. rocThrust runs the
// same Thrust API on AMD GPUs but reports THRUST_DEVICE_SYSTEM_HIP, so that
// check would otherwise route the ROCm build onto the CPU fallback path.
//
// GOOFIT_DEVICE_IS_GPU is true for either GPU Thrust backend (CUDA or HIP);
// the GPU-specific code paths key off it instead of the CUDA-only constant.

#include <thrust/detail/config/device_system.h>

#if(THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA) || (THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP)
#define GOOFIT_DEVICE_IS_GPU 1
#else
#define GOOFIT_DEVICE_IS_GPU 0
#endif
