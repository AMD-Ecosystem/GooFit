#pragma once

#include <goofit/GlobalCudaDefines.h>

#include <vector>

#if GOOFIT_DEVICE_IS_GPU
#include "SmartVectorGPU.h"
#else
#include "SmartVectorCPU.h"
#endif
