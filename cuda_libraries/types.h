#pragma once

#include <assert.h>
#include <float.h>

#include "GPUPair.cu"
#include <math_constants.h>
//#include "../geometry/Ray.cu"
//#include "../shapes/Box.cu"

#include <cmath>
#include <iostream>
#include <execinfo.h>
#include <cstdlib>
#include <cxxabi.h>

#include <atomic>


std::atomic<int> triangle_meshes_ids = -1;

#define Float float
#define MachineEpsilon (FLT_EPSILON * 0.5)  // Change for double precission

#define PI CUDART_PI_F
constexpr Float INV_PI = (1/PI);
constexpr Float INV_FOUR_PI = (1/(4*PI));
constexpr Float PIhalf = (PI/2);
constexpr Float PIquarter = (PI/4);
constexpr Float TAU = (2*PI);
#define SQRT2 1.41421356237309504880168872420969807856967187537694807317667973799
#define BLOCK_SIZE 1024

#define int2 GPUPair<int, int>
//#define float2 GPUPair<Float, Float>
#define uint2 GPUPair<unsigned int, unsigned int>
#define Collision GPUPair<Ray, Box*>

#define FLOAT_ERROR_MARGIN 0.0000001

#define MAX_RAY_RECURSION_DEPTH 50
#define SAMPLES_PER_PIXEL 100
#define TILE_SIDE_X 1280
#define TILE_SIDE_Y 720
#define N_IMAGE_REPROCESSES 1
#define N_RENDERS 1
#define DENOISER_SAMPLES_PER_PIXEL 0

#define MAX_CONCURRENT_PIXELS 1920*1080

enum RenderMode { PATH_TRACING, PHOTON_MAPPING, HYBRID, MACHINE_LEARNING };

#define RENDER_MODE PATH_TRACING
#define CHECK_SUBSURFACE_SCATTERING true
#define USE_BVH 0

#define N_PHOTON_SAMPLES 1000000
#define PHOTON_SEARCH_RADIUS 100
//#define MAX_NEIGHBOURS (ulong)-1
#define MAX_NEIGHBOURS 100
#define MAX_PHOTON_RECURSION_DEPTH 30

#define N_ML_SAMPLES 1000
#define ML_DENSITY_RADIUS 0.3


#ifdef DEBUG
#define N_THREADS 1
#else
#define N_THREADS 128
#endif

#define resolutionX 640
#define resolutionY 360
#define RESCALE_IMAGE false

#define AMBIENT_LIGHT false

#define resolution uint2(resolutionX, resolutionY)

#define GENERAL_BLOCK_NUMBER 300
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
constexpr int THREADS_PER_BLOCK = THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y;


const Float ShadowEpsilon = 0.0001;


enum TransportMode { Radiance, Importance };

class ErrorFloat
{
    public:
    Float data, err;

    __host__ __device__
    ErrorFloat () 
        : data(0.0), err(0.0) 
    {}
    
    __host__ __device__
    ErrorFloat (Float data, Float err = 0.0) 
        : data(data), err(err) 
    {}

    __host__ __device__
    inline ErrorFloat operator+(ErrorFloat rhs) const
    {
        ErrorFloat ret;
        ret.data = data + rhs.data;
        ret.err = err + rhs.err;
        return ret;
    }

    __host__ __device__
    inline ErrorFloat operator-(ErrorFloat rhs) const
    {
        ErrorFloat ret;
        ret.data = data - rhs.data;
        ret.err = err + rhs.err;
        return ret;
    }

    __host__ __device__
    inline ErrorFloat operator*(ErrorFloat rhs) const
    {
        ErrorFloat ret;
        ret.data = data * rhs.data;
        ret.err = abs(ret.data) * rhs.err + abs(rhs.data) * err;
        return ret;
    }

    __host__ __device__
    inline ErrorFloat operator*(Float rhs) const
    {
        ErrorFloat ret;
        ret.data = data * rhs;
        ret.err = abs(ret.data) * err;
        return ret;
    }

    __host__ __device__
    inline ErrorFloat operator/(ErrorFloat rhs) const
    {
        ErrorFloat ret;
        ret.data = data / rhs.data;
        ret.err = (abs(ret.data) * rhs.err + abs(rhs.data) * err) / (rhs.data * rhs.data);
        return ret;
    }

    __host__ __device__
    inline bool operator==(ErrorFloat rhs) const
    {
        return data == rhs.data && err == rhs.err;
    }

    __host__ __device__
    inline bool operator!=(ErrorFloat rhs) const
    {
        return data != rhs.data || err != rhs.err;
    }

    __host__ __device__
    inline bool operator<(ErrorFloat rhs) const
    {
        return data < rhs.data;
    }

    __host__ __device__
    inline Float upperBound()
    {
        return data + err;
    }

    __host__ __device__
    inline Float lowerBound()
    {
        return data - err;
    }

    __host__ __device__
    inline operator float() const
    {
        return data;
    }

    __host__ __device__
    inline operator double() const
    {
        return data;
    }
};

__host__ __device__
inline ErrorFloat operator/(Float lhs, ErrorFloat rhs)
{
    ErrorFloat ret;
    ret.data = lhs / rhs.data;
    ret.err = (abs(ret.data) * rhs.err) / (rhs.data * rhs.data);
    return ret;
}

__host__ __device__
inline ErrorFloat operator*(Float lhs, ErrorFloat rhs)
{
    ErrorFloat ret;
    ret.data = lhs * rhs.data;
    ret.err = abs(ret.data) * rhs.err;
    return ret;
}

__host__ __device__
inline ErrorFloat sqrt (ErrorFloat f)
{
    ErrorFloat ret;
    ret.data = sqrt(f.data);
    ret.err = 0.5 * f.err / ret.data;
    return ret;
}

template <typename T>
__global__
void initVirtualClass(T **newClass, T oldClass)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *newClass = new T(oldClass);
}

void printStackCallTrace()
{
    const int maxFrames = 64; // Adjust this to the desired depth of the stack trace
    void* callStack[maxFrames];
    int numFrames = backtrace(callStack, maxFrames);

    char** symbols = backtrace_symbols(callStack, numFrames);

    for (int i = 0; i < numFrames; i++) {
        int status;
        char* demangled = abi::__cxa_demangle(symbols[i], 0, 0, &status);
        printf("#%d %s\n", i, (status == 0 ? demangled : symbols[i]));
        free(demangled);
    }

    free(symbols);
}
