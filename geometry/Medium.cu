#pragma once

class Medium
{
    public:

    Medium** toGPU() const
    {
        // Allocate memory on gpu for pointer
        Medium** gpuPtr;
        cudaMalloc(&gpuPtr, sizeof(Medium*));

        return gpuPtr;
    }
};