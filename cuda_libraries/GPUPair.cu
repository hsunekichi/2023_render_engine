#pragma once

template<typename T1, typename T2>
class GPUPair
{
    public:

    T1 x;
    T2 y;

    __host__ __device__ 
    GPUPair(T1 first, T2 second) : x(first), y(second) {}

    __host__ __device__
    GPUPair() : x(T1()), y(T2()) {}

    __host__ __device__
    void swap() 
    {
        T1 temp = x;
        x = y;
        y = temp;
    }
};