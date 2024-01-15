#include "cuda_libraries/GPUVector.cu"
#include <iostream>


class increment
{
    int i;
    GPUVector<int> *out;

public:
    increment(int i, GPUVector<int> &out) : i(i), out(&out) {}

    __host__ __device__ 
    void operator()(int x) 
    { 
        x += i; 
        out->push_back(x);
    }
};

class checkNumber
{
    int i;

    public:

    checkNumber(int i) : i(i) {}

    __host__ __device__
    void operator()(int x)
    {
        assert(x == i);
    }
};


int main()
{
    GPUVector<int> vec(1000*1000*1000, 1);
    GPUVector<int> vec2(1000*1000*1000);
    
    auto func = increment(1, vec2);

    vec.GPUparallelForEach(func);

    //cudaDeviceSynchronize();

    vec2.GPUparallelForEach(checkNumber(2));


    return 0;
}