#pragma once
#include "../cuda_libraries/types.h"
#include "../cuda_libraries/math.cu"
#include "Vector.cu"



class Complex
{
public:
    Float real, imaginary;

    __host__ __device__
    Complex(Float real=0, Float imaginary=0) : real(real), imaginary(imaginary) {}


    __host__ __device__
    Complex operator+(const Complex &c) const {
        return Complex(real + c.real, imaginary + c.imaginary);
    }

    __host__ __device__
    Complex operator-(const Complex &c) const {
        return Complex(real - c.real, imaginary - c.imaginary);
    }    

    __host__ __device__
    Complex operator*(const Complex &c) const 
    {
        return Complex(real * c.real - imaginary * c.imaginary,
                       real * c.imaginary + imaginary * c.real);
    }

    __host__ __device__
    Complex operator/(const Complex &c) const 
    {
        Float scale = 1 / (c.real * c.real + c.imaginary * c.imaginary);
        return Complex(scale * (real * c.real + imaginary * c.imaginary),
                       scale * (imaginary * c.real - real * c.imaginary));
    }

    __host__ __device__
    friend Complex operator/(Float f, Complex z) 
    {
        return Complex(f) / z;
    }

    __host__ __device__
    friend Complex operator*(Float f, Complex z) 
    {
        return Complex(f) * z;
    }

    __host__ __device__
    friend Complex operator+(Float f, Complex z) 
    {
        return Complex(f) + z;
    }

    __host__ __device__
    friend Complex operator-(Float f, Complex z) 
    {
        return Complex(f) - z;
    }


};

__host__ __device__
Float norm(const Complex &p) {
    return p.real * p.real + p.imaginary * p.imaginary;
}

__host__ __device__
Float abs(const Complex &p) {
    return sqrt(norm(p));
}



__host__ __device__
Complex sqrt(const Complex &z) 
{
    Float n = abs(z), t1 = sqrt(Float(.5) * (n + abs(z.real))),
    t2 = Float(.5) * z.imaginary / t1;

    if (n == 0)
        return Complex(0);

    if (z.real >= 0)
        return Complex(t1, t2);
    else
        return Complex(abs(t2), copysign(t1, z.imaginary));
}



__host__ __device__
Complex pow2(const Complex &p) {
    return p * p;
}
