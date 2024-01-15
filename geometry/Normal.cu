#pragma once

#include <iostream>

#include "../cuda_libraries/types.h"
#include "Vector.cu"

template <typename T>
class Normal3
{
    public:

    T x, y, z;

    __host__ __device__
    Normal3() 
    { 
        x = 0;
        y = 0;
        z = 0;
    }

    __host__ __device__
    Normal3(T xx, T yy, T zz)
    : x(xx), y(yy), z(zz) 
    {}

    __host__ __device__
    Normal3<T> operator-() const {
        return Normal3(-x, -y, -z);
    }

    __host__ __device__
    Normal3<T> operator+(const Normal3<T> &n) const {
        return Normal3<T>(x + n.x, y + n.y, z + n.z);
    }
    
    __host__ __device__
    Normal3<T>& operator+=(const Normal3<T> &n) 
    {
        x += n.x; 
        y += n.y; 
        z += n.z;

        return *this;
    }

    __host__ __device__
    Normal3<T> operator- (const Normal3<T> &n) const {
        return Normal3<T>(x - n.x, y - n.y, z - n.z);
    }
    
    __host__ __device__
    Normal3<T>& operator-=(const Normal3<T> &n) 
    {
        x -= n.x; 
        y -= n.y; 
        z -= n.z;

        return *this;
    }

    __host__ __device__
    Vector3<T> toVector() const {
        return Vector3<T>(x, y, z);
    }

    __host__ __device__
    bool hasNaNs() const {
        return std::isnan(x) || std::isnan(y) || std::isnan(z);
    }

    __host__ __device__
    Normal3<T> operator*(T f) const {
        return Normal3<T>(f*x, f*y, f*z);
    }
    
    __host__ __device__
    Normal3<T> &operator*=(T f) 
    {
        x *= f; 
        y *= f; 
        z *= f;

        return *this;
    }

    __host__ __device__
    Normal3<T> operator/(T f) const 
    {
        assert(f != 0);
        Float inv = (Float)1 / f;

        return Normal3<T>(x * inv, y * inv, z * inv);
    }
    
    __host__ __device__
    Normal3<T> &operator/=(T f) 
    {
        assert(f != 0);
        Float inv = (Float)1 / f;
        x *= inv; 
        y *= inv; 
        z *= inv;

        return *this;
    }

    __host__ __device__
    Float lengthSquared() const { 
        return x*x + y*y + z*z; 
    }

    __host__ __device__
    Float length() const { 
        return sqrt(lengthSquared()); 
    }


    __host__ __device__
    Normal3<T>(const Normal3<T> &n) 
    {
        assert(!n.hasNaNs());
        x = n.x; 
        y = n.y; 
        z = n.z;
    }
    
    __host__ __device__
    Normal3<T> &operator=(const Normal3<T> &n) 
    {
        assert(!n.hasNaNs());
        x = n.x; 
        y = n.y;
        z = n.z;

        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Normal3<T> &v) 
    {
        os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
        return os;
    }
    
    //Construct a normal from a vector
    explicit
    __host__ __device__
    Normal3<T>(const Vector3<T> &v) 
    {
        assert(!v.hasNaNs());
        x = v.x; 
        y = v.y; 
        z = v.z;
    }

    __host__ __device__
    bool operator==(const Normal3<T> &n) const {
        return x == n.x && y == n.y && z == n.z;
    }

    __host__ __device__
    bool operator!=(const Normal3<T> &n) const {
        return x != n.x || y != n.y || z != n.z;
    }
    
    __host__ __device__
    T operator[](int i) const 
    { 
        assert(i >= 0 && i <= 2);

        if (i == 0) 
            return x;
        else if (i == 1) 
            return y;
        else 
            return z;
    }
    
    __host__ __device__
    T &operator[](int i) 
    { 
        assert(i >= 0 && i <= 2);

        if (i == 0) 
            return x;
        else if (i == 1) 
            return y;
        else 
            return z;
    }

    __host__ __device__
    void print() const
    {
        printf("normal[%f, %f, %f]\n", x, y, z);
    }

    // Inverse * multiplication with T
    __host__ __device__
    friend Normal3<T> operator*(T f, const Normal3<T> &n) 
    {
        return Normal3<T>(f*n.x, f*n.y, f*n.z);
    }
};

typedef Normal3<Float> Normal3f;
typedef Normal3<int> Normal3i;


// Absolute value of normal
template <typename T>
__host__ __device__
Normal3<T> abs(const Normal3<T> &n) {
    return Normal3<T>(std::abs(n.x), std::abs(n.y), std::abs(n.z));
}


// Normal3 with vector3 versions
template <typename T>
__host__ __device__
T dot (const Normal3<T> &v1, const Vector3<T> &v2){
    return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}

template <typename T>
__host__ __device__
T absDot (const Normal3<T> &v1, const Vector3<T> &v2) {
    return abs(dot(v1, v2));
}

template <typename T>
__host__ __device__
T dot (const Normal3<T> &v1, const Normal3<T> &v2){
    return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}

template <typename T>
__host__ __device__
Vector3<T> cross (const Normal3<T> &v1, const Vector3<T> &v2)
{
    return Vector3<T>((v1.y * v2.z) - (v1.z * v2.y),
                      (v1.z * v2.x) - (v1.x * v2.z),
                      (v1.x * v2.y) - (v1.y * v2.x));
}

template <typename T>
__host__ __device__
Normal3<T> normalize (const Normal3<T> &v) {
    return v / v.length();
}


//Vector3 with normal3 versions
template <typename T>
__host__ __device__
T dot (const Vector3<T> &v1, const Normal3<T> &v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template <typename T>
__host__ __device__
T absDot (const Vector3<T> &v1, const Normal3<T> &v2) {
    return abs(dot(v1, v2));
}

// Cross product
template <typename T>
__host__ __device__
Vector3<T> cross (const Vector3<T> &v1, const Normal3<T> &v2)
{
    return Vector3<T>((v1.y * v2.z) - (v1.z * v2.y),
                      (v1.z * v2.x) - (v1.x * v2.z),
                      (v1.x * v2.y) - (v1.y * v2.x));
}
