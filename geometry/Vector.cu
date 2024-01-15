#pragma once
#include <iostream>
#include "../cuda_libraries/types.h"

template <typename T>
class Vector3
{
    public:
    T x, y, z;

    __host__ __device__
    Vector3() 
    { 
        x = 0;
        y = 0;
        z = 0; 
    }

    __host__ __device__
    Vector3(T x, T y, T z)
        : x(x), y(y), z(z) 
    {
       assert(!hasNaNs());
    }
    
    __host__ __device__ 
    Vector3(T f) 
        : x(f), y(f), z(f) 
    {
        assert(!hasNaNs());
    }

    __host__ __device__
    bool hasNaNs() const {
        return std::isnan(x) || std::isnan(y) || std::isnan(z);
    }

    __host__ __device__
    Vector3(const Vector3<T> &v) 
    {
        assert(!v.hasNaNs());
        x = v.x; y = v.y; z = v.z;
    }


    __host__ __device__
    T operator[](int i) const 
    { 
        assert(i >= 0 && i <= 2);

        // Better to have dataDivergence than different return points
        //  On GPU, the else will be indifferent
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

        // Here, we must return the reference to the original value
        if (i == 0) 
            return x;
        else if (i == 1) 
            return y;
        else
            return z;
    }
       
    __host__ __device__
    Vector3<T> &operator=(const Vector3<T> &v) 
    {
        assert(!v.hasNaNs());

        x = v.x; 
        y = v.y; 
        z = v.z;

        return *this;
    }

    __host__ __device__
    inline int maxComponent() const
    {
        return max(x, max(y, z));
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector3<T> &v) 
    {
        Vector3<T> v2(v.x, v.y, v.z);
        std::string output;

        if (abs(v2.x) < FLOAT_ERROR_MARGIN)
            v2.x = 0;

        if (abs(v2.y) < FLOAT_ERROR_MARGIN)
            v2.y = 0;

        if (abs(v2.z) < FLOAT_ERROR_MARGIN)
            v2.z = 0;

        output = "vector[" + std::to_string(v2.x) + ", " + std::to_string(v2.y) + ", " + std::to_string(v2.z) + "]";
        os << output;

        return os;
    }

    __host__ __device__
    Vector3<T> operator+(const Vector3<T> &v) const {
        return Vector3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__
    Vector3<T>& operator+=(const Vector3<T> &v) 
    {
        x += v.x; 
        y += v.y; 
        z += v.z;

        return *this;
    }

    __host__ __device__
    Vector3<T> operator-(const Vector3<T> &v) const {
        return Vector3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__
    Vector3<T>& operator-=(const Vector3<T> &v)
    {
        x -= v.x; y -= v.y; z -= v.z;

        return *this;
    }

    __host__ __device__
    bool operator==(const Vector3<T> &v) const {
        return x == v.x && y == v.y && z == v.z;
    }

    __host__ __device__
    bool operator!=(const Vector3<T> &v) const {
        return x != v.x || y != v.y || z != v.z;
    }

    __host__ __device__
    Vector3<T> operator*(const Vector3<T> &v) const {
        return Vector3(x * v.x, y * v.y, z * v.z);
    }

    __host__ __device__
    Vector3<T> operator*(T s) const { 
        return Vector3<T>(s*x, s*y, s*z); 
    }

    __host__ __device__
    Vector3<T> &operator*=(T s) 
    {
        x *= s; 
        y *= s; 
        z *= s;

        return *this;
    }
    __host__ __device__
    Vector3<T> operator/(T f) const 
    {
        assert(f != 0);
        Float inv = (Float)1 / f;

        return Vector3<T>(x * inv, y * inv, z * inv);
    }
    
    __host__ __device__
    Vector3<T> &operator/=(T f) 
    {
        assert(f != 0);

        Float inv = (Float)1 / f;

        x *= inv; 
        y *= inv; 
        z *= inv;

        return *this;
    }

    __host__ __device__
    Vector3<T> operator-() const { 
        return Vector3<T>(-x, -y, -z); 
    }

    __host__ __device__
    Float lengthSquared() const { 
        return x * x + y * y + z * z; 
    }

    __host__ __device__
    Float length() const { 
        return sqrt(lengthSquared()); 
    }

    __host__ __device__
    void print() const
    {
        printf("Vector3[%f, %f, %f]\n", x, y, z);
    }
};

// Specialization for Float
template <>
__host__ __device__
bool Vector3<Float>::operator==(const Vector3<Float> &v) const 
{
    return (abs(x - v.x) < FLOAT_ERROR_MARGIN) 
            && (abs(y - v.y) < FLOAT_ERROR_MARGIN) 
            && (abs(z - v.z) < FLOAT_ERROR_MARGIN);
}


typedef Vector3<Float> Vector3f;
typedef Vector3<int>   Vector3i;


template <typename T>
class Vector2
{
    public:
    T x, y;

    __host__ __device__
    Vector2() 
    { 
        x = 0;
        y = 0; 
    }

    __host__ __device__
    Vector2(T xx, T yy)
        : x(xx), y(yy) 
    {
        assert(!hasNaNs());
    }

    __host__ __device__
    bool hasNaNs() const {
        return std::isnan(x) || std::isnan(y); 
    }

    // The default versions of these are fine for release builds; for debug
    // we define them so that we can add the assert checks.
    __host__ __device__
    Vector2(const Vector2<T> &v) 
    {
        assert(!v.hasNaNs());

        x = v.x; y = v.y;
    }

    __host__ __device__
    Vector2<T> &operator=(const Vector2<T> &v) 
    {
        assert(!v.hasNaNs());
        x = v.x; 
        y = v.y;

        return *this;
    }

    __host__ __device__
    friend std::ostream& operator<<(std::ostream& os, const Vector2<T> &v) 
    {
        Vector2<T> v2(v.x, v.y);

        if (abs(v2.x) < FLOAT_ERROR_MARGIN)
            v2.x = 0;
        
        if (abs(v2.y) < FLOAT_ERROR_MARGIN)
            v2.y = 0;

        os << "vector[" << v2.x << ", " << v2.y << "]";
        return os;
    }
    
    __host__ __device__
    Vector2<T> operator+(const Vector2<T> &v) const 
    {
        assert(!v.hasNaNs());
        return Vector2(x + v.x, y + v.y);
    }
    
    __host__ __device__
    Vector2<T>& operator+=(const Vector2<T> &v) 
    {
        assert(!v.hasNaNs());
        x += v.x; 
        y += v.y;

        return *this;
    }

    __host__ __device__
    Vector2<T> operator-(const Vector2<T> &v) const 
    {
        assert(!v.hasNaNs());
        return Vector2(x - v.x, y - v.y);
    }

    
    __host__ __device__
    Vector2<T>& operator-=(const Vector2<T> &v) 
    {
        assert(!v.hasNaNs());
        x -= v.x; 
        y -= v.y;

        return *this;
    }

    __host__ __device__
    bool operator==(const Vector2<T> &v) const {
        return x == v.x && y == v.y;
    }

    __host__ __device__
    bool operator!=(const Vector2<T> &v) const {
        return x != v.x || y != v.y;
    }

    __host__ __device__
    Vector2<T> operator*(T f) const { 
        return Vector2<T>(f*x, f*y); 
        }
    
    __host__ __device__
    Vector2<T> &operator*=(T f) 
    {
        assert(!std::isnan(f));
        x *= f; 
        y *= f;

        return *this;
    }

    __host__ __device__
    Vector2<T> operator/(T f) const 
    {
        assert(f != 0);
        Float inv = (Float)1 / f;

        return Vector2<T>(x * inv, y * inv);
    }
    
    __host__ __device__
    Vector2<T> &operator/=(T f) 
    {
        assert(f != 0);

        Float inv = (Float)1 / f;
        x *= inv; 
        y *= inv;

        return *this;
    }

    __host__ __device__
    Vector2<T> operator-() const { 
        return Vector2<T>(-x, -y); 
    }

    __host__ __device__
    T operator[](int i) const 
    {
        assert(i >= 0 && i <= 1);

        if (i == 0) 
            return x;
        else
            return y;
    }
    
    __host__ __device__
    T &operator[](int i) 
    {
        assert(i >= 0 && i <= 1);
        
        if (i == 0) 
            return x;
        else
            return y;
    }

    __host__ __device__
    Float lengthSquared() const { 
        return x*x + y*y; 
    }

    __host__ __device__
    Float length() const { 
        return std::sqrt(lengthSquared()); 
    }

    __host__ __device__
    void print() const
    {
        printf("Vector2[%f, %f]\n", x, y);
    }
};

// Specialization for float
template <>
__host__ __device__
bool Vector2<Float>::operator==(const Vector2<Float> &v) const 
{
    return (abs(x - v.x) < FLOAT_ERROR_MARGIN) 
            && (abs(y - v.y) < FLOAT_ERROR_MARGIN);
}


typedef Vector2<Float> Vector2f;
typedef Vector2<int>   Vector2i;



template <typename T>
__host__ __device__
T dot (const Vector3<T> &v1, const Vector3<T> &v2) {
    return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}

template <typename T>
__host__ __device__
T absDot (const Vector3<T> &v1, const Vector3<T> &v2) {
    return abs(dot(v1, v2));
}

// Cross product
template <typename T>
__host__ __device__
Vector3<T> cross (const Vector3<T> &v1, const Vector3<T> &v2)
{
    return Vector3<T>((v1.y * v2.z) - (v1.z * v2.y),
                      (v1.z * v2.x) - (v1.x * v2.z),
                      (v1.x * v2.y) - (v1.y * v2.x));
}

template <typename T>
__host__ __device__
Vector3<T> normalize (const Vector3<T> &v)
{
    return v / v.length();
}

template <typename T>
__host__ __device__
Vector3<T> abs (const Vector3<T> &v)
{
    return Vector3<T>(abs(v.x), abs(v.y), abs(v.z));
}


template <typename T>
__host__ __device__
Vector3<T> faceForward(const Vector3<T> &n, const Vector3<T> &v)
{
    return (dot(n, v) < 0.f) ? -n : n;
}





// 2D versions
template <typename T>
__host__ __device__
T dot (const Vector2<T> &v1, const Vector2<T> &v2) {
    return (v1.x * v2.x) + (v1.y * v2.y);
}

template <typename T>
__host__ __device__
T absDot (const Vector2<T> &v1, const Vector2<T> &v2) {
    return abs(dot(v1, v2));
}

// Cross
template <typename T>
__host__ __device__
T cross (const Vector2<T> &v1, const Vector2<T> &v2) {
    return (v1.x * v2.y) - (v1.y * v2.x);
}

template <typename T>
__host__ __device__
Vector2<T> normalize (const Vector2<T> &v)
{
    return v / v.length();
}

template <typename T>
__host__ __device__
Vector2<T> abs (const Vector2<T> &v)
{
    return Vector2<T>(abs(v.x), abs(v.y));
}

template <typename T>
__host__ __device__
Vector2<T> faceForward(const Vector2<T> &n, const Vector2<T> &v)
{
    return (dot(n, v) < 0.f) ? -n : n;
}


//Inverse * for Vector3
template <typename T>
__host__ __device__
Vector3<T> operator*(T f, const Vector3<T> &v) 
{
    return v * f; 
}

//Inverse * for Vector2
template <typename T>
__host__ __device__
Vector2<T> operator*(T f, const Vector2<T> &v) 
{
    return v * f; 
}


template <typename T>
__host__ __device__
void generateCoordinateSystem(const Vector3<T> &v1, 
        Vector3<T> &v2, 
        Vector3<T> &v3)
{
    if (abs(v1.x) > abs(v1.y)) 
        v2 = Vector3<T>(-v1.z, 0, v1.x) / sqrt(v1.x*v1.x + v1.z*v1.z);
    else
        v2 = Vector3<T>(0, v1.z, -v1.y) / sqrt(v1.y*v1.y + v1.z*v1.z);

    v3 = cross(v1, v2);
}


template <typename T>
__host__ __device__
T maxComponent(Vector3<T> v)
{
    return max(v.x, max(v.y, v.z));
}