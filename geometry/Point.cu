#pragma once
#include "../cuda_libraries/types.h"
#include "../cuda_libraries/math.cu"
#include "Vector.cu"

template <typename T>
class Point3
{
    public:
    T x, y, z;

    __host__ __device__
    Point3() 
    { 
        x = 0;
        y = 0;
        z = 0; 
    }

    __host__ __device__
    Point3(T x, T y, T z) : x(x), y(y), z(z) {
        assert(!hasNaNs());
    }

    __host__ __device__
    Point3(const Point3<T> &p) 
    {
        assert(!p.hasNaNs());

        x = p.x; 
        y = p.y; 
        z = p.z;
    }

    __host__ __device__
    explicit Point3(const Vector3<T> &v)
    {
        assert(!v.hasNaNs());

        x = v.x; 
        y = v.y; 
        z = v.z;
    }

    // Construct a vector with a point
    __host__ __device__
    Vector3<T> toVector() const {
        return Vector3<T>(x, y, z);
    }
    
    __host__ __device__
    Point3<T> &operator=(const Point3<T> &p) 
    {
        assert(!p.hasNaNs());
        x = p.x; 
        y = p.y; 
        z = p.z;

        return *this;
    }

    __host__
    friend std::ostream& operator<<(std::ostream& os, const Point3<T> &p) 
    {
        Point3<T> p2 = p;

        if (abs(p2.x) < FLOAT_ERROR_MARGIN)
            p2.x = 0;

        if (abs(p2.y) < FLOAT_ERROR_MARGIN)
            p2.y = 0;
        
        if (abs(p2.z) < FLOAT_ERROR_MARGIN)
            p2.z = 0;


        os << "point[" << p2.x << ", " << p2.y << ", " << p2.z << "]";
        return os;
    }

    __host__ __device__
    Point3<T> operator+(const Vector3<T> &v) const {
        return Point3<T>(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__
    Point3<T> &operator+=(const Vector3<T> &v) 
    {
        x += v.x; 
        y += v.y; 
        z += v.z;

        return *this;
    }

    __host__ __device__
    Vector3<T> operator-(const Point3<T> &p) const {
        return Vector3<T>(x - p.x, y - p.y, z - p.z);
    }

    __host__ __device__
    Point3<T> operator-(const Vector3<T> &v) const {
        return Point3<T>(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__
    Point3<T> &operator-=(const Vector3<T> &v) 
    {
        x -= v.x; 
        y -= v.y; 
        z -= v.z;

        return *this;
    }

    __host__ __device__
    Point3<T> &operator+=(const Point3<T> &p) 
    {
        x += p.x; 
        y += p.y; 
        z += p.z;

        return *this;
    }

    __host__ __device__
    Point3<T> operator+(const Point3<T> &p) const {
        return Point3<T>(x + p.x, y + p.y, z + p.z);
    }

    __host__ __device__
    Point3<T> operator*(T f) const {
        return Point3<T>(f*x, f*y, f*z);
    }

    __host__ __device__
    Point3<T> &operator*=(T f) 
    {
        x *= f; 
        y *= f; 
        z *= f;

        return *this;
    }

    __host__ __device__
    Point3<T> operator/(T f) const 
    {
        Float inv = (Float)1 / f;
        return Point3<T>(inv*x, inv*y, inv*z);
    }

    __host__ __device__
    Point3<T> &operator/=(T f) 
    {
        Float inv = (Float)1 / f;
        x *= inv; 
        y *= inv; 
        z *= inv;

        return *this;
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

        return z;
    }

    __host__ __device__
    inline bool isZero() const {
        return x == 0 && y == 0 && z == 0;
    }

    __host__ __device__
    inline bool operator==(const Point3<T> &p) const {
        return x == p.x && y == p.y && z == p.z;
    }

    __host__ __device__
    inline bool operator!=(const Point3<T> &p) const {
        return x != p.x || y != p.y || z != p.z;
    }

    __host__ __device__
    inline bool hasNaNs() const {
        return isNaN(x) || isNaN(y) || isNaN(z);
    }

    __host__ __device__
    bool hasInf() const {
        return isInf(x) || isInf(y);
    }

    __host__ __device__
    inline Point3<T> operator-() const { 
        return Point3<T>(-x, -y, -z); 
    }

    __host__ __device__
    void print() const {
        printf("point[%f, %f, %f]\n", x, y, z);
    }
};

typedef Point3<Float> Point3f;
typedef Point3<int>   Point3i;

template <typename T> 
__host__ __device__
inline Float distance(const Point3<T> &p1, const Point3<T> &p2) {
    return (p1 - p2).length();
}

template <typename T> 
__host__ __device__
inline Float distanceSquared(const Point3<T> &p1, const Point3<T> &p2) {
    return (p1 - p2).lengthSquared();
}


template <typename T>
class Point2
{
    public:

    T x, y;

    __host__ __device__
    explicit Point2(const Point3<T> &p) 
    : x(p.x), y(p.y) 
    {
        assert(!hasNaNs());
    }

    __host__ __device__
    Point2() 
    { 
        x = 0;
        y = 0; 
    }
    
    __host__ __device__
    Point2(T xx, T yy)
        : x(xx), y(yy) {
        assert(!hasNaNs());
    }
    
    __host__ __device__
    Point2(const Point2<T> &p) 
    {
        assert(!p.hasNaNs());
        x = p.x; 
        y = p.y;
    }

    __host__ __device__
    Point2 (const Vector2<T> &v)
    {
        assert(!v.hasNaNs());
        x = v.x; 
        y = v.y;
    }

    template <typename U>
    __host__ __device__
    Point2 (const Point2<U> &p)
    {
        assert(!p.hasNaNs());
        x = (T)p.x; 
        y = (T)p.y;
    }

    // Construct a vector with a point
    Vector2<T> toVector() const
    {
        return Vector2<T>(x, y);
    }

    __host__ __device__
    bool isZero() const {
        return x == 0 && y == 0;
    }

    
    __host__ __device__
    Point2<T> &operator=(const Point2<T> &p) 
    {
        assert(!p.hasNaNs());
        x = p.x; 
        y = p.y;

        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Point2<T> &p) 
    {
        Point2<T> p2 = p;

        if (abs(p2.x) < FLOAT_ERROR_MARGIN)
            p2.x = 0;
        
        if (abs(p2.y) < FLOAT_ERROR_MARGIN)
            p2.y = 0;

        os << "point[" << p2.x << ", " << p2.y << "]";
        return os;
    }

    
    __host__ __device__
    Point2<T> operator+(const Vector2<T> &v) const 
    {
        assert(!v.hasNaNs());
        return Point2<T>(x + v.x, y + v.y);
    }

    
    __host__ __device__
    Point2<T> &operator+=(const Vector2<T> &v) 
    {
        assert(!v.hasNaNs());
        x += v.x; 
        y += v.y;

        return *this;
    }

    __host__ __device__
    Vector2<T> operator-(const Point2<T> &p) const 
    {
        assert(!p.hasNaNs());
        return Vector2<T>(x - p.x, y - p.y);
    }
    
    __host__ __device__
    Point2<T> operator-(const Vector2<T> &v) const 
    {
        assert(!v.hasNaNs());
        return Point2<T>(x - v.x, y - v.y);
    }

    __host__ __device__
    Point2<T> operator-() const { 
        return Point2<T>(-x, -y);
    }

    __host__ __device__
    Point2<T> operator-(const T &f) const 
    {
        return Point2<T>(x - f, y - f);
    }

    __host__ __device__
    Point2<T> &operator-=(const Vector2<T> &v) 
    {
        assert(!v.hasNaNs());
        x -= v.x; 
        y -= v.y;

        return *this;
    }

    __host__ __device__
    Point2<T> &operator+=(const Point2<T> &p) 
    {
        assert(!p.hasNaNs());
        x += p.x; 
        y += p.y;

        return *this;
    }

    __host__ __device__
    Point2<T> operator+(const Point2<T> &p) const 
    {
        assert(!p.hasNaNs());
        return Point2<T>(x + p.x, y + p.y);
    }

    // + operator with int
    Point2<T> operator+(const int &f) const 
    {
        return Point2<T>(x + f, y + f);
    }

    __host__ __device__
    Point2<T> operator* (T f) const {
        return Point2<T>(f*x, f*y);
    }

    __host__ __device__
    Point2<T> &operator*=(T f) 
    {
        x *= f; 
        y *= f;

        return *this;
    }

    __host__ __device__
    Point2<T> operator/ (T f) const 
    {
        Float inv = (Float)1 / f;
        return Point2<T>(inv*x, inv*y);
    }

    __host__ __device__
    Point2<T> &operator/=(T f) 
    {
        Float inv = (Float)1 / f;
        x *= inv; 
        y *= inv;

        return *this;
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
    bool operator==(const Point2<T> &p) const {
        return x == p.x && y == p.y;
    }

    __host__ __device__
    bool operator!=(const Point2<T> &p) const {
        return x != p.x || y != p.y;
    }

    __host__ __device__
    bool hasNaNs() const {
        return isNaN(x) || isNaN(y);
    }

    __host__ __device__
    bool hasInf() const {
        return isInf(x) || isInf(y);
    }
};

typedef Point2<Float> Point2f;
typedef Point2<int>   Point2i;

template <typename T>
__host__ __device__
inline Float distance(const Point2<T> &p1, const Point2<T> &p2) {
    return (p1 - p2).length();
}

template <typename T>
__host__ __device__
inline Float distanceSquared(const Point2<T> &p1, const Point2<T> &p2) {
    return (p1 - p2).lengthSquared();
}

// pow2
__host__ __device__
inline Point2f pow2(const Point2f &p) {
    return Point2f(p.x*p.x, p.y*p.y);
}


__host__ __device__
Point2f operator*(Float f, const Point2f &p) {
    assert(!p.hasNaNs());
    return p*f;
}

__host__ __device__
Point3f operator*(Float f, const Point3f &p) {
    assert(!p.hasNaNs());
    return p*f;
}

// Inverse - operator
__host__ __device__
Point2f operator-(Float f, const Point2f &p) {
    assert(!p.hasNaNs());
    return Point2f(f - p.x, f - p.y);
}


// Inverse - operator
__host__ __device__
Point2f operator*(Point2f p1, const Point2f &p2) {
    assert(!p1.hasNaNs());
    assert(!p2.hasNaNs());
    return Point2f(p1.x * p2.x, p1.y * p2.y);
}

// Inverse / operator
__host__ __device__
Point2f operator/(Float f, const Point2f &p) {
    assert(!p.hasNaNs());
    return Point2f(f / p.x, f / p.y);
}
