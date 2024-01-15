#pragma once

#include <iostream>

#include "Vector.cu"
#include "Point.cu"
#include "../cuda_libraries/types.h"

#pragma once

class Medium;

class Ray
{
    public:
    Point3f origin;
    Vector3f direction;
    Vector3f invDirection;

    bool cameraRay = false;

    mutable Float maximumOffset;
    Float time;
    const Medium *medium;

    __host__ __device__
    Ray() 
    : maximumOffset(INFINITY), time(0.f), medium(nullptr) 
    { }

    __host__ __device__
    Ray(const Point3f &origin, const Vector3f &direction, Float maximumOffset = INFINITY,
            Float time = 0.f, const Medium *medium = nullptr,
            bool _cameraRay = false)
        : origin(origin), direction(direction), 
            maximumOffset(maximumOffset), time(time), medium(medium),
            invDirection(1 / direction.x, 1 / direction.y, 1 / direction.z),
            cameraRay(_cameraRay) 
    { }

    // Returns a point along the ray at a given offset
    __host__ __device__
    Point3f operator()(Float offset) const { 
        return origin + direction * offset; 
    }

    __host__ __device__
    bool hasNaNs() const {
        return (origin.hasNaNs() || direction.hasNaNs() || std::isnan(maximumOffset));
    }

    __host__
    friend std::ostream& operator<<(std::ostream& os, const Ray &ray) 
    {
        os << "ray[origin=" << ray.origin << ", direction=" << ray.direction << ", maximumOffset="
            << ray.maximumOffset << ", time=" << ray.time << "]";
            
        return os;
    }

    __host__ __device__
    void print() const
    {
        printf("Ray[origin=%f, %f, %f, direction=%f, %f, %f, maximumOffset=%f, time=%f]\n", 
                origin.x, origin.y, origin.z, direction.x, direction.y, direction.z, maximumOffset, time);
    }

    __host__ __device__
    Float getOffset(Point3f p) const
    {
        return (p.x - origin.x) * invDirection.x;
    }

    __host__ __device__
    bool isCameraRay() const
    {
        return cameraRay;
    }
};


/*
inline Point3f offsetRayOrigin(const Point3f &p, const Vector3f &pError,
                               const Normal3f &n, const Vector3f &w) 
{
    Float d = dot(abs(n), pError);

    Vector3f offset = Vector3f(n.x, n.y, n.z) * d;

    if (dot(w, n) < 0)
        offset = -offset;

    Point3f po = p + offset;

    // Round offset point po away from p>
    return po;
}
*/
