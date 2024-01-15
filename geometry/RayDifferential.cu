#pragma once

#include "Ray.cu"

class RayDifferential : public Ray
{
    public:

    bool hasDifferentials;
    Point3f rxOrigin, ryOrigin;
    Vector3f rxDirection, ryDirection;

    __host__ __device__
    RayDifferential() { 
        hasDifferentials = false; 
    }

    __host__ __device__
    RayDifferential(const Point3f &origin, const Vector3f &direction,
            Float maxOffset = INFINITY, Float time = 0.f,
            const Medium *medium = nullptr)
        : Ray(origin, direction, maxOffset, time, medium) 
    {
        hasDifferentials = false; 
    }

    __host__ __device__
    RayDifferential(const Ray &ray) 
        : Ray(ray) 
    {
        hasDifferentials = false; 
    }

    __host__ __device__
    bool hasNaNs() const 
    {
        return Ray::hasNaNs() || 
            (hasDifferentials && (rxOrigin.hasNaNs() || ryOrigin.hasNaNs() ||
                                rxDirection.hasNaNs() || ryDirection.hasNaNs()));
    }

    __host__ __device__
    void scaleDifferentials(Float s) 
    {
        rxOrigin = origin + (rxOrigin - origin) * s;
        ryOrigin = origin + (ryOrigin - origin) * s;
        rxDirection = direction + (rxDirection - direction) * s;
        ryDirection = direction + (ryDirection - direction) * s;
    }
};