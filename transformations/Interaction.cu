#pragma once

#include "../geometry/Point.cu"
#include "../radiometry/Spectrum.cu"
#include "../geometry/Ray.cu"
#include "../sampling/Sampler.cu"

class Interaction
{
public:

    Point3f worldPoint;

    __host__ __device__
    Interaction(Point3f worldPoint)
    {
        this->worldPoint = worldPoint;
    }

    __host__ __device__
    Interaction() {};
    
    __host__ __device__
    ~Interaction() {};

    virtual bool isSurfaceInteraction() const
    {
        return false;
    }

    virtual Spectrum sampleRay(
            Sampler *sampler,
            Ray &newRay) const = 0;

    virtual Spectrum getEmittedLight() const = 0;

    virtual Spectrum getAmbientScattering() const = 0;

    virtual inline Ray spawnRay(const Point3f &outgoingPoint, const Vector3f &direction) const = 0;

    virtual bool isMediumInteraction() const
    {
        return false;
    }

    virtual bool isScatteringMaterial() const
    {
        return false;
    }

    virtual bool isEmissiveMaterial() const
    {
        return false;
    }

    virtual bool isSpecularMaterial() const
    {
        return false;
    }

    virtual bool isSubsurfaceScattering() const
    {
        return false;
    }
};