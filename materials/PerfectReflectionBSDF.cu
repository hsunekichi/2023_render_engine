#pragma once

#include "BSDF.cu"
#include "../radiometry/Spectrum.cu"
#include "../geometry/Vector.cu"
#include "../geometry/Point.cu"

class PerfectReflectionBSDF : BSDF
{
public:

    Spectrum color;

    PerfectReflectionBSDF(Spectrum color)
    {
        this->color = color;
    }

    virtual Spectrum lightReflected(const Vector3f& cameraDirection, 
            const Vector3f& sampleDirection,
            Point2f localPoint) const
    {
        return Spectrum(0);
    }

    virtual Spectrum sampleDirection(Vector3f cameraDirection, 
            Sampler &sampler,
            Vector3f &sampleDirection,
            Point2f localPoint) const override
    {
        sampleDirection = Vector3f(-cameraDirection.x, -cameraDirection.y, cameraDirection.z);
        return color;
    }

    virtual Float pdf(const Vector3f& cameraDirection, const Vector3f& sampleDirection) const
    {
        return 0;
    }
};