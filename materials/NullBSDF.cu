#pragma once

#include "BSDF.cu"
#include "../radiometry/Spectrum.cu"
#include "../cuda_libraries/types.h"

class NullBSDF : public BSDF
{
    protected:

    public:

    NullBSDF() 
    {}

    virtual Spectrum lightReflected(const Vector3f& wo, const Vector3f& wi, Point2f localPoint) const
    {
        return Spectrum(0);
    }

    virtual Spectrum sampleDirection(
            Vector3f cameraDirection, 
            Sampler &sampler,
            Vector3f &sampleDirection,
            Point2f localPoint) const 
    {
        sampleDirection = Vector3f(0, 0, 0);
        return Spectrum(0);
    }


    virtual Float pdf(const Vector3f& cameraRay, const Vector3f& incomingRay) const
    {
        return 0;
    }
};