#pragma once

#include "../geometry/Vector.cu"
#include "../radiometry/Spectrum.cu"
#include "../sampling/Sampler.cu"

#include "../cuda_libraries/math.cu"
#include "../cuda_libraries/geometricMath.cu"

class BSDF
{
public:
    BSDF() {};
    
    ~BSDF() {};

    virtual Spectrum lightReflected(const Vector3f& cameraDirection, 
            const Vector3f& sampleDirection, 
            Point2f localPoint) const = 0;

    virtual Spectrum sampleDirection(
            Vector3f cameraDirection, 
            Sampler &sampler,
            Vector3f &sampleDirection,
            Point2f localPoint) const = 0;

    virtual Float pdf(const Vector3f& wo, const Vector3f& wi) const = 0;
};