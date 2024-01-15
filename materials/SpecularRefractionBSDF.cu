#pragma once

#include "BSDF.cu"
#include "../radiometry/Spectrum.cu"
#include "../geometry/Vector.cu"
#include "../geometry/Point.cu"

class SpecularRefractionBSDF : BSDF
{
public:

    Float mediumRefractionIndex;
    Spectrum color;

    SpecularRefractionBSDF(Spectrum color, Float refractionIndex)
    {
        this->color = color;
        this->mediumRefractionIndex = refractionIndex;
    }


    ~SpecularRefractionBSDF() {}

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
        Float originRefrIndex = 1;
        Float destinyRefrIndex = mediumRefractionIndex;

        // Check if the ray is entering or exiting the medium
        if (cosine(cameraDirection) < 0)
            swap(originRefrIndex, destinyRefrIndex);
        
        Float refractionIndex = originRefrIndex / destinyRefrIndex;
        Vector3f normal = faceForward(Vector3f(0, 0, 1), cameraDirection);

        // Compute the refraction direction vector with snells law
        Float cosThetaOrigin = cosine(normal, cameraDirection);
        Float sin2ThetaOrigin = max(0.f, 1.f - cosThetaOrigin * cosThetaOrigin);
        Float sin2ThetaDestiny = refractionIndex * refractionIndex * sin2ThetaOrigin;

        // Check if the ray is totaly reflected
        if (sin2ThetaDestiny >= 1)
            return Spectrum(0);

        Float cosThetaT = sqrt(1 - sin2ThetaDestiny);
        sampleDirection = refractionIndex * -cameraDirection + 
            (refractionIndex * cosThetaOrigin - cosThetaT) * normal;

        /************ Compute fresnel equations for transmited radiance *************/
        Float refractedRadiance = 1 - fresnelDielectric(cosThetaOrigin, originRefrIndex, destinyRefrIndex);

        return color * refractedRadiance;
    }

    virtual Float pdf(const Vector3f& cameraDirection, const Vector3f& sampleDirection) const
    {
        return 0;
    }
};