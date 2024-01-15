#pragma once

#include "BSDF.cu"
#include "../radiometry/Spectrum.cu"
#include "../geometry/Vector.cu"
#include "../geometry/Point.cu"

class SpecularReflectionBSDF : BSDF
{
public:

    Float refractionIndex;
    Spectrum color;
    
    SpecularReflectionBSDF(Spectrum color, Float refractionIndex)
    {
        this->color = color;
        this->refractionIndex = refractionIndex;
    }

    ~SpecularReflectionBSDF() {}
    
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
        Float originRefractionIndex = 1;

        Float reflectedRadiance = fresnelDielectric(cosine(sampleDirection), 
                                    originRefractionIndex, refractionIndex);
        
        return color * reflectedRadiance;
    }
    
    virtual Float pdf(const Vector3f& cameraDirection, const Vector3f& sampleDirection) const
    {
        return 0;
    }
};