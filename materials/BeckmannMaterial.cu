#pragma once


#include "Material.cu"
#include "BeckmannBSDF.cu"


class BeckmannMaterial : public Material
{
public:
    BeckmannBSDF bsdf;

    bool transparent;
 
    BeckmannMaterial(const Spectrum& kd, 
        const Spectrum& eta, const Spectrum& k,
        Float roughnessX, Float roughnessY,
        Texture *texture = nullptr) 
    : bsdf(kd, eta, k, roughnessX, roughnessY, texture)
    {
        this->transparent = false;
    }



    Spectrum lightReflected(const Vector3f &cameraDirection, 
            const Vector3f &sampleDirection,
            Point2f localSamplePoint) const override
    {
        return bsdf.lightReflected(cameraDirection, sampleDirection, localSamplePoint);
    }

    Float pdf(const Vector3f &cameraDirection, const Vector3f &sampleDirection) const override
    {
        return bsdf.pdf(cameraDirection, sampleDirection);
    }

    Spectrum sampleDirection(
            const Vector3f &cameraDirection, 
            Sampler &sampler,
            Vector3f &sampleDirection,
            Point2f localSamplePoint,
            Point3f &outgoingPoint,
            Vector3f L) const override
    {
        outgoingPoint = Point3f(0, 0, 0);
        return bsdf.sampleDirection(cameraDirection, sampler, sampleDirection, localSamplePoint);
    }

    Spectrum getAmbientScattering() const override
    {
        return Spectrum(0);
    }


    bool isScattering() const override
    {
        return true;
    }

    bool isSpecular() const
    {
        return true;
    }

    bool isTransparent() const override
    {
        return transparent;
    }

    Material* toGPU() const override
    {
        throw std::runtime_error("BeckmannMaterial toGPU not implemented");
        return nullptr;
    }


};
