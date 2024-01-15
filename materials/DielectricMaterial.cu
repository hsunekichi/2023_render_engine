#pragma once


#include "Material.cu"
#include "SpecularReflectionBSDF.cu"
#include "SpecularRefractionBSDF.cu"

#include "BeckmannBSDF.cu"

#include "../transformations/SurfaceInteraction.cu"

class DielectricMaterial : public Material
{
public:
    Spectrum ks, kt;
    Float ior;

    SpecularReflectionBSDF reflectionBSDF;
    SpecularRefractionBSDF refractionBSDF;
 
    DielectricMaterial(const Spectrum& color, Float ior) 
    : ks(color), kt(color),
        refractionBSDF(kt, ior),
        reflectionBSDF(ks, ior)
    { }

    bool isScattering() const override
    {
        return false;
    }

    bool isEmissive() const override
    {
        return false;
    }

    bool isSpecular() const override
    {
        return true;
    }

    bool isTransparent() const override
    {
        return true;
    }

    virtual Spectrum lightReflected(const Vector3f &cameraDirection, 
            const Vector3f &sampleDirection,
            Point2f localSamplePoint) const override
    {
        return Spectrum(0);
    }

    Float pdf(const Vector3f &cameraDirection, const Vector3f &sampleDirection) const override
    {
        return 0;
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
        
        Float rouletteSample = sampler.get1Dsample();
        
        Float ps = ks.maxComponent() / (ks.maxComponent() + kt.maxComponent());

        if (rouletteSample < 0.7)
            return refractionBSDF.sampleDirection(cameraDirection, sampler, sampleDirection, localSamplePoint);
        else 
            return reflectionBSDF.sampleDirection(cameraDirection, sampler, sampleDirection, localSamplePoint);
    }
    
    Material* toGPU() const override
    {
        throw std::runtime_error("Dielectric material toGPU not implemented");
        return nullptr;
    }
};