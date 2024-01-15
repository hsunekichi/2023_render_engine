#pragma once


#include "Material.cu"
#include "PerfectReflectionBSDF.cu"
#include "SpecularRefractionBSDF.cu"
#include "LambertianBSDF.cu"

#include "BeckmannBSDF.cu"

#include "../transformations/SurfaceInteraction.cu"

class PlasticUnrealMaterial : public Material
{
public:
    Spectrum ks, kd;
    Float ior;

    PerfectReflectionBSDF reflectionBSDF;
    LambertianBRDF lambertian;
 
    PlasticUnrealMaterial(const Spectrum& color, Float ior) 
    : ks(color), kd(color),
        lambertian(kd),
        reflectionBSDF(ks)
    { }

    bool isScattering() const override
    {
        return true;
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
        return false;
    }

    virtual Spectrum lightReflected(const Vector3f &cameraDirection, 
            const Vector3f &sampleDirection,
            Point2f localSamplePoint) const override
    {
        return lambertian.lightReflected(cameraDirection, sampleDirection, localSamplePoint);
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
        
        Float ps = ks.maxComponent() / (ks.maxComponent() + kd.maxComponent());

        if (rouletteSample < 0.7)
            return lambertian.sampleDirection(cameraDirection, sampler, sampleDirection, localSamplePoint);
        else 
            return reflectionBSDF.sampleDirection(cameraDirection, sampler, sampleDirection, localSamplePoint);
    }
    
    Material* toGPU() const override
    {
        throw std::runtime_error("Dielectric material toGPU not implemented");
        return nullptr;
    }
};