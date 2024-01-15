#pragma once


#include "Material.cu"
#include "LambertianBSDF.cu"

#include "../transformations/SurfaceInteraction.cu"


class LambertianMaterial : public Material
{
public:
    Spectrum ka;
    LambertianBRDF bsdf;

    bool transparent;
 
    LambertianMaterial(const Spectrum& _kd, 
        const Spectrum& _ka = Spectrum(0),
        Texture *texture = nullptr,
        bool _transparent = false) 
    : bsdf(_kd, texture)
    {
        this->ka = _ka;
        this->transparent = _transparent;
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
        return ka;
    }


    bool isScattering() const override
    {
        return true;
    }

    bool isSpecular() const
    {
        return false;
    }

    bool isTransparent() const override
    {
        return transparent;
    }
    
    friend
    __global__
    void copyLambertianToGPU(LambertianMaterial *cpuMaterial, LambertianMaterial *gpuMaterial);

    virtual Material* toGPU() const override
    {
        // Alloc space for the material
        LambertianMaterial *gpuMaterial;
        cudaMalloc(&gpuMaterial, sizeof(LambertianMaterial));

        // Copy the material to the GPU
        copyLambertianToGPU<<<1, 1>>>(const_cast<LambertianMaterial*>(this), gpuMaterial);

        // Return the pointer
        return gpuMaterial;
    }
};


__global__
void copyLambertianToGPU(LambertianMaterial *cpuMaterial, LambertianMaterial *gpuMaterial)
{
    // Copy the ka
    gpuMaterial->ka = cpuMaterial->ka;

    // Copy the BRDF
    gpuMaterial->bsdf = cpuMaterial->bsdf;
}

