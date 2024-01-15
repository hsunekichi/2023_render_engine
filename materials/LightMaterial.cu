#pragma once

#include "Material.cu"
#include "NullBSDF.cu"
#include "BSDF.cu"
#include "../radiometry/Spectrum.cu"

#include "../cuda_libraries/types.h"


class LightMaterial : public Material
{
protected:
    Spectrum color;
    Float intensity;
    NullBSDF bsdf;

public: 
    LightMaterial(const Spectrum& color, Float intensity) 
    : bsdf(),
        color(color), 
        intensity(intensity)
    {}

    Spectrum getEmittedRadiance(const SurfaceInteraction &interaction) const override
    {
        return color * intensity;
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


    bool isScattering() const
    {
        return false;  
    }
 
    bool isEmissive() const
    {
        return true;
    }

    bool isSpecular() const
    {
        return false;
    }

    friend
    __global__
    void copyLightToGPU(LightMaterial *cpuMaterial, LightMaterial *gpuMaterial);

    virtual Material* toGPU() const override
    {
        // Alloc space for the material
        LightMaterial *gpuMaterial;
        cudaMalloc(&gpuMaterial, sizeof(LightMaterial));

        // Copy the material to the GPU
        copyLightToGPU<<<1, 1>>>(const_cast<LightMaterial*>(this), gpuMaterial);

        // Return the pointer
        return gpuMaterial;
    }
};


__global__
void copyLightToGPU(LightMaterial *cpuMaterial, LightMaterial *gpuMaterial)
{
    // Copy the color
    gpuMaterial->color = cpuMaterial->color;
    gpuMaterial->intensity = cpuMaterial->intensity;

    // Copy the BSDF
    gpuMaterial->bsdf = cpuMaterial->bsdf;
}
