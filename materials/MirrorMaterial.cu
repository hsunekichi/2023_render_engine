#pragma once


#include "Material.cu"
#include "PerfectReflectionBSDF.cu"

#include "../transformations/SurfaceInteraction.cu"


class MirrorMaterial : public Material
{
public:
    Spectrum color;
    PerfectReflectionBSDF bsdf;
 
    MirrorMaterial(const Spectrum& color) 
    : bsdf(color)
    {
        this->color = color;
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

    friend
    __global__
    void copyMirrorToGPU(MirrorMaterial *cpuMaterial, MirrorMaterial *gpuMaterial);

    virtual Material* toGPU() const override
    {
        // Alloc space for the material
        MirrorMaterial *gpuMaterial;
        cudaMalloc(&gpuMaterial, sizeof(MirrorMaterial));

        // Copy the material to the GPU
        copyMirrorToGPU<<<1, 1>>>(const_cast<MirrorMaterial*>(this), gpuMaterial);

        // Return the pointer
        return gpuMaterial;
    }
};


__global__
void copyMirrorToGPU(MirrorMaterial *cpuMaterial, MirrorMaterial *gpuMaterial)
{
    // Copy the color
    gpuMaterial->color = cpuMaterial->color;

    // Copy the BSDF
    gpuMaterial->bsdf = cpuMaterial->bsdf;
}
