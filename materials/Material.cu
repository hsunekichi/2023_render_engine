#pragma once

#include "../radiometry/Spectrum.cu"
#include "BSDF.cu"

class SurfaceInteraction;

class Material
{
    protected:
    public:
    
    Material() {}

    ~Material() {}

    //virtual void computeScattering(SurfaceInteraction &interaction) const = 0;

    virtual Spectrum getEmittedRadiance(const SurfaceInteraction &interaction) const
    {
        return Spectrum(0.0f);
    }

    virtual Spectrum lightReflected(const Vector3f &cameraDirection, 
            const Vector3f &sampleDirection,
            Point2f localSamplePoint) const = 0;

    virtual Float pdf(const Vector3f &cameraDirection, const Vector3f &sampleDirection) const = 0;

    virtual Spectrum sampleDirection(
            const Vector3f &cameraDirection, 
            Sampler &sampler,
            Vector3f &sampleDir,
            Point2f localSamplePoint,
            Point3f &outgoingPoint,
            Vector3f L) const = 0;

    virtual Spectrum getAmbientScattering() const {
        return Spectrum(0.0f);
    }

    virtual bool isScattering() const {
        return false;
    }

    virtual bool isEmissive() const {
        return false;
    }

    virtual bool isSpecular() const {
        return false;
    }

    virtual bool isTransparent() const {
        return false;
    }

    virtual bool hasIndirect() const {
        return true;
    }

    virtual bool isSubsurfaceScattering() const {
        return false;
    }

    virtual void getBSSDFparameters(
        Spectrum &kd,
        Spectrum &sigmaS,
        Spectrum &sigmaA,
        Float &g,
        Float &eta,
        Point2f surfacePoint) const
    {
        sigmaS = Spectrum(0.0f);
        sigmaA = Spectrum(0.0f);
        g = 0.0f;
        eta = 1.0f;
    }

    //static void Bump(const *Texture<Float> &d, SurfaceInteraction *si);

    virtual Material* toGPU() const = 0;
};