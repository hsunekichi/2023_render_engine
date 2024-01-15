#pragma once


#include "Material.cu"
#include "SpecularReflectionBSDF.cu"
#include "SpecularRefractionBSDF.cu"
#include "LambertianBSDF.cu"
#include "../textures/Texture.cu"

#include "../transformations/SurfaceInteraction.cu"

class PhongMaterial : public Material
{
public:
    Spectrum ka, kd, ks, ke;
    Float ni, ns, roughness;

    SpecularReflectionBSDF reflectionBSDF;
    LambertianBRDF lambertianBRDF;

    Texture *text;
    bool transparent, hasIndirectLight;
    
    // Generic material for MTL files
    PhongMaterial(Spectrum _ka, Spectrum _kd, 
        Spectrum _ks, Spectrum _ke, 
        Float _ni, Float _ns,
        Texture *texture=nullptr,
        bool _transparent = false,
        bool _hasIndirect = true)
    : reflectionBSDF(_ks, _ni),
        lambertianBRDF(_kd, texture)
    {
        this->text = texture;

        this->ka = _ka;

        this->kd = _kd;
        this->ks = _ks;
        this->ke = _ke;

        this->ni = _ni;
        this->ns = _ns;

        this->transparent = _transparent;
        this->hasIndirectLight = _hasIndirect;
    }

    PhongMaterial()
    : PhongMaterial(Spectrum(0), Spectrum(0), Spectrum(0), Spectrum(0), 1, 1, nullptr, false)
    {}

    bool isScattering() const override
    {
        Float max = this->kd.maxComponent();
        return max > 0;
    }

    bool isEmissive() const override
    {
        Float max = this->ke.maxComponent();
        return max > 0;
    }

    bool isSpecular() const override
    {
        Float max = this->ks.maxComponent();
        return max > 0;
    }

    bool isTransparent() const override
    {
        return transparent;
    }

    bool hasIndirect() const override
    {
        return hasIndirectLight;
    }

    Spectrum getEmittedRadiance(const SurfaceInteraction &interaction) const override
    {
        return ke;
    }

    // Phong reflection model
    Spectrum lightReflected(const Vector3f &cameraDirection, 
            const Vector3f &sampleDirection,
            Point2f localSamplePoint) const
    {
        // Calculate the reflection direction of the incoming light (sample direction)
        Vector3f reflectedDirection (-sampleDirection.x, -sampleDirection.y, sampleDirection.z);

        // Calculate the ambient, diffuse, and specular components
        Spectrum diffuse = lambertianBRDF.lightReflected(cameraDirection, sampleDirection, localSamplePoint);

        Spectrum specular;
        Float specularCosine = cosine(cameraDirection, reflectedDirection);
        if (cosine(sampleDirection) > 0)
            specular = ks * pow(max(0, specularCosine), ns);


        // Calculate the final reflected light spectrum
        Spectrum reflectedLight = diffuse + specular;
        return reflectedLight;
    }

    Spectrum getAmbientScattering() const override
    {
        return ka;
    }

    Float pdf(const Vector3f &cameraDirection, const Vector3f &sampleDirection) const override
    {
        if (sameHemisphere(cameraDirection, sampleDirection))
            return Sampling::cosineHemispherePdf(sampleDirection);
        else
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

        Point2f sample = sampler.get2Dsample();
        sampleDirection = Sampling::cosineSampleHemisphere(sample);

        Float samplePdf = Sampling::cosineHemispherePdf(sampleDirection); 

        if (hasIndirect())
            return lightReflected(cameraDirection, sampleDirection, localSamplePoint) / (samplePdf); 
        else
            return Spectrum(0);
    }

    Material* toGPU() const
    {
        throw std::runtime_error("MTLMaterial toGPU not implemented");
        return nullptr;
    }
};