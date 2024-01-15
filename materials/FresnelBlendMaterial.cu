#pragma once


#include "Material.cu"
#include "SpecularReflectionBSDF.cu"
#include "SpecularRefractionBSDF.cu"
#include "LambertianBSDF.cu"
#include "../textures/Texture.cu"

#include "../transformations/SurfaceInteraction.cu"

class FresnelBlendMaterial : public Material
{
public:
    Spectrum ka, kd, ks, ke;
    Float ni, ns, alpha;

    SpecularReflectionBSDF reflectionBSDF;
    LambertianBRDF lambertianBRDF;

    Texture *text;
    bool hasSSS;

    
    // Generic material for MTL files
    FresnelBlendMaterial(Spectrum _ka, Spectrum _kd, 
        Spectrum _ks, Spectrum _ke, 
        Float _ni, Float _ns, Texture *texture=nullptr,
        bool _hasSSS=false)
    : reflectionBSDF(_ks, _ni),
        lambertianBRDF(_kd, texture)
    {
        this->text = texture;

        this->ka = _ka;

        this->kd = _kd;
        this->ks = _ks;
        this->ke = _ke;

        kd = Spectrum(1);
        ks = Spectrum(0);

        this->ni = _ni;
        this->ns = _ns;
        this->alpha = 0.5;

        this->hasSSS = _hasSSS;
    }

    FresnelBlendMaterial()
    : FresnelBlendMaterial(Spectrum(0), Spectrum(0), Spectrum(0), Spectrum(0), 1, 1, nullptr)
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

    Spectrum getEmittedRadiance(const SurfaceInteraction &interaction) const override
    {
        return ke;
    }

    Spectrum getAmbientScattering () const override
    {
        return ka;
    }

    Vector3f sampleHalfBeckman(const Vector3f &cameraDirection, Sampler &sampler) const
    {
        Float tan2Theta, phi;
        Float logSample = log(1 - sampler.get1Dsample());

        if (isinf(logSample)) 
            logSample = 0;

        tan2Theta = -alpha * alpha * logSample;
        phi = TAU * sampler.get1Dsample();    

        // Map sampled Beckmann angles to normal direction
        Float cosTheta = 1 / sqrt(1 + tan2Theta);
        Float sinTheta = sqrt(max((Float)0, 1 - cosTheta * cosTheta));
        Vector3f halfVector = sphericalDirection(sinTheta, cosTheta, phi);
        
        if (!sameHemisphere(cameraDirection, halfVector)) 
            halfVector = -halfVector;
        
        return halfVector;
    }

    Float beckmannPDF(const Vector3f &sampleHalfBeckman) const
    {
        return beckmannDistribution(sampleHalfBeckman, alpha, alpha) * absCosine(sampleHalfBeckman);
    }

    // Torrance material model
    Spectrum lightReflected(const Vector3f &cameraDirection, 
        const Vector3f &sampleDirection,
        Point2f localSamplePoint) const
    {
        Spectrum diffuse = (28.f/(23.f*PI)) * kd *
            (Spectrum(1) - ks) * 
            (1 - pow5(1 - 0.5 * absCosine(sampleDirection))) *
            (1 - pow5(1 - 0.5 * absCosine(cameraDirection)));

        Vector3f wh = sampleDirection + cameraDirection;

        if (wh.x == 0 && wh.y == 0 && wh.z == 0) 
            return Spectrum(0);

        wh = normalize(wh);

        Spectrum num = schlickFresnel(cosine(sampleDirection, wh), ks) * 
                beckmannDistribution(wh, alpha, alpha);

        Float denom = (4 * absCosine(sampleDirection, wh) * 
                max(absCosine(sampleDirection), absCosine(cameraDirection)));

        Spectrum specular = num / denom;

        return diffuse + specular;
    }
    

    inline Float pdf(const Vector3f &cameraDirection, const Vector3f &sampleDirection) const override
    {
        Vector3f halfVector = sampleDirection + cameraDirection;
        return beckmannPDF(halfVector);
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
        Float kdb = kd.brightness();
        Float ksb = ks.brightness();

        Float specularFraction = ksb / (kdb + ksb);
        Float specularP = clamp(specularFraction, 0.2, 0.8);

        if (rouletteSample < 0)
        {
            // Make the specular reflection direction
            sampleDirection = Vector3f(-cameraDirection.x, -cameraDirection.y, cameraDirection.z);
            
            return lightReflected(cameraDirection, sampleDirection, localSamplePoint);
        }
        else 
        {
            Point2f sample = sampler.get2Dsample();
            Vector3f halfVector = sampleHalfBeckman(cameraDirection, sampler);
            sampleDirection = halfVector - cameraDirection;

            Float samplePdf = pdf(cameraDirection, sampleDirection);
            return lightReflected(cameraDirection, sampleDirection, localSamplePoint) / (samplePdf); 
        }
    }

    Material* toGPU() const
    {
        throw std::runtime_error("MTLMaterial toGPU not implemented");
        return nullptr;
    }
};