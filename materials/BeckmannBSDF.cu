#pragma once


#include "Material.cu"
#include "SpecularReflectionBSDF.cu"
#include "SpecularRefractionBSDF.cu"
#include "LambertianBSDF.cu"
#include "../textures/Texture.cu"

#include "../transformations/SurfaceInteraction.cu"

class BeckmannBSDF : public BSDF
{
public:
    Spectrum kd;
    Float alphaX, alphaY;
    Spectrum eta, k;

    Texture *text;

    // Generic material for MTL files
    BeckmannBSDF(Spectrum _kd, 
        Spectrum _eta, Spectrum _k,
        Float roughnessX, Float roughnessY, 
        Texture *texture=nullptr)
    {
        this->text = texture;
        this->kd = _kd;

        this->alphaX = roughnessToAlpha(roughnessX);
        this->alphaY = roughnessToAlpha(roughnessY);

        this->eta = _eta;
        this->k = _k;
    }

    static Float roughnessToAlpha(Float roughness) 
    {
        return sqrt(roughness);
    }

    bool effectivelySmooth() const {
        return max(alphaX, alphaY) < 1e-3f;
    }


    BeckmannBSDF()
    : BeckmannBSDF(Spectrum(0), Spectrum(0), Spectrum(0), 1, 1, nullptr)
    {}

    Spectrum lightReflected(const Vector3f &cameraDirection, 
            const Vector3f &sampleDirection,
            Point2f localSamplePoint) const override
    {
        if (!sameHemisphere(cameraDirection, sampleDirection)) 
           return Spectrum(0);

        if (effectivelySmooth()) 
            return Spectrum(0);

        Spectrum kd_txt = kd;
        if (text != nullptr)
            kd_txt = text->getColor(localSamplePoint);

        Float cosThetaO = absCosine(cameraDirection);
        Float cosThetaI = absCosine(sampleDirection);
        Vector3f halfVector = sampleDirection + cameraDirection;

        // Handle degenerate cases for microfacet reflection 
        if (cosThetaI == 0 || cosThetaO == 0) 
            return Spectrum(0.);

        if (halfVector.x == 0 && halfVector.y == 0 && halfVector.z == 0) 
            return Spectrum(0.);
        
        halfVector = normalize(halfVector);

        Spectrum fresnel = fresnelConductor(cosine(sampleDirection, halfVector), eta, k);

        Float distribution = beckmannDistribution(halfVector, alphaX, alphaY);

        Float lambda1 = beckmannDistributionLambda(cameraDirection, alphaX, alphaY);
        Float lambda2 = beckmannDistributionLambda(sampleDirection, alphaX, alphaY);
        Float attenuation = geometricAttenuation(lambda1, lambda2);

        Spectrum result = kd_txt * distribution * attenuation * fresnel /
            (4 * cosThetaI * cosThetaO);

        return result * absCosine(sampleDirection);
    }

    Vector3f sampleHalfBeckman(const Vector3f &cameraDirection, Sampler &sampler) const
    {
        Float tan2Theta, phi;
        Float logSample = log(1 - sampler.get1Dsample());

        if (isinf(logSample)) 
            logSample = 0;

        tan2Theta = -alphaX * alphaY * logSample;
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
        Float halfPdf = beckmannDistribution(sampleHalfBeckman, alphaX, alphaY) * absCosine(sampleHalfBeckman);
        return halfPdf;
    }


    inline Float pdf(const Vector3f &cameraDirection, const Vector3f &sampleDirection) const override
    {
        Vector3f halfVector = sampleDirection + cameraDirection;
        return beckmannPDF(halfVector) / (4 * absCosine(cameraDirection, halfVector));
    }

    Spectrum sampleDirection(
            Vector3f cameraDirection, 
            Sampler &sampler,
            Vector3f &sampleDirection,
            Point2f localSamplePoint) const override
    {     
        if (effectivelySmooth()) 
        {
            sampleDirection = Vector3f(-cameraDirection.x, -cameraDirection.y, cameraDirection.z);
            Spectrum F = fresnelConductor(absCosine(sampleDirection), eta, k);
            //F /= absCosine(sampleDirection);

            return F;
        }

        Point2f sample = sampler.get2Dsample();
        Vector3f halfVector = sampleHalfBeckman(cameraDirection, sampler);
        sampleDirection = reflect(cameraDirection, halfVector);

        Float samplePdf = beckmannPDF(halfVector) / (4 * absCosine(cameraDirection, halfVector));
        return lightReflected(cameraDirection, sampleDirection, localSamplePoint) / samplePdf; 
    }

    Material* toGPU() const
    {
        throw std::runtime_error("MTLMaterial toGPU not implemented");
        return nullptr;
    }
};