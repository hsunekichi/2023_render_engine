#pragma once


#include "Material.cu"
#include "SpecularReflectionBSDF.cu"
#include "SpecularRefractionBSDF.cu"
#include "LambertianBSDF.cu"
#include "../textures/Texture.cu"

#include "../transformations/SurfaceInteraction.cu"

class TrowbridgeBasicBSDF : public BSDF
{
public:
    Spectrum kd;
    Float alphaX, alphaY;
    Spectrum eta;
    Spectrum k;

    Texture *text;

    // Generic material for MTL files
    TrowbridgeBasicBSDF(Spectrum _kd, 
        Spectrum _eta, Spectrum _k,
        Float roughnessX, Float roughnessY,
        Texture *texture=nullptr)
    {
        this->text = texture;

        this->kd = _kd;

        // Silver
        //this->eta = Spectrum(0.167, 0.145, 0.135);
        //this->k = Spectrum(4.28, 3.19, 2.38);

        // gold
        //this->eta = Spectrum(0.169, 0.421, 0.45);
        //this->k = Spectrum(3.88, 2.343, 1.770);

        this->eta = _eta;
        this->k = _k;

        //this->alpha = roughnessToAlpha(roughness);
        this->alphaX = roughnessToAlpha(roughnessX);
        this->alphaY = roughnessToAlpha(roughnessY);
    }

    TrowbridgeBasicBSDF()
    : TrowbridgeBasicBSDF(Spectrum(0), Spectrum(0), Spectrum(0), 0, 0, nullptr)
    {}

    bool effectivelySmooth() const {
        return max(alphaX, alphaY) < 1e-3f;
    }

    static Float roughnessToAlpha(Float roughness) 
    {
        return sqrt(roughness);
    }

    void regularize() 
    {
        if (alphaX < 0.3f) 
            alphaX = clamp(2 * alphaX, 0.1f, 0.3f);

        if (alphaY < 0.3f) 
            alphaY = clamp(2 * alphaY, 0.1f, 0.3f);
    }


    Float distribution(const Vector3f &halfVector) const
    {
        Float tan2Th = tan2Theta(halfVector);
        
        if (isinf(tan2Th)) 
            return 0;

        const Float cos4Theta = pow2(cos2Theta(halfVector));

        Float e = tan2Th * (pow2(cosPhi(halfVector) / alphaX) + 
                  pow2(sinPhi(halfVector) / alphaY));

        return 1 / (PI * alphaX * alphaY * cos4Theta * pow2(1 + e));
    }

    Float G(Vector3f v) const
    {
        return 1 / (1 + lambda(v));
    }

    Float G(Vector3f v1, Vector3f v2) const
    {
        return 1 / (1 + lambda(v1) + lambda(v2));
    }

    // Distribution of the visible normals
    Float distribution(Vector3f cameraDir, Vector3f half) const {
        Float result = G(cameraDir) 
                / absCosine(cameraDir) * distribution(half) * absCosine(cameraDir, half);

        return result;
    }

    Float lambda(const Vector3f &rayDir) const
    {
        Float tan2Th = tan2Theta(rayDir);

        if (isInf(tan2Th)) 
            return 0;

        Float alpha2 = pow2(cosPhi(rayDir) * alphaX) + pow2(sinPhi(rayDir) * alphaY);
        return (sqrt(1 + alpha2 * tan2Th) - 1) / 2;
    }

    Float trowbridgePdf(const Vector3f &cameraDirection, const Vector3f &halfVector) const
    {
        return distribution(cameraDirection, halfVector);
    }

    Vector3f sampleHalf(Vector3f cameraRay, Point2f sample) const
    {
        // Transform to hemispherical config
        Vector3f half = normalize(Vector3f(alphaX * cameraRay.x, alphaY * cameraRay.y, cameraRay.z));
        
        // Make sure it looks outside
        if (half.z < 0)
            half = -half;

        // Find visible basis
        Vector3f T1;
        if (half.z < 0.99999f)
            T1 = normalize(cross(Vector3f(0, 0, 1), half));
        else
            T1 = Vector3f(1, 0, 0);
                                
        Vector3f T2 = cross(half, T1);

        sample = 2 * sample - 1;
        Point2f p = Sampling::uniformSampleDisk(sample);

        // Warp projection for visible normal
        Float h = sqrt(1 - pow2(p.x));
        p.y = lerp((1 + half.z) / 2, h, p.y);

        // Project to hemisphere
        Float pz = sqrt(max((Float)0, 1 - p.toVector().lengthSquared()));
        Vector3f nh = p.x * T1 + p.y * T2 + pz * half;
        Vector3f finalDirection = normalize(Vector3f(alphaX * nh.x, alphaY * nh.y, max(1e-6, nh.z)));
        
        return finalDirection;
    }


    Spectrum lightReflected(const Vector3f &cameraDirection, 
            const Vector3f &sampleDirection,
            Point2f localSamplePoint) const override
    {
        Vector3f cameraDir = normalize(cameraDirection);
        Vector3f sampleDir = normalize(sampleDirection);

        if (!sameHemisphere(cameraDir, sampleDir)) 
           return Spectrum(0);

        if (effectivelySmooth()) 
            return Spectrum(0);

        Spectrum kd_txt = kd;
        if (text != nullptr)
            kd_txt = text->getColor(localSamplePoint);

        // Compute cosines and half vector
        Float cosTheta_o = absCosine(cameraDir);
        Float cosTheta_i = absCosine(sampleDir);

        if (cosTheta_i == 0 || cosTheta_o == 0) 
            return Spectrum(0);
        
        Vector3f half = sampleDir + cameraDir;
        if (half.lengthSquared() == 0) 
            return Spectrum(0);

        half = normalize(half);

        // Compute fresnel        
        //Spectrum F = fresnelConductor(absCosine(cameraDir, half), eta, k);
        Spectrum F (1);
        Spectrum D = distribution(half);
        Spectrum att = G(cameraDir, sampleDir);

        Spectrum result = D * F * att / (4 * cosTheta_i * cosTheta_o);

        result = result * kd_txt * absCosine(sampleDir);

        return result;
    }

    Float pdf(const Vector3f &cameraDirection, const Vector3f &sampleDirection) const override
    {
        if (!sameHemisphere(cameraDirection, sampleDirection)) 
            return 0;

        if (effectivelySmooth()) 
            return 0;

        // Copmpute pdf of a rough conductor
        Vector3f half = cameraDirection + sampleDirection;
        if (half.lengthSquared() == 0) 
            return 0;

        half = faceForward(normalize(half), Vector3f(0, 0, 1)); 
        return trowbridgePdf(cameraDirection, half) / (4 * absCosine(cameraDirection, half));
    }


    Spectrum sampleDirection(
            Vector3f cameraDirection, 
            Sampler &sampler,
            Vector3f &sampleDirection,
            Point2f localSamplePoint) const override
    {     
        cameraDirection = normalize(cameraDirection);

        if (effectivelySmooth()) 
        {
            sampleDirection = Vector3f(-cameraDirection.x, -cameraDirection.y, cameraDirection.z);
            //Spectrum F = fresnelConductor(absCosine(sampleDirection), eta, k);
            //F /= absCosine(sampleDirection);

            return Spectrum(1);
        }

        // Sample microfacet
        Vector3f half = sampleHalf(cameraDirection, sampler.get2Dsample());
        sampleDirection = reflect(cameraDirection, half);

        //Point2f sample = sampler.get2Dsample();
        //sampleDirection = Sampling::cosineSampleHemisphere(sample);

        // Compute pdf
        Float pdf = trowbridgePdf(cameraDirection, half) / (4 * absCosine(cameraDirection, half));
        //pdf = Sampling::cosineHemispherePdf(sampleDirection);

        Spectrum result = lightReflected(cameraDirection, sampleDirection, localSamplePoint);

        result /= pdf;
        return result;
    }

    Material* toGPU() const
    {
        throw std::runtime_error("MTLMaterial toGPU not implemented");
        return nullptr;
    }
};