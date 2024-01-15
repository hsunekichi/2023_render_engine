#pragma once


#include "Material.cu"
#include "SpecularReflectionBSDF.cu"
#include "SpecularRefractionBSDF.cu"
#include "PhongMaterial.cu"
#include "../textures/Texture.cu"

#include "../transformations/SurfaceInteraction.cu"

class SSMaterial : public Material
{
public:
    Spectrum kd, ka;
    Float eta, roughness;
    
    PhongMaterial surfaceMaterial;

    Texture *text;

    bool transparent;
    Spectrum sigmaS, sigmaA;
    Float g;

    
    // Generic material for MTL files
    SSMaterial(Spectrum _ka, Spectrum _kd,
        Spectrum _ks,
        Float _eta,
        Float ns,
        Texture *texture=nullptr,
        bool _transparent = false,
        Spectrum sigmaS = Spectrum(0),
        Spectrum sigmaA = Spectrum(0),
        Float g=0)
    :   surfaceMaterial(_ka, _kd, _ks, Spectrum(0), _eta, ns, texture, transparent)
    {
        this->text = texture;
        this->kd = _kd;
        this->ka = _ka;

        this->eta = _eta;

        this->transparent = _transparent;
        this->sigmaS = sigmaS;
        this->sigmaA = sigmaA;
        this->g = g;
    }

    SSMaterial()
    : SSMaterial(Spectrum(0), Spectrum(0), Spectrum(0), 1, 0, nullptr, false, Spectrum(0), Spectrum(0), 0)
    {}

    bool isScattering() const override
    {
        Float max = this->kd.maxComponent();
        return max > 0;
    }

    bool isTransparent() const override
    {
        return transparent;
    }

    bool isSubsurfaceScattering() const override 
    {
        return true;
    }

    Spectrum getAmbientScattering() const override
    {
        return ka;
    }


    /*
    Spectrum lightReflected(const Vector3f &cameraDirection, 
            const Vector3f &sampleDirection,
            Point3f localSamplePoint) const
    {
        Spectrum diffuse = (28/(23*PI)) * kd *
            (Spectrum(1) - ks) * 
            (1 - pow5(1 - 0.5 * absCosine(sampleDirection))) *
            (1 - pow5(1 - 0.5 * absCosine(cameraDirection)));

        Vector3f halfVector = sampleDirection + cameraDirection;

        if (halfVector.x == 0 && halfVector.y == 0 && halfVector.z == 0) 
            return Spectrum(0);

        halfVector = normalize(halfVector);

        Float beckmann = beckmannDistribution(halfVector, roughness, roughness);
        Spectrum fresnel = schlickFresnel(cosine(sampleDirection, halfVector), ks);
        Float maxCosine = max(absCosine(sampleDirection), absCosine(cameraDirection));

        Spectrum specular = fresnel*beckmann / 
            (4 * absDot(sampleDirection, halfVector) * maxCosine);

        // Weighted sum of diffuse and specular
        Spectrum result = diffuse + specular;

        return result;
    }
    */


    Spectrum singleScattering(
            const Vector3f &cameraDirection, 
            Sampler &sampler,
            Vector3f &sampleDirection,
            Point2f localSamplePoint,
            Point3f &outgoingPoint,
            Vector3f L) const
    {
        // Handle photon scattering (No BSSRDF, it needs the photons precomputed)
        if (L.length() == 0)
            return Spectrum(0);

        // Get refraction direction entering the material
        Float etaI = 1, etaT = eta;
        Float invEta = etaI / etaT;

        Spectrum sigmaT = sigmaA + sigmaS;
        Vector3f To = normalize(refract(-cameraDirection, Vector3f(0, 0, 1), etaI, etaT));
        if (To.x == 0 && To.y == 0 && To.z == 0)
            return Spectrum(0);

        Float sp_o = -log(sampler.get1Dsample()) / sigmaT.maxComponent();
        Point3f p_o = Point3f(0, 0, 0) + sp_o * To;
        Ray shadowRay = Ray(p_o, L);
        
        // Get intersection point of shadowRay with surface
        Float Pi_offset = -shadowRay.origin.z / shadowRay.direction.z;
        Point3f Pi = shadowRay(Pi_offset);
        outgoingPoint = Pi;

        Float si = distance(Pi, p_o);
        Float Lcos = absCosine(L, Vector3f(0, 0, 1));
        Float sp_i = si * Lcos / sqrt(1 - pow2(invEta) * (1 - pow2(Lcos)));

        Vector3f Ri, Ti;
        Float Kri = fresnelDielectric(cosine(L, Vector3f(0, 0, 1)), etaI, etaT);
        Float Kti = 1 - Kri;

        Ti = normalize(refract(L, Vector3f(0, 0, 1), etaI, etaT));

        Float g2 = pow2(g);
        Float phase = (1-g2) / pow(1+2*g*cosine(Ti, To) + g2, 1.5);

        sampleDirection = Ti;

        Float G = absCosine(cameraDirection) / absCosine(sampleDirection);
        Spectrum sigmaTc = sigmaT + sigmaT*G;

        Spectrum result = exp(-sp_i*sigmaT) / sigmaTc * phase * Kti;


        /********** Compute the correction of the montecarlo estimation (pdf?) *********/
        // Compute isotropic phase function
        Spectrum _sigmaS = (1 - g) * sigmaS;
        Spectrum _sigmaT = _sigmaS + sigmaA;
        Spectrum _alpha = _sigmaS / _sigmaT;
        // Effective transport coefficient
        Spectrum sigmaTr = sqrt(3 * sigmaA * _sigmaT);
        Spectrum zr = sqrt(3*(1-_alpha)) / sigmaTr;
        Float r = distance(Point3f(0, 0, 0), Pi);

        Spectrum distanceR = sqrt(pow2(r) + pow2(zr));
        Spectrum C1 = zr * (sigmaTr + 1/distanceR);

        Float cosWi = cosine(sampleDirection);
        Float cosWo = cosine(cameraDirection);

        // Assuming Ft is a function that returns the Fresnel term
        double Ft_o = 1 - fresnelDielectric(cosWo, 1.0, eta);
        double Ft_i = 1 - fresnelDielectric(cosWi, 1.0, eta);

        return result * PI * C1 * Ft_o * Ft_i;
    }


    inline Float pdf(const Vector3f &cameraDirection, const Vector3f &sampleDirection) const override
    {
        if (sameHemisphere(cameraDirection, sampleDirection))
            return Sampling::cosineHemispherePdf(sampleDirection);
        else
            return 0;
    }


    inline Float pdfCamera(const Vector3f &cameraDirection, const Vector3f &sampleDirection) const
    {
        return 1;
    }


    // Phong reflection model
    Spectrum lightReflected(const Vector3f &cameraDirection, 
            const Vector3f &sampleDirection,
            Point2f localSamplePoint) const
    {
        Spectrum radiance = surfaceMaterial.lightReflected(cameraDirection, sampleDirection, localSamplePoint);
        return radiance;
    }

    // Phong reflection model
    Spectrum lightReflectedCamera(const Vector3f &cameraDirection, 
            const Vector3f &sampleDirection,
            Point2f localSamplePoint) const
    {
        Spectrum(0);
        Spectrum radiance = surfaceMaterial.lightReflected(cameraDirection, sampleDirection, localSamplePoint);
        return radiance;
    }


    Spectrum sampleDirection(
            const Vector3f &cameraDirection, 
            Sampler &sampler,
            Vector3f &sampleDirection,
            Point2f localSamplePoint,
            Point3f &outgoingPoint,
            Vector3f L) const override
    {
        return surfaceMaterial.sampleDirection(cameraDirection, sampler, 
                sampleDirection, localSamplePoint, outgoingPoint, L);
    }

    // Special sampling for rays coming directly from the camera
    Spectrum sampleCameraRay(
        const Vector3f &cameraDirection, 
        Sampler &sampler,
        Vector3f &sampleDirection,
        Point2f localSamplePoint,
        Point3f &outgoingPoint,
        Vector3f L) const
    {
        return Spectrum(0);
        // return surfaceMaterial.sampleDirection(cameraDirection, sampler, sampleDirection, 
        //         localSamplePoint, outgoingPoint, L) / 2;
        return singleScattering(cameraDirection, sampler, sampleDirection, localSamplePoint, outgoingPoint, L);
    }

    void getBSSDFparameters(
        Spectrum &kd,
        Spectrum &sigmaS,
        Spectrum &sigmaA,
        Float &g,
        Float &eta,
        Point2f surfacePoint) const override
    {
        sigmaS = this->sigmaS;
        sigmaA = this->sigmaA;
        g = this->g;
        eta = this->eta;

        kd = this->kd;

        if (text != nullptr)
            kd *= text->getColor(surfacePoint);
    }

    Material* toGPU() const
    {
        throw std::runtime_error("MTLMaterial toGPU not implemented");
        return nullptr;
    }
};