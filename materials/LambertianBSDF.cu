#pragma once

#include "BSDF.cu"
#include "../radiometry/Spectrum.cu"
#include "../sampling/Sampler.cu"

#include "../textures/Texture.cu"


class LambertianBRDF : public BSDF
{
public:
    Spectrum color;
    Texture *texture;

    LambertianBRDF(const Spectrum& color, Texture *texture=nullptr)
    {
        this->color = color;
        this->texture = texture;
    }

    Spectrum BRDF(Point2f localPoint) const
    {
        Spectrum diffuse = color * INV_PI;

        if (texture != nullptr)
            diffuse *= texture->getColor(localPoint);

        return diffuse;
    }

    Spectrum lightReflected(const Vector3f& cameraDirection, 
            const Vector3f& sampleDirection,
            Point2f localPoint) const
    {
        Spectrum diffuse = BRDF(localPoint) * absCosine(sampleDirection);
        return diffuse;
    }

    inline Float pdf(const Vector3f& cameraRay, const Vector3f& incomingRay) const
    {
        if (sameHemisphere(cameraRay, incomingRay))
            return Sampling::cosineHemispherePdf(incomingRay);
        else {
            //std::cout << "LambertianBRDF::pdf: rays are not in the same hemisphere" << std::endl;
            return 0;
        }
    }

    virtual Spectrum sampleDirection(
            Vector3f cameraDirection, 
            Sampler &sampler,
            Vector3f &sampleDirection,
            Point2f localPoint) const override
    {
        Point2f sample = sampler.get2Dsample();

        sampleDirection = Sampling::cosineSampleHemisphere(sample);
        Float samplePdf = Sampling::cosineHemispherePdf(sampleDirection);

        //if (sampler.get1Dsample() < 0.5)
        //    sampleDirection = Vector3f(0.1, 0.1, 1);
        //else
        //    sampleDirection = Vector3f(0.1, -0.1, 1);

        Spectrum diffuse = color;

        if (texture != nullptr)
            diffuse *= texture->getColor(localPoint);

        Spectrum result = diffuse; // * absCosine(sampleDirection) / samplePdf;

        return result;
    }

/*
    Float pdf(const Vector3f& wo, const Vector3f& wi) const
    {
        return cosineHemispherePdf(wi.z);
    }
*/

};