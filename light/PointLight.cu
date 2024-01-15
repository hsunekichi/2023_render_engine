#pragma once

#include "Light.cu"

#include <thrust/host_vector.h>

#include "../geometry/Vector.cu"
#include "../geometry/Point.cu"
#include "../transformations/Transform.cu"

#include "../cuda_libraries/geometricMath.cu"
#include "../sampling/Sampler.cu"


class PointLight : public Light
{
protected:
    Transform lightToWorld, worldToLight;

public: 
    PointLight(Transform lightToWorld, const Spectrum& color, Float intensity) 
    : Light(color, intensity), lightToWorld(lightToWorld), worldToLight(inverse(lightToWorld))
    {}

    Vector3f getDirection(const Point3f& point) const override
    {
        return normalize(Point3f() - worldToLight(point));
    }
 
    Float getDistance(const Point3f& point) const override
    {
        return distance(Point3f(), worldToLight(point));
    }
 
    inline Point3f getPosition() const override
    {
        return lightToWorld(Point3f());
    }

    inline Spectrum power() const override
    {
        return color * intensity * 4.0f * PI;
    }

    Spectrum getRadiance(const Point3f& lightPoint, const Point3f &interactionPoint) const override
    {
        Float lengthSquared = distanceSquared(lightPoint, interactionPoint);
        return color*intensity*lightDecay(lengthSquared);
    }

    Spectrum getEmittedRadiance() const
    {
        return color*intensity;
    }

    Spectrum sampleLight(Interaction *interaction, 
                        Point2f lightSampleGenerator, 
                        Vector3f &surfaceToLight, 
                        Float &lightPdf,
                        Point3f &samplePoint) const override
    {
        samplePoint = getPosition();
        surfaceToLight = normalize(samplePoint - interaction->worldPoint);
        lightPdf = 1.0f;
        return getRadiance(samplePoint, interaction->worldPoint);
    }

    Spectrum sampleLight(Point2f randomSample, 
                        Ray &outgoingRay) const override
    {
        Point3f lightPoint = getPosition();
        Vector3f direction = Sampling::uniformSampleSphere(randomSample);
        Float samplePdf = Sampling::uniformSpherePdf();

        outgoingRay = Ray(lightPoint, direction);
        return getEmittedRadiance() / samplePdf;
    }


    thrust::host_vector<Photon> sampleLight(Sampler *sampler, int nSamples) const
    {
        thrust::host_vector<Photon> samples(nSamples);
        Point3f wp = getPosition();

        for (int i = 0; i < nSamples; i++)
        {
            Point2f randomSample = sampler->get2Dsample();
            Ray outgoingRay;
            Spectrum l = sampleLight(randomSample, outgoingRay) / nSamples;

            samples[i] = Photon(outgoingRay, wp, Vector3f(), l, -1, Point2f());
        }

        return samples;
    }
};