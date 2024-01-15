#pragma once

#include <thrust/host_vector.h>

#include "../radiometry/Spectrum.cu"
#include "../geometry/Point.cu"
#include "../geometry/Vector.cu"
#include "../transformations/Interaction.cu"
#include "Photon.cu"

class Light
{
    protected:
    Spectrum color;
    Float intensity;

    public:
    Light(const Spectrum& color, Float intensity) 
    : color(color), intensity(intensity) 
    {}

    virtual Vector3f getDirection(const Point3f& point) const = 0;


    virtual Float getDistance(const Point3f& point) const = 0;

    virtual Point3f getPosition() const = 0;
    
    virtual Spectrum power() const = 0;

    virtual Spectrum getRadiance(const Point3f& lightPoint, const Point3f &interactionPoint) const = 0;


    // Returns a shadow ray from the light to the interaction point
    virtual Spectrum sampleLight(Interaction *interaction, 
                        Point2f lightSampleGenerator, 
                        Vector3f &incidentDirection, 
                        Float &lightPdf,
                        Point3f &samplePoint) const = 0;

    virtual Spectrum sampleLight(Point2f randomSample, 
                        Ray &outgoingRay) const = 0;

    virtual thrust::host_vector<Photon> sampleLight(Sampler *sampler, int nSamples) const = 0;
};
