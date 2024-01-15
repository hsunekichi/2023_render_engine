#pragma once


#include "Material.cu"
#include "LambertianBSDF.cu"

#include "../transformations/SurfaceInteraction.cu"


class PlainMaterial : public Material
{
public:
    Spectrum ke, kd;
    Texture *texture;

    bool transparent;
 
    PlainMaterial(const Spectrum& _kd,
        const Spectrum& _ke = Spectrum(0),
        Texture *_texture = nullptr,
        bool _transparent = false) 
    {
        this->ke = _ke;
        this->kd = _kd;
        this->transparent = _transparent;
        this->texture = _texture;
    }


    Spectrum lightReflected(const Vector3f &cameraDirection, 
            const Vector3f &sampleDirection,
            Point2f localSamplePoint) const override
    {
        return Spectrum(0);

        /*
        if (texture == nullptr)
            return kd * INV_PI;
        else
            return kd * texture->getColor(localSamplePoint) * INV_PI;
        */
    }

    Float pdf(const Vector3f &cameraDirection, const Vector3f &sampleDirection) const override
    {
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
        sampleDirection = Vector3f(0, 0, 0);
        return Spectrum(0);
    }

    Spectrum getEmittedRadiance(const SurfaceInteraction &interaction) const override
    {
        if (interaction.worldCameraRay.isCameraRay())
        {
            if (texture != nullptr)
                return kd * texture->getColor(interaction.surfacePoint);
            else
                return kd;
        }
        else {
            return ke;
        }
    }


    bool isScattering() const override
    {
        return true;
    }

    bool isSpecular() const
    {
        return false;
    }

    bool isEmissive() const override
    {
        return true;
    }

    bool isTransparent() const override
    {
        return transparent;
    }

    virtual Material* toGPU() const override
    {
        throw std::runtime_error("PlainMaterial::toGPU() not implemented");
    }
};

