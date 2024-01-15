#pragma once

#include "Texture.cu"


class SphericalTexture : public Texture
{
    public:

    SphericalTexture(std::string filename) : Texture(filename) {}

    //Spectrum getColor(Point3f point) const override
    //{
    //    Point2f samplePoint = Texture::projectSphere(point);
    //    return Texture::getColorPlane(samplePoint);
    //}
};