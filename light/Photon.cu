#pragma once

#include "../radiometry/Spectrum.cu"
#include "../geometry/Point.cu"
#include "../geometry/Vector.cu"
#include "../geometry/Ray.cu"


class Photon
{
    public:

    Ray ray;
    Point3f location;
    Spectrum radiance;
    Vector3f surfaceNormal;

    Point2f surfacePoint;

    int shapeId;

    __host__ __device__
    Photon(Ray ray, Point3f location,
            Vector3f normal, 
            Spectrum radiance, 
            int shapeId,
            Point2f surfacePoint)
    {
        this->ray = ray;
        this->location = location;
        this->radiance = radiance;
        this->shapeId = shapeId;
        this->surfaceNormal = normal;
        this->surfacePoint = surfacePoint;
    }

    __host__ __device__
    Photon()
    {
        this->ray = Ray();
        this->location = Point3f();
        this->radiance = Spectrum();
        this->shapeId = -1;
        this->surfaceNormal = Vector3f();
        this->surfacePoint = Point2f();
    }
};

// Swap two photons
__host__ __device__ 
void swap(Photon& a, Photon& b)
{
    Photon temp = a;
    a = b;
    b = temp;
}


/* 
    An additional struct that allows the KD-Tree to access your photon position
*/
struct PhotonPosition 
{
    Float operator()(const Photon& ph, size_t i) const {
        return ph.location[i];
    }
};