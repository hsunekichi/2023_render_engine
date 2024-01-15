#pragma once

#include "Film.cu"

#include "../geometry/Point.cu"
#include "../geometry/Vector.cu"
#include "../geometry/Medium.cu"
#include "../geometry/Ray.cu"
#include "../geometry/RayDifferential.cu"

#include "../cuda_libraries/math.cu"
#include "../cuda_libraries/types.h"

#include "../transformations/Transform.cu"
#include "../transformations/Interaction.cu"

#include "../radiometry/Spectrum.cu"


struct CameraSample 
{
    Point2f pFilm;
    Point2f pLens;
    Float time;
};

class Camera
{
    public:
    
    Transform cameraToWorld;
    Film *film;
    Medium *medium;
    
    Camera(const Transform &cameraToWorld, 
        Film *film, Medium *medium)
        : cameraToWorld(cameraToWorld)
    {
        this->film = film;
        this->medium = medium;
    }

    virtual ~Camera() {}

    virtual Float generateRay(const CameraSample &sample,
        Ray &ray) const = 0;
    
    virtual Float generateRayDifferential(const CameraSample &sample,
        RayDifferential &rayDifferential) const
    {
        Float rayPonderation = generateRay(sample, rayDifferential);

        // Shift one pixel on x and calculate ray
        CameraSample sshift = sample;
        sshift.pFilm.x++;
        Ray rayX;

        Float ponderationX = generateRay(sshift, rayX);

        if (ponderationX == 0) // If radiance ponderation is 0
            return 0;

        rayDifferential.rxOrigin = rayX.origin;
        rayDifferential.rxDirection = rayX.direction;
        
        // Shift one pixel on y and calculate ray
        sshift.pFilm.x--;
        sshift.pFilm.y++;
        Ray rayY;

        Float ponderationY = generateRay(sshift, rayY);

        if (ponderationY == 0)   // If radiance ponderation is 0
            return 0;

        rayDifferential.ryOrigin = rayY.origin;
        rayDifferential.ryDirection = rayY.direction;
        
        rayDifferential.hasDifferentials = true;

        return rayPonderation;     
    }

    Film* getFilm() const
    {
        return film;
    }

    //virtual Spectrum we(const Ray &ray, Point2f *pRaster2 = nullptr) const = 0;
    
    //virtual void pdf_we(const Ray &ray, Float *pdfPos, Float *pdfDir) const = 0;

    //virtual Spectrum sample_wi(const Interaction &ref, const Point2f &u,
    //    Vector3f *wi, Float *pdf, Point2f *pRaster,
    //    VisibilityTester *vis) const = 0;
};