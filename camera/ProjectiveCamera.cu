#pragma once

#include "Camera.cu"
#include "../transformations/Transform.cu"

#include "../geometry/BoundingBox.cu"
#include "../geometry/Vector.cu"
#include "../geometry/Ray.cu"
#include "../geometry/Point.cu"

#include "../sampling/Sampler.cu"

#include "Film.cu"


class ProjectiveCamera : public Camera
{
    protected:

    Transform cameraToScreen, rasterToCamera;
    Transform screenToRaster, rasterToScreen;
    Float lensRadius, focalDistance;
    
    public:

    ProjectiveCamera(const Transform &cameraToWorld, 
                const Transform &cameraToScreen, 
                const Bound2f &screenWindow,
                Float lensRadius, Float focalDistance,
                Film *film, Medium *medium)
           : Camera(cameraToWorld, film, medium),
             cameraToScreen(cameraToScreen) 
    {
        this->lensRadius = lensRadius;
        this->focalDistance = focalDistance;

        Point2f origin(0.0, 0.0);
        Vector2f direction = origin - screenWindow.pMin;

        // First, translate the upper left corner to the origin
        Transform move_to_origin = Transformations::translate(Vector3f(-screenWindow.pMin.x, -screenWindow.pMax.y, 0));

        // Second, scale the 2d coordinates between 0 to 1
        Transform scale_to_0_1 = Transformations::scale(1 / (screenWindow.pMax.x - screenWindow.pMin.x),
                1 / (screenWindow.pMin.y - screenWindow.pMax.y), 1);

        // Third, scale the 2d coordinates to the raster resolution
        Transform scale_to_resolution = Transformations::scale(resolution.x, resolution.y, 1);


        // Screen ransformations. Y coordinate is inverted because the raster appears inverted
        screenToRaster = scale_to_resolution(scale_to_0_1(move_to_origin));
            
        rasterToScreen = inverse(screenToRaster);
        rasterToCamera = inverse(cameraToScreen) * rasterToScreen;
    }

    // Copy constructor
    ProjectiveCamera(const ProjectiveCamera &camera)
    : Camera(camera.cameraToWorld, camera.film, camera.medium)
    {
        this->lensRadius = camera.lensRadius;
        this->focalDistance = camera.focalDistance;
        this->cameraToScreen = camera.cameraToScreen;
        this->screenToRaster = camera.screenToRaster;
        this->rasterToScreen = camera.rasterToScreen;
        this->rasterToCamera = camera.rasterToCamera;
    }


    Float generateRay(const CameraSample &sample, Ray &ray) const
    {
        // Get film sample position
        Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);

        // Transform raster to camera space
        Point3f pCamera = rasterToCamera(pFilm);

        // Generate ray from camera origin through raster sample
        ray = Ray(Point3f(0, 0, 0), normalize(pCamera.toVector()));
        ray.cameraRay = true;

        // Modify ray for depth of field
        if (lensRadius > 0) 
        {
            // Sample point on lens
            Point2f pLens = lensRadius * Sampling::concentricSampleDisk(sample.pLens);

            // Compute point on plane of focus
            Float ft = focalDistance / pCamera.z;
            Point3f pFocus = pCamera * ft;

            // Update the effect of the lens in the ray
            pCamera = Point3f(pLens.x, pLens.y, 0);
            ray.origin = Point3f(pLens.x, pLens.y, 0);
            ray.direction = normalize(pFocus - pCamera);
        } 

        // Interpolate between shutter open and close time positions
        ray.medium = medium;
        ray = cameraToWorld(ray);

        return 1;
    }
};
