#pragma once

#include "Interaction.cu"
#include "../shapes/Shape.cu"
#include "../materials/Material.cu"
#include "../materials/SSMaterial.cu"
#include "../materials/BSDF.cu"
#include "Transform.cu"

class SurfaceInteraction : public Interaction
{
    public:

    struct ShadingInfo
    {
        Normal3f normal;
        //Vector3f tangentU, tangentV;
        //Vector3f dndu, dndv;
    };


    //Point3f worldPoint; // Defined in Interaction
    Point2f surfacePoint;
    //Point3f localPoint;
    //Vector3f tangentU, tangentV;
    //Normal3f normal;
    int shapeId = -1;

    Vector3f pError;

    //Transform localToWorld, worldToLocal;
    Transform shadingToWorld; //worldToShading;

    Ray worldCameraRay;
    Vector3f shadingCameraDirection;
    Vector3f L; // Light direction

    ShadingInfo shading;

    Material *material = nullptr;    

    //BSDF *bsdf = nullptr;
    //BSSRDF *bssrdf = nullptr;
    //mutable Vector3f dpdx, dpdy;
    //mutable Float dudx = 0, dvdx = 0, dudy = 0, dvdy = 0;
    

    public:

    __host__ __device__
    SurfaceInteraction() { }

    __host__ __device__
    SurfaceInteraction(const Point3f &point,
                        const Point3f &localPoint,
                        const Point2f &surfacePoint,
                        const Normal3f &normal,
                        const Vector3f &tangentU,
                        const Vector3f &tangentV,
                        const Vector3f &dndu,
                        const Vector3f &dndv,
                        Material *material,
                        const Ray &ray,
                        unsigned int shapeId,
                        Vector3f pError)
    : Interaction(point)
    {
        this->surfacePoint = surfacePoint;
        //this->localPoint = localPoint;
        //this->normal = normal;
        //this->tangentU = tangentU;
        //this->tangentV = tangentV;
        this->material = material;
        this->shapeId = shapeId;

        this->pError = pError;

        // Initialize transformations
        //localToWorld = Transformations::baseChange(worldPoint, tangentU, tangentV, normal.toVector());
        //worldToLocal = inverse(localToWorld);
        this->shading.normal = normalize(normal);        
        //shading.tangentU = normalize(cross(shading.normal, tangentV));
        //shading.tangentV = normalize(cross(shading.normal, shading.tangentU));
        Vector3f shadingTangentU = normalize(cross(shading.normal, tangentV));
        Vector3f shadingTangentV = normalize(cross(shading.normal, shadingTangentU));


        //shading.dndu = dndu;
        //shading.dndv = dndv;

        // Check if tangentU is pointing in the same direction as the world tangentU
        if (cosine(shadingTangentU, tangentU) < 0)
            shadingTangentU = -shadingTangentU;

        // Check if tangentV is pointing in the same direction as the world tangentV
        if (cosine(shadingTangentV, tangentV) < 0)
            shadingTangentV = -shadingTangentV;

        shadingToWorld = Transformations::baseChange(
                    worldPoint, 
                    shadingTangentU, 
                    shadingTangentV, 
                    shading.normal.toVector());

        this->worldCameraRay = ray;
        this->shadingCameraDirection = -worldToShading(ray.direction);
    }

    __host__ __device__
    ~SurfaceInteraction() 
    { 
    }

 
    // Copy constructor
    __host__ __device__
    SurfaceInteraction(const SurfaceInteraction &other)
    {
        worldPoint = other.worldPoint;
        surfacePoint = other.surfacePoint;
        //localPoint = other.localPoint;
        //tangentU = other.tangentU;
        //tangentV = other.tangentV;
        //normal = other.normal;
        shapeId = other.shapeId;

        pError = other.pError;
        worldCameraRay = other.worldCameraRay;
        shadingCameraDirection = other.shadingCameraDirection;
        shading = other.shading;
        material = other.material;
        
        shadingToWorld = other.shadingToWorld;
        //worldToShading = other.worldToShading;
    }

    // Copy operator
    __host__ __device__
    SurfaceInteraction& operator=(const SurfaceInteraction &other)
    {
        worldPoint = other.worldPoint;
        surfacePoint = other.surfacePoint;
        //localPoint = other.localPoint;
        //tangentU = other.tangentU;
        //tangentV = other.tangentV;
        //normal = other.normal;
        shapeId = other.shapeId;

        pError = other.pError;
        worldCameraRay = other.worldCameraRay;
        shadingCameraDirection = other.shadingCameraDirection;
        shading = other.shading;
        material = other.material;

        shadingToWorld = other.shadingToWorld;
        //worldToShading = other.worldToShading;

        return *this;
    }


    // Spawns a world ray from surface, given a world direction 
    inline Ray spawnRay(const Point3f &outgoingPoint, 
            const Vector3f &sampleDirection) const override
    {
        // Offset ray origin to prevent self-intersection
        Float d = dot(abs(shading.normal), pError);
        Vector3f offset = d * shading.normal.toVector();
        
        if (dot(sampleDirection, shading.normal) < 0)
            offset = -offset;
        
        Point3f offsetedPoint = outgoingPoint + offset;

        // Round offset point away from p 
        for (int i = 0; i < 3; i++) 
        {
            if (offset[i] > 0)      
                offsetedPoint[i] = nextFloatUp(offsetedPoint[i]);
            else if (offset[i] < 0) 
                offsetedPoint[i] = nextFloatDown(offsetedPoint[i]);
        }

        return Ray(offsetedPoint, sampleDirection, INFINITY, worldCameraRay.time);
    }

    template <typename T>
    __host__ __device__
    inline T worldToShading(T v) const
    {
        return inverse(shadingToWorld)(v);
    }

    int getShapeId() const
    {
        return shapeId;
    }

    Point2f getSurfacePoint() const
    {
        return surfacePoint;
    }


    Spectrum getEmittedLight() const
    {
        return material->getEmittedRadiance(*this);
    }

    void getMaterialBSSDFparameters(Spectrum &kd, Spectrum &sigmaS, Spectrum &sigmaA, Float &g, Float &eta) const
    {
        material->getBSSDFparameters(kd, sigmaS, sigmaA, g, eta, surfacePoint);
    }


    Spectrum sampleRay(
            Sampler *sampler,
            Ray &newRay) const override
    {        
        Vector3f newDirection;
        Spectrum reflectedLight;

        Point3f outgoingPoint;

        if (!material->isSubsurfaceScattering() || !worldCameraRay.isCameraRay())
        {
            reflectedLight = material->sampleDirection(shadingCameraDirection, 
                                    *sampler, 
                                    newDirection,
                                    surfacePoint,
                                    outgoingPoint,
                                    worldToShading(L));
        }
        // Subsurface scattering has a special case for camera rays (indirect light just computes without sss)
        else 
        {
            SSMaterial *sssMaterial = (SSMaterial*) material;
            reflectedLight = sssMaterial->sampleCameraRay(shadingCameraDirection, 
                                    *sampler, 
                                    newDirection,
                                    surfacePoint,
                                    outgoingPoint,
                                    worldToShading(L));
        }

        Vector3f newWorldDirection = shadingToWorld(newDirection);
        Point3f worldOutgoingPoint = shadingToWorld(outgoingPoint);

        newRay = SurfaceInteraction::spawnRay(worldOutgoingPoint, newWorldDirection);

        // Specular materials will still see the subsurface scattering
        if (worldCameraRay.isCameraRay() && material->isSpecular()) {
            newRay.cameraRay = true;
        }

        return reflectedLight;
    }

    // Sample direction points to light
    Float pdf (Vector3f sampleDirection) const
    {
        if (worldCameraRay.isCameraRay() && material->isSubsurfaceScattering())
        {
            SSMaterial *sssMaterial = (SSMaterial*) material;
            return sssMaterial->pdfCamera(shadingCameraDirection, sampleDirection);
        }

        Vector3f localSampleDir = worldToShading(sampleDirection);

        return material->pdf(shadingCameraDirection, localSampleDir);
    }

    bool isOrientedTo(Vector3f objectDirection) const
    {
        return dot(objectDirection, shading.normal) > 0;
    }

    Spectrum lightScattered(Vector3f sampleDirection=Vector3f()) const
    {
        if (sampleDirection != Vector3f())
            sampleDirection = worldToShading(sampleDirection);
        else 
            sampleDirection = Vector3f(0, 0, 1);

        if (worldCameraRay.isCameraRay() && material->isSubsurfaceScattering())
        {
            SSMaterial *sssMaterial = (SSMaterial*) material;
            return sssMaterial->lightReflectedCamera(shadingCameraDirection, sampleDirection, surfacePoint);
        }

        return material->lightReflected(shadingCameraDirection, sampleDirection, surfacePoint);
    }

    Spectrum getAmbientScattering() const
    {
        return material->getAmbientScattering();
    }


    bool isScatteringMaterial() const override
    {
        return material->isScattering();
    }

    bool isEmissiveMaterial() const override
    {
        return material->isEmissive();
    }

    bool isSpecularMaterial() const override
    {
        return material->isSpecular();
    }

    bool isSubsurfaceScattering() const override
    {
        return material->isSubsurfaceScattering();
    }


    //void setShadingGeometry(const Vector3f &dpdu, const Vector3f &dpdv,
    //    const Normal3f &dndu, const Normal3f &dndv, bool orientationIsAuthoritative);


    void computeDifferentials(const RayDifferential &r) const {};

    bool isSurfaceInteraction() const override
    {
        return true;
    }

    //Spectrum Le(const Vector3f &w) const;

    friend SurfaceInteraction transform(Transform &t, SurfaceInteraction &s);

    friend SurfaceInteraction* transform(Transform &t, SurfaceInteraction *s);
};

SurfaceInteraction transform(const Transform &tr, SurfaceInteraction oldS)
{
    oldS.worldPoint = tr(oldS.worldPoint, oldS.pError);
    //oldS.tangentU = tr(oldS.tangentU);
    //oldS.tangentV = tr(oldS.tangentV);
    oldS.shadingToWorld = tr(oldS.shadingToWorld);
    //oldS.worldToShading = inverse(oldS.shadingToWorld);

    //oldS.normal = tr(oldS.normal);
    oldS.worldCameraRay = tr(oldS.worldCameraRay);

    oldS.shading.normal = tr(oldS.shading.normal);
    //oldS.shading.tangentU = tr(oldS.shading.tangentU);
    //oldS.shading.tangentV = tr(oldS.shading.tangentV);
    //oldS.shading.dndu = tr(oldS.shading.dndu);
    //oldS.shading.dndv = tr(oldS.shading.dndv);

    return oldS;
}

SurfaceInteraction* transform(const Transform &tr, SurfaceInteraction *oldS)
{
    oldS->worldPoint = tr(oldS->worldPoint, oldS->pError);
    //oldS->tangentU = tr(oldS->tangentU);
    //oldS->tangentV = tr(oldS->tangentV);
    oldS->shadingToWorld = tr(oldS->shadingToWorld);
    //oldS->worldToShading = inverse(oldS->shadingToWorld);

    //oldS->normal = tr(oldS->normal);
    oldS->worldCameraRay = tr(oldS->worldCameraRay);

    oldS->shading.normal = tr(oldS->shading.normal);
    //oldS->shading.tangentU = tr(oldS->shading.tangentU);
    //oldS->shading.tangentV = tr(oldS->shading.tangentV);
    //oldS->shading.dndu = tr(oldS->shading.dndu);
    //oldS->shading.dndv = tr(oldS->shading.dndv);

    return oldS;
}