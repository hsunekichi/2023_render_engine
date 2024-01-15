#pragma once

#include "../cuda_libraries/types.h"
#include "../transformations/Transform.cu"
#include "../transformations/Interaction.cu"
//#include "../transformations/SurfaceInteraction.cu"

#include "../geometry/Ray.cu"
#include "../geometry/Point.cu"
#include "../geometry/Vector.cu"
#include "../geometry/BoundingBox.cu"

#include "../materials/Material.cu"

class SurfaceInteraction;

class Shape
{
    protected:

    Material *material = nullptr;

    public:

    Shape() {}
    

    virtual Bound3f boundObject() const = 0;

    virtual Bound3f worldBound() const = 0;

    
    //virtual Point3f surfaceToWorld(Point2f const &p) const = 0;

    virtual bool intersect(const Ray &ray, Float &hitOffset,
        SurfaceInteraction &interaction, unsigned int shapeId, 
        bool testAlphaTexture = true) const = 0;

    virtual bool intersects(const Ray &ray,
            Float &hitPoint,
            bool checkShadow = false, 
            bool testAlphaTexture = true) const = 0;

    virtual Float area() const = 0;

    virtual Point2f surfacePointTo01(Point2f sample) const {
        return sample;
    }

/*
    __host__ 
    virtual Interaction* sample(const Point2f &u) const = 0;

    __host__ 
    virtual Float pdf(const Interaction *ref) const {
        return 1 / area();
    }

    __host__ 
    virtual Interaction* sample(const Interaction *ref,
                                const Point2f &u) const 
    {
        return sample(u);
    }

    __host__ 
    virtual Float pdf(const Interaction *ref, const Vector3f &wi) const = 0;
*/

    Material* getMaterial() const { 
        return material; 
    }

    virtual void setMaterial(Material *material) 
    { 
        this->material = material; 
    }
};