#pragma once


#include "Shape.cu"
#include "../geometry/Vector.cu"
#include "../geometry/Point.cu"
#include "../geometry/Ray.cu"
#include "../transformations/SurfaceInteraction.cu"
#include "../transformations/Transform.cu"


class Plane : public Shape
{
    protected:

    Transform objectToWorld, worldToObject;


    public:

    Plane (Transform &objectToWorld, Transform &worldToObject,
            bool reverseOrientation) 
    : objectToWorld(objectToWorld), worldToObject(worldToObject)
    {}

    Plane (Point3f bottomLeftCorner, 
            Vector3f axisX, 
            Vector3f axisY, 
            Vector3f normal)
    {
        int size_X = axisX.length();
        int size_Y = axisY.length();

        // Warrantee that normal is perpendicular
        Vector3f tempNormal = cross(axisX, axisY);
        
        if (dot(tempNormal, normal) > 0)
            normal = tempNormal;
        else
            normal = -tempNormal;

        axisX = normalize(axisX) * size_X;
        axisY = normalize(axisY) * size_Y;
        normal = normalize(normal);

        objectToWorld = Transformations::baseChange(bottomLeftCorner, axisX, axisY, normal);
        worldToObject = inverse(objectToWorld);
    }


    Bound3f boundObject() const override
    {
        return Bound3f(Point3f(0, 0, -0.00001), Point3f(1, 1, 0.00001));
    }

    Bound3f worldBound() const 
    {
        return objectToWorld(Bound3f(Point3f(0, 0, -0.00001), Point3f(1, 1, 0.00001)));
    }


    bool intersect(const Ray &ray, Float &hitOffset,
            SurfaceInteraction &interaction, unsigned int shapeId, 
            bool testAlphaTexture = true) const
    {
        Ray localRay = worldToObject(ray);

        Normal3f normal = Normal3f(0, 0, 1);
        Float cosine = dot(normal, localRay.direction);
        Vector3f planeToRayOrigin = localRay.origin - Point3f(0, 0, 0);

        
        // If the ray is parallel to the plane, it will never intersect
        if (abs(cosine) > FLOAT_ERROR_MARGIN)
        {
            Point3f hitPoint = localRay.origin - (dot(planeToRayOrigin, normal) / dot(localRay.direction, normal)) * localRay.direction;
            Float currentHitOffset = localRay.getOffset(hitPoint);

            // Reproject point to surface
            hitPoint.z = 0;

            Point2f surfacePoint = Point2f(hitPoint.x, hitPoint.y);

            if (!(currentHitOffset <= 0
                || currentHitOffset > ray.maximumOffset
                || surfacePoint.x < 0
                || surfacePoint.x > 1
                || surfacePoint.y < 0
                || surfacePoint.y > 1
                )
                && currentHitOffset < hitOffset)
            {
                hitOffset = currentHitOffset;
                Vector3f tangentU = Vector3f(1, 0, 0);
                Vector3f tangentV = Vector3f(0, 1, 0);

                if (sameHemisphere(normal.toVector(), localRay.direction))
                {
                    normal = -normal;
                }

                // Compute error bound
                // For now, we hardcode a small bound
                Vector3f pError = Vector3f(0.0001, 0.0001, 0.0001);

                normal = normalize(normal);
                
                interaction = transform(objectToWorld, 
                                SurfaceInteraction(hitPoint, hitPoint, 
                                            surfacePoint, 
                                            normal, 
                                            tangentU, tangentV,
                                            Vector3f(0), Vector3f(0),
                                            material, localRay, shapeId,
                                            pError));
                
                return true;
            } 
        }

        return false;
    }

    bool intersects(const Ray &ray, Float &hitOffset,
        bool checkShadow = false,
        bool testAlphaTexture = true) const
    {
        if (checkShadow && material->isTransparent())
            return false;

        Ray localRay = worldToObject(ray);

        Normal3f normal = Normal3f(0, 0, 1);
        Float cosine = dot(normal, localRay.direction);
        Vector3f planeToRayOrigin = localRay.origin - Point3f(0, 0, 0);

        
        // If the ray is parallel to the plane, it will never intersect
        if (abs(cosine) > FLOAT_ERROR_MARGIN)
        {
            Point3f hitPoint = localRay.origin - (dot(planeToRayOrigin, normal) / dot(localRay.direction, normal)) * localRay.direction;
            Float currentHitOffset = localRay.getOffset(hitPoint);

            // Reproject point to surface
            hitPoint.z = 0;

            Point2f surfacePoint = Point2f(hitPoint.x, hitPoint.y);

            if (!(currentHitOffset <= 0
                || currentHitOffset > ray.maximumOffset
                || surfacePoint.x < 0
                || surfacePoint.x > 1
                || surfacePoint.y < 0
                || surfacePoint.y > 1
                )
                && currentHitOffset < hitOffset)
            {
                hitOffset = currentHitOffset;
                return true;
            } 
        }

        return false;
    }

    inline Float area() const override
    {
        Vector3f axisX = Vector3f(1, 0, 0);
        Vector3f axisY = Vector3f(0, 1, 0);

        Vector3f worldX = objectToWorld(axisX);
        Vector3f worldY = objectToWorld(axisY);

        return worldX.length() * worldY.length();
    }

    Material* getMaterial() const
    {
        return material;
    }

    void setMaterial(Material *material)
    {
        this->material = material;
    }
};