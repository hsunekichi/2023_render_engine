#pragma once

#include "../cuda_libraries/types.h"
#include "Shape.cu"
#include "../cuda_libraries/math.cu"
#include "../cuda_libraries/geometricMath.cu"
#include "../transformations/SurfaceInteraction.cu"

class Sphere : public Shape
{
    protected:
    Float sph_radius;
    Float zMin, zMax;
    Float thetaMin, thetaMax, phiMax;
    
    Transform objectToWorld, worldToObject;

    inline bool getSphereCollisions(
            Ray &ray, 
            Vector3f &oErr, Vector3f &dErr, 
            ErrorFloat &offset0, ErrorFloat &offset1) const
    {
        // Initialize EFloat ray coordinates
        ErrorFloat ox(ray.origin.x, oErr.x);
        ErrorFloat oy(ray.origin.y, oErr.y);
        ErrorFloat oz(ray.origin.z, oErr.z);

        ErrorFloat dx(ray.direction.x, dErr.x);
        ErrorFloat dy(ray.direction.y, dErr.y);
        ErrorFloat dz(ray.direction.z, dErr.z);

        // Compute quadratic sphere coefficients
        ErrorFloat a = (dx * dx) + (dy * dy) + (dz * dz); // dx^2 + dy^2 + dz^2
        ErrorFloat b = Float(2.0) * ((dx * ox) + (dy * oy) + (dz * oz));  // 2(dx*ox + dy*oy + dz*oz)
        ErrorFloat c = (ox * ox) + (oy * oy) + (oz * oz) - (ErrorFloat(sph_radius) * ErrorFloat(sph_radius));   // ox^2 + oy^2 + oz^2 - r^2

        // Solve quadratic equation of sphere and ray
        return quadratic(a, b, c, offset0, offset1);
    }

    inline bool checkRayBounds(
            Ray &ray, Point3f &hitPoint, Float &phi, 
            ErrorFloat &offset0, ErrorFloat &offset1, ErrorFloat &hitOffset) const
    {
        // Check quadric shape offset0 and offset1 for nearest intersection
        // It uses the error bounds to compensate the floating point error
        // If the min collision is further than the ray limit or the max behind its origin,
        //   we can discard both collisions
        if (offset0.upperBound() > ray.maximumOffset || offset1.lowerBound() <= 0)
            return false;

        hitOffset = offset0;
        
        // If the min collision is behind the origin, we check the max
        if (offset0.lowerBound() <= 0) 
        {
            hitOffset = offset1;

            // If the max collision is further than the ray limit, we discard both
            if (offset1.upperBound() > ray.maximumOffset)   
                return false;
        }


        // Convert intersection to spherical coordinates
        hitPoint = ray((Float)hitOffset);

        // Reproject point to surface to refine intersection 
        hitPoint *= (sph_radius / distance(hitPoint, Point3f(0, 0, 0)));

        if (hitPoint.x == 0 && hitPoint.y == 0) 
            hitPoint.x = 1e-5f * sph_radius;
        
        phi = atan2(hitPoint.y, hitPoint.x);

        // Remap phi to [0, 2pi]
        if (phi < 0) 
            phi += 2 * PI;

        return true;
    }

    inline bool checkClippingBounds(
            Ray &ray, Point3f &hitPoint, ErrorFloat &hitOffset, Float &phi, 
            ErrorFloat &offset0, ErrorFloat &offset1) const
    {
        // Test sphere intersection against clipping parameters
        if ((zMin > -sph_radius && hitPoint.z < zMin) ||
            (zMax <  sph_radius && hitPoint.z > zMax) || phi > phiMax) 
        {
            if (hitOffset == offset1) 
                return false;

            if (offset1.upperBound() > ray.maximumOffset) 
                return false;

            hitOffset = offset1;
            // Compute sphere hit position and phi
            hitPoint = ray((Float)hitOffset);


            // Reproject point to surface to refine intersection 
            hitPoint *= sph_radius / distance(hitPoint, Point3f(0, 0, 0));

            if (hitPoint.x == 0 && hitPoint.y == 0) 
                hitPoint.x = 1e-5f * sph_radius;

            phi = atan2(hitPoint.y, hitPoint.x);

            if (phi < 0) 
                phi += 2 * PI;


            // If neither of the sphere intersections were valid 
            //  within the clipping parameters, return false
            if ((zMin > -sph_radius && hitPoint.z < zMin) ||
                (zMax <  sph_radius && hitPoint.z > zMax) || phi > phiMax)
            {
                return false;
            }
        }

        return true;
    }

    inline void computeIntersection(
            Point3f &hitPoint, Vector3f &tangentU, 
            Vector3f &tangentV, Normal3f &normal,
            Normal3f &normalU, Normal3f &normalV) const
    {
        // Find spherical/parametric representation of sphere hit
        //Float u = phi / phiMax;
        Float theta = acos(clamp(hitPoint.z / sph_radius, -1.0, 1.0));
        //Float v = (theta - thetaMin) / (thetaMax - thetaMin);

        // Compute sphere partial derivates 
        Float zRadius = sqrt(hitPoint.x * hitPoint.x + hitPoint.y * hitPoint.y);
        Float invZRadius = 1 / zRadius;
        Float cosPhi = hitPoint.x * invZRadius;
        Float sinPhi = hitPoint.y * invZRadius;

        tangentU = Vector3f(-phiMax * hitPoint.y, phiMax * hitPoint.x, 0);
        tangentV = Vector3f(hitPoint.z * cosPhi, hitPoint.z * sinPhi,
                    -sph_radius * sin(theta)) * (thetaMax - thetaMin);


        Vector3f d2Pduu = Vector3f(hitPoint.x, hitPoint.y, 0) * -phiMax * phiMax;
        Vector3f d2Pduv = Vector3f(-sinPhi, cosPhi, 0.) *(thetaMax - thetaMin) * hitPoint.z * phiMax;
        
        Vector3f d2Pdvv = Vector3f(hitPoint.x, hitPoint.y, hitPoint.z) * 
                                -(thetaMax - thetaMin) * (thetaMax - thetaMin);


        // Compute coefficients for fundamental forms
        Float E = dot(tangentU, tangentU);
        Float F = dot(tangentU, tangentV);
        Float G = dot(tangentV, tangentV);

        // On CPU normal appears inversed, on gpu it is correct
        normal = Normal3f(normalize(-cross(tangentU, tangentV)));
        
        Float e = dot(normal, d2Pduu);
        Float f = dot(normal, d2Pduv);
        Float g = dot(normal, d2Pdvv);

        // Compute and from fundamental form coefficients
        Float invEGF2 = 1 / (E * G - F * F);
        normalU = Normal3f(tangentU * (f * F - e * G) * invEGF2 + 
                                tangentV * (e * F - f * E) * invEGF2);
        normalV = Normal3f(tangentU * (g * F - f * G) * invEGF2 + 
                                tangentV * (f * F - g * E) * invEGF2);
    }


    public:

    Sphere(Transform ObjectToWorld, Transform WorldToObject,
              bool reverseOrientation, Float sph_radius, Float zMin, Float zMax,
              Float phiMax)
           : objectToWorld(ObjectToWorld), worldToObject(WorldToObject),
             sph_radius(sph_radius), 
             zMin(clamp(min(zMin, zMax), -sph_radius, sph_radius)),
             zMax(clamp(max(zMin, zMax), -sph_radius, sph_radius)),
             thetaMin(acos(clamp(zMin / sph_radius, -1.0, 1.0))),
             thetaMax(acos(clamp(zMax / sph_radius, -1.0, 1.0))),
             phiMax(clamp(phiMax, 0.0, 2*PI)) 
    { }

    Sphere(Point3f center, 
        Vector3f axis, 
        Point3f city)
    {
        assert (abs(axis.length()/2 - distance(center, city)) < FLOAT_ERROR_MARGIN);

        // Obtener la nueva base
        sph_radius = axis.length()/2;

        // Vector from center to north pole, through axis
        Vector3f vectorCenterNorthPole = axis/2;

        // Vector from center to city
        Vector3f vectorCenterCity = city - center;

        Vector3f ejeX = cross(vectorCenterCity, vectorCenterNorthPole); // Third vector
        Vector3f ejeY = cross(ejeX, vectorCenterNorthPole);             // Vector to meridian
        Vector3f ejeZ = vectorCenterNorthPole;                          // Vector through axis

        ejeX = normalize(ejeX) * sph_radius;
        ejeY = -normalize(ejeY) * sph_radius;   // Invert handedness
        ejeZ = normalize(ejeZ) * sph_radius;

        objectToWorld = Transformations::baseChange(center, ejeX, ejeY, ejeZ);
        worldToObject = inverse(objectToWorld);

        zMin = -sph_radius;
        zMax = sph_radius;
        thetaMin = zMin / sph_radius;
        thetaMax = zMax / sph_radius;
        phiMax = 2 * PI;
    }

    
    //Sphere(Point3f center, Float radius)
    //: Sphere(Transformations::translate(center.toVector()), 
    //        inverse(Transformations::translate(center.toVector())), 
    //        false, radius, -radius, radius, 2*PI)
    //{}
    
    //Sphere(Point3f center, Float radius)
    //: Sphere (center, Vector3f(0, 0, 2*radius), center + Vector3f(0, radius, 0))
    //{}

    Sphere(Point3f center, Float radius)
    {
        sph_radius = radius;
        zMin = -radius;
        zMax = radius;
        thetaMin = zMin / radius;
        thetaMax = zMax / radius;
        phiMax = 2 * PI;

        objectToWorld = Transformations::translate(center.toVector());
        objectToWorld = objectToWorld(Transformations::rotateZ(90));
        worldToObject = inverse(objectToWorld);
    }


    inline Bound3f boundObject() const 
    {
        return Bound3f(Point3f(-sph_radius, -sph_radius, zMin),
                        Point3f( sph_radius,  sph_radius, zMax));
    }

    Bound3f worldBound() const 
    {
        Point3f worldLeft = objectToWorld(Point3f(-sph_radius, -sph_radius, zMin));
        Point3f worldRight = objectToWorld(Point3f( sph_radius,  sph_radius, zMax));

        return Bound3f(worldLeft, worldRight);
    }

    Point2f worldToSurface(const Point3f &p) const 
    {
        Point3f localP =  worldToObject(p);
        
        return localToSurface(localP);
    }

    Point2f localToSurface(const Point3f &p) const
    {
        Float radius, phi, theta;
        to_spherical(p, radius, theta, phi);

        return Point2f(theta, phi);
    }

    bool intersect(const Ray &r, Float &offsetHit,
            SurfaceInteraction &intersection, unsigned int shapeId, 
            bool testAlphaTexture) const override
    {     
        Float phi;
        Point3f hitPoint;

        // Transform ray to object space
        Vector3f oErr, dErr;
        Ray ray = worldToObject(r, oErr, dErr);  // Transform computing the error
        ray.direction = normalize(ray.direction);   // Normalize to prevent float errors on big spheres

        ErrorFloat offset0, offset1, hitOffset;
        bool b_solutions = getSphereCollisions(ray, oErr, dErr, offset0, offset1);

        if (!b_solutions)   // No collisions
            return false;

        b_solutions = checkRayBounds(ray, hitPoint, phi, offset0, offset1, hitOffset);

        if (!b_solutions)   // No collisions
            return false;

        b_solutions = checkClippingBounds(ray, hitPoint, hitOffset, phi, offset0, offset1);

        if (!b_solutions)   // No collisions
            return false;

        Vector3f tangentU, tangentV;
        Normal3f normal, normalU, normalV;

        // There is a valid intersection
        computeIntersection(hitPoint, tangentU, tangentV, 
                normal, normalU, normalV);
        
        // Compute error bounds for sphere intersection
        Vector3f pError = abs(hitPoint.toVector()) * errorBound(5);

        Normal3f hitNormal = Normal3f(normal);
        Vector3f hitTangentU = normalize(tangentU);
        Vector3f hitTangentV = normalize(tangentV);

        Point2f surfacePoint = localToSurface(hitPoint);

        if (hitOffset.data < offsetHit)
        {
            SurfaceInteraction surfaceInteraction(hitPoint,
                                    hitPoint, surfacePoint,
                                    hitNormal, 
                                    hitTangentU, hitTangentV,
                                    Vector3f(0), Vector3f(0),
                                    material, ray, shapeId,
                                    pError);
                        
            // Initialize SurfaceInteraction from parametric information 
            intersection = transform(objectToWorld,
                            surfaceInteraction);

            if (intersection.worldPoint.x < 3.5 &&
                intersection.worldPoint.x > 1.5 &&
                intersection.worldPoint.y < 1 &&
                intersection.worldPoint.y > -1 &&
                intersection.worldPoint.z < 1 &&
                intersection.worldPoint.z > -1)
            {
                int a = 0;
                a++;
            }
            

            // Update tHit for quadric intersection
            offsetHit = (Float)hitOffset;

            return true;
        }
        else {
            return false;
        }
    }

    bool intersects(const Ray &r,
            Float &offsetHit,
            bool checkShadow = false, 
            bool testAlphaTexture = true) const override
    {
        if (checkShadow && material->isTransparent())
            return false;

        Float phi;
        Point3f hitPoint;

        // Transform ray to object space
        Vector3f oErr, dErr;
        Ray ray = worldToObject(r, oErr, dErr);  // Transform computing the error
        ray.direction = normalize(ray.direction);   // Normalize to prevent float errors on big spheres

        ErrorFloat offset0, offset1, hitOffset;
        bool b_solutions = getSphereCollisions(ray, oErr, dErr, offset0, offset1);

        if (!b_solutions)   // No collisions
            return false;

        b_solutions = checkRayBounds(ray, hitPoint, phi, offset0, offset1, hitOffset);

        if (!b_solutions)   // No collisions
            return false;

        b_solutions = checkClippingBounds(ray, hitPoint, hitOffset, phi, offset0, offset1);

        if (!b_solutions)   // No collisions
            return false;


        if (hitOffset.data < offsetHit)
        {
            offsetHit = (Float)hitOffset;
            return true;
        }
        else {
            return false;
        }
    }

    Point2f surfacePointTo01(Point2f sample) const override 
    {
        sample = sample + 180;
        sample = sample / 360;

        return sample;
    }

    inline Float area() const {
        return 4 * PI * sph_radius * sph_radius;
    }
    
    inline Float volume() const {
        return (4.0/3.0) * PI * sph_radius * sph_radius * sph_radius;
    }

    inline Float radius() const {
        return sph_radius;
    }
};