#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "Shape.cu"
#include "../geometry/Vector.cu"
#include "../geometry/Point.cu"
#include "../geometry/Ray.cu"
#include "../transformations/SurfaceInteraction.cu"
#include "../transformations/Transform.cu"
#include "../textures/AlphaTexture.cu"

struct TriangleMeshData
{
    Point3f *vertices = nullptr;
    Point2f *textureCoords = nullptr;
    Vector3f *normals = nullptr;
};

struct TriangleMesh
{
    thrust::device_vector<Point3f> vertices;
    thrust::device_vector<Point2f> textureCoords;
    thrust::device_vector<Vector3f> normals;

    AlphaTexture alphaTexture;
};

struct CpuTriangleMesh
{
    thrust::host_vector<Point3f> vertices;
    thrust::host_vector<Point2f> textureCoords;
    thrust::host_vector<Vector3f> normals;

    AlphaTexture alphaTexture;

    // Construct with triangle mesh
    CpuTriangleMesh(const TriangleMesh &mesh)
    : vertices(mesh.vertices),
      textureCoords(mesh.textureCoords),
      normals(mesh.normals),
      alphaTexture(mesh.alphaTexture)
    {}
};



class Triangle
{
    public:

    bool isTransparent = false;

    unsigned int vertexIndices[3];
    unsigned int textureIndices[3];
    unsigned int normalIndices[3];
    int meshId = -1;
    
    TriangleMeshData *mesh = nullptr;

    Material *material;

    public:

    __host__ __device__
    Triangle(TriangleMeshData *mesh, 
                int vertexIndices[3],
                int textureIndices[3],
                int normalIndices[3],
                bool reverseOrientation)
    : mesh(mesh) 
    {
        // All vertices are world points
        this->vertexIndices[0] = vertexIndices[0];
        this->vertexIndices[1] = vertexIndices[1];
        this->vertexIndices[2] = vertexIndices[2];

        this->textureIndices[0] = textureIndices[0];
        this->textureIndices[1] = textureIndices[1];
        this->textureIndices[2] = textureIndices[2];

        this->normalIndices[0] = normalIndices[0];
        this->normalIndices[1] = normalIndices[1];
        this->normalIndices[2] = normalIndices[2];
    }   
    
    __host__ __device__
    Triangle (int vertexIndices[3])
    {
        this->vertexIndices[0] = vertexIndices[0];
        this->vertexIndices[1] = vertexIndices[1];
        this->vertexIndices[2] = vertexIndices[2];
    }

    __host__ __device__
    Triangle()
    {
        vertexIndices[0] = 0;
        vertexIndices[1] = 0;
        vertexIndices[2] = 0;
    }

    Bound3f boundObject() const
    {
        Point3f p0 = mesh->vertices[vertexIndices[0]];
        Point3f p1 = mesh->vertices[vertexIndices[1]];
        Point3f p2 = mesh->vertices[vertexIndices[2]];

        // Transform points to object
        //p0 = worldToObject(p0);
        //p1 = worldToObject(p1);
        //p2 = worldToObject(p2);

        //throw std::runtime_error("Triangle bound object not implemented");

        printf("Triangle bound object not implemented\n");
        exit(1);
        
        return Bound3f(p0, p1) + p2;
    }

    __host__ __device__
    Bound3f worldBound() const
    {
        const Point3f &p0 = mesh->vertices[vertexIndices[0]];
        const Point3f &p1 = mesh->vertices[vertexIndices[1]];
        const Point3f &p2 = mesh->vertices[vertexIndices[2]];

        return Bound3f(p0, p1) + p2;
    }

    __host__ __device__
    void getSurfaceCoordinates(Point2f uv[3]) const
    {
        // If the mesh has no textureCoords, use the default uv coordinates
        if (mesh->textureCoords == nullptr)
        {
            uv[0] = Point2f(0, 0);
            uv[1] = Point2f(1, 0);
            uv[2] = Point2f(1, 1);
        }
        else
        {
            //std::cout << "Getting coordinates\n";
            uv[0] = mesh->textureCoords[textureIndices[0]];
            uv[1] = mesh->textureCoords[textureIndices[1]];
            uv[2] = mesh->textureCoords[textureIndices[2]];
            //std::cout << "coordinates: " << uv[0] << " " << uv[1] << " " << uv[2] << "\n";

        }
    }

    __host__ __device__
    bool evalAlphaTexture(Vector3f dpdu, Vector3f dpdv, Point2f surfacePoint) const
    {
        return true;
    }

    // Compute intersection directly on world space
    __host__ __device__
    bool intersect(const Ray &worldRay, Float &hitOffset,
        SurfaceInteraction &interaction, unsigned int shapeId, 
        bool testAlphaTexture = true) const
    {
        const Point3f &wp0 = mesh->vertices[vertexIndices[0]];
        const Point3f &wp1 = mesh->vertices[vertexIndices[1]];
        const Point3f &wp2 = mesh->vertices[vertexIndices[2]];

        /********************************************************************* 
         * Change the coordinates to the ray-triangle intersection base
         * The ray starts at 0,0,0 and points to 0,0,1 (goes through z axis)
         *********************************************************************/
        
        // Translate certices with respect to ray origin
        Point3f lp0 = wp0 - worldRay.origin.toVector();
        Point3f lp1 = wp1 - worldRay.origin.toVector();
        Point3f lp2 = wp2 - worldRay.origin.toVector();

        // Permute components of triangle vertices and ray direction
        Vector3f absDirection = abs(worldRay.direction);

        // Get index of the maximum dimension
        int zPosition = -1;
        if (absDirection[0] > absDirection[1])
            zPosition = 0;
        else
            zPosition = 1;

        if (absDirection[2] > absDirection[zPosition])
            zPosition = 2;

        // Permute x, y, z components based on the maximum dimension
        int xPosition = zPosition + 1; 
        if (xPosition == 3) 
            xPosition = 0;

        int yPosition = xPosition + 1; 
        if (yPosition == 3) 
            yPosition = 0;

        Vector3f localDirection = Vector3f(worldRay.direction[xPosition], 
                            worldRay.direction[yPosition], 
                            worldRay.direction[zPosition]);

        lp0 = Point3f(lp0[xPosition], lp0[yPosition], lp0[zPosition]);
        lp1 = Point3f(lp1[xPosition], lp1[yPosition], lp1[zPosition]);
        lp2 = Point3f(lp2[xPosition], lp2[yPosition], lp2[zPosition]);

        // Align ray direction with z axis
        Float Sx = -localDirection.x / localDirection.z;
        Float Sy = -localDirection.y / localDirection.z;
        Float Sz = 1.f / localDirection.z;
        lp0.x += Sx * lp0.z;
        lp0.y += Sy * lp0.z;
        lp1.x += Sx * lp1.z;
        lp1.y += Sy * lp1.z;
        lp2.x += Sx * lp2.z;
        lp2.y += Sy * lp2.z;

        /********************************* Compute intersection point ******************************/

        // Compute edge function coefficients
        Float edge0 = lp1.x * lp2.y - lp1.y * lp2.x;
        Float edge1 = lp2.x * lp0.y - lp2.y * lp0.x;
        Float edge2 = lp0.x * lp1.y - lp0.y * lp1.x;

        // If we are working with floats there is a chance that the edge function
        //  will return 0.0f even if the triangle is not degenerate
        //  In this case we compute the edge function using doubles
        if (sizeof(Float) == sizeof(float) 
            &&
            (edge0 == 0.0f || edge1 == 0.0f || edge2 == 0.0f)) 
        {
            double p2txp1ty = (double)lp2.x * (double)lp1.y;
            double p2typ1tx = (double)lp2.y * (double)lp1.x;
            edge0 = (float)(p2typ1tx - p2txp1ty);

            double p0txp2ty = (double)lp0.x * (double)lp2.y;
            double p0typ2tx = (double)lp0.y * (double)lp2.x;
            edge1 = (float)(p0typ2tx - p0txp2ty);

            double p1txp0ty = (double)lp1.x * (double)lp0.y;
            double p1typ0tx = (double)lp1.y * (double)lp0.x;
            edge2 = (float)(p1typ0tx - p1txp0ty);
        }

        // Perform first intersection test
        // If the edge functions dont have the same sign,
        //  the point is outside the triangle
        if ((edge0 < 0 || edge1 < 0 || edge2 < 0) 
            && 
            (edge0 > 0 || edge1 > 0 || edge2 > 0))
        {
            return false;
        }

        // Compute scaled barycentric coordinates
        Float det = edge0 + edge1 + edge2;
        if (det == 0)
            return false;
  
        // Compute scaled hit distance
        lp0.z *= Sz;
        lp1.z *= Sz;
        lp2.z *= Sz;

        Float scaledOffset = edge0 * lp0.z + edge1 * lp1.z + edge2 * lp2.z;

        // If the determinant has the wrong sign, the ray is outside the triangle
        if (det < 0 
            && 
            (scaledOffset >= 0 || scaledOffset < worldRay.maximumOffset * det))
        {
            return false;
        }
        else if (det > 0 
            && 
            (scaledOffset <= 0 || scaledOffset > worldRay.maximumOffset * det))
        {
            return false;
        }

        // Now there is an intersection

        // Compute barycentric coordinates
        Float invDet = 1 / det;
        Float b0 = edge0 * invDet;
        Float b1 = edge1 * invDet;
        Float b2 = edge2 * invDet;

        // Compute hit point
        Float currentHitOffset = scaledOffset * invDet;
        if (currentHitOffset > hitOffset)
            return false;
        
        hitOffset = currentHitOffset;

        // Move the hit point conservatively to avoid precision errors
        Float maxZt = maxComponent(abs(Vector3f(lp0.z, lp1.z, lp2.z)));
        Float deltaZ = errorBound(3) * maxZt;
        Float maxXt = maxComponent(abs(Vector3f(lp0.x, lp1.x, lp2.x)));
        Float maxYt = maxComponent(abs(Vector3f(lp0.y, lp1.y, lp2.y)));
        Float deltaX = errorBound(5) * (maxXt + maxZt);
        Float deltaY = errorBound(5) * (maxYt + maxZt);

        Float deltaE = 2 * (errorBound(2) * maxXt * maxYt + deltaY * maxXt +
                          deltaX * maxYt);

        Float maxE = maxComponent(abs(Vector3f(edge0, edge1, edge2)));
        
        Float deltaT = 3 * (errorBound(3) * maxE * maxZt + deltaE * maxZt +
                            deltaZ * maxE) * abs(invDet);
        
        // Check the offset against the error bounds
        if (hitOffset <= deltaT)
            return false;

        Float xAbsSum = (abs(b0 * wp0.x) + abs(b1 * wp1.x) +
                        abs(b2 * wp2.x));
        Float yAbsSum = (abs(b0 * wp0.y) + abs(b1 * wp1.y) +
                            abs(b2 * wp2.y));
        Float zAbsSum = (abs(b0 * wp0.z) + abs(b1 * wp1.z) +
                            abs(b2 * wp2.z));
        Vector3f pError = errorBound(7) * Vector3f(xAbsSum, yAbsSum, zAbsSum);


        // Compute triangle partial derivatives
        Vector3f dpdu, dpdv;
        Point2f surfaceVertices[3];
        getSurfaceCoordinates(surfaceVertices);

        // Compute deltas for triangle partial derivatives
        Vector2f duv02 = surfaceVertices[0] - surfaceVertices[2];
        Vector2f duv12 = surfaceVertices[1] - surfaceVertices[2];
        Vector3f dp02 = wp0 - wp2;
        Vector3f dp12 = wp1 - wp2;

        Float determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];

        if (abs(determinant) >= FLOAT_ERROR_MARGIN)
        {
            Float invDet = 1 / determinant;
            dpdu = (duv12[1] * dp02 - duv02[1] * dp12) * invDet;
            dpdv = (-duv12[0] * dp02 + duv02[0] * dp12) * invDet;
        }
        else
        {
            // Handle degenerate triangle
            generateCoordinateSystem(normalize(cross(wp2 - wp0, wp1 - wp0)), dpdu, dpdv);
        }

        // Interpolate hit coordinates using barycentric formula 
        //  (more precise than the ray parametric equation)
        Point3f hitPoint = b0 * wp0 + b1 * wp1 + b2 * wp2;
        Point2f surfacePoint = b0 * surfaceVertices[0] + 
                                b1 * surfaceVertices[1] + 
                                b2 * surfaceVertices[2];

        // Evaluate alpha texture if present
        if (testAlphaTexture
            && !evalAlphaTexture(dpdu, dpdv, surfacePoint))
        {
            return false;
        }

        // Fill surface interaction
        Normal3f normal = Normal3f(normalize(cross(dp02, dp12)));
        
        // If the ray is coming from the inside of the triangle,
        //  flip the normal
        if (dot(normal, worldRay.direction) > 0)
            normal = -normal;
        
        Normal3f ns = normal;
        Vector3f dndu = dpdu;
        Vector3f dndv = dpdv;

        if (mesh->normals != nullptr) 
        {
            ns = Normal3f(normalize(b0 * mesh->normals[normalIndices[0]] +
                b1 * mesh->normals[normalIndices[1]] + 
                b2 * mesh->normals[normalIndices[2]]));

            Vector3f ss;
            ss = normalize(dpdu);

            Vector3f ts = cross(ns, ss);
            if (ts.lengthSquared() > 0.f) 
            {
                ts = normalize(ts);
                ss = cross(ts, ns);
            }
            else {
                generateCoordinateSystem(ns.toVector(), ss, ts);
            }
            
            // Compute deltas for triangle partial derivatives of normal
            Vector2f duv02 = surfaceVertices[0] - surfaceVertices[2];
            Vector2f duv12 = surfaceVertices[1] - surfaceVertices[2];

            Vector3f dn1 = Vector3f(mesh->normals[normalIndices[0]] - mesh->normals[normalIndices[2]]);
            Vector3f dn2 = Vector3f(mesh->normals[normalIndices[1]] - mesh->normals[normalIndices[2]]);

            Float determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];
            if (determinant == 0) 
            {
                dndu = Vector3f(0);
                dndv = Vector3f(0);
            }
            else 
            {
                Float invDet = 1 / determinant;
                dndu = ( duv12[1] * dn1 - duv02[1] * dn2) * invDet;
                dndv = (-duv12[0] * dn1 + duv02[0] * dn2) * invDet;
            }

            hitPoint += (Float)0.15 * ns.toVector();
        }

        interaction = SurfaceInteraction(hitPoint, hitPoint, 
                surfacePoint, 
                ns, dpdu, dpdv,
                dndu, dndv, 
                material,
                worldRay,
                meshId,
                pError);

        return true;
    }

    __host__ __device__
    bool intersects(const Ray &worldRay,
            Float &hitOffset,
            bool checkShadow = false,
            bool testAlphaTexture = true) const
    {       
        if (checkShadow && isTransparent)
            return false;
     
        const Point3f &wp0 = mesh->vertices[vertexIndices[0]];
        const Point3f &wp1 = mesh->vertices[vertexIndices[1]];
        const Point3f &wp2 = mesh->vertices[vertexIndices[2]];

        /********************************************************************* 
         * Change the coordinates to the ray-triangle intersection base
         * The ray starts at 0,0,0 and points to 0,0,1 (goes through z axis)
         *********************************************************************/
        
        // Translate certices with respect to ray origin
        Point3f lp0 = wp0 - worldRay.origin.toVector();
        Point3f lp1 = wp1 - worldRay.origin.toVector();
        Point3f lp2 = wp2 - worldRay.origin.toVector();

        // Permute components of triangle vertices and ray direction
        Vector3f absDirection = abs(worldRay.direction);

        // Get index of the maximum dimension
        int zPosition = -1;
        if (absDirection[0] > absDirection[1])
            zPosition = 0;
        else
            zPosition = 1;

        if (absDirection[2] > absDirection[zPosition])
            zPosition = 2;

        // Permute x, y, z components based on the maximum dimension
        int xPosition = zPosition + 1; 
        if (xPosition == 3) 
            xPosition = 0;

        int yPosition = xPosition + 1; 
        if (yPosition == 3) 
            yPosition = 0;

        Vector3f localDirection = Vector3f(worldRay.direction[xPosition], 
                            worldRay.direction[yPosition], 
                            worldRay.direction[zPosition]);

        lp0 = Point3f(lp0[xPosition], lp0[yPosition], lp0[zPosition]);
        lp1 = Point3f(lp1[xPosition], lp1[yPosition], lp1[zPosition]);
        lp2 = Point3f(lp2[xPosition], lp2[yPosition], lp2[zPosition]);

        // Align ray direction with z axis
        Float Sx = -localDirection.x / localDirection.z;
        Float Sy = -localDirection.y / localDirection.z;
        Float Sz = 1.f / localDirection.z;
        lp0.x += Sx * lp0.z;
        lp0.y += Sy * lp0.z;
        lp1.x += Sx * lp1.z;
        lp1.y += Sy * lp1.z;
        lp2.x += Sx * lp2.z;
        lp2.y += Sy * lp2.z;

        /********************************* Compute intersection point ******************************/

        // Compute edge function coefficients
        Float edge0 = lp1.x * lp2.y - lp1.y * lp2.x;
        Float edge1 = lp2.x * lp0.y - lp2.y * lp0.x;
        Float edge2 = lp0.x * lp1.y - lp0.y * lp1.x;

        // If we are working with floats there is a chance that the edge function
        //  will return 0.0f even if the triangle is not degenerate
        //  In this case we compute the edge function using doubles
        if (sizeof(Float) == sizeof(float) 
            &&
            (edge0 == 0.0f || edge1 == 0.0f || edge2 == 0.0f)) 
        {
            double p2txp1ty = (double)lp2.x * (double)lp1.y;
            double p2typ1tx = (double)lp2.y * (double)lp1.x;
            edge0 = (float)(p2typ1tx - p2txp1ty);

            double p0txp2ty = (double)lp0.x * (double)lp2.y;
            double p0typ2tx = (double)lp0.y * (double)lp2.x;
            edge1 = (float)(p0typ2tx - p0txp2ty);

            double p1txp0ty = (double)lp1.x * (double)lp0.y;
            double p1typ0tx = (double)lp1.y * (double)lp0.x;
            edge2 = (float)(p1typ0tx - p1txp0ty);
        }

        // Perform first intersection test
        // If the edge functions dont have the same sign,
        //  the point is outside the triangle
        if ((edge0 < 0 || edge1 < 0 || edge2 < 0) 
            && 
            (edge0 > 0 || edge1 > 0 || edge2 > 0))
        {
            return false;
        }

        // Compute scaled barycentric coordinates
        Float det = edge0 + edge1 + edge2;
        if (det == 0)
            return false;
  
        // Compute scaled hit distance
        lp0.z *= Sz;
        lp1.z *= Sz;
        lp2.z *= Sz;

        Float scaledOffset = edge0 * lp0.z + edge1 * lp1.z + edge2 * lp2.z;

        // If the determinant has the wrong sign, the ray is outside the triangle
        if (det < 0 
            && 
            (scaledOffset >= 0 || scaledOffset < worldRay.maximumOffset * det))
        {
            return false;
        }
        else if (det > 0 
            && 
            (scaledOffset <= 0 || scaledOffset > worldRay.maximumOffset * det))
        {
            return false;
        }

        // Now there is an intersection

        // Compute barycentric coordinates
        Float invDet = 1 / det;
        Float b0 = edge0 * invDet;
        Float b1 = edge1 * invDet;
        Float b2 = edge2 * invDet;

        // Compute hit point
        Float currentHitOffset = scaledOffset * invDet;
        if (currentHitOffset > hitOffset)
            return false;
        
        hitOffset = currentHitOffset;

        // Move the hit point conservatively to avoid precision errors
        Float maxZt = maxComponent(abs(Vector3f(lp0.z, lp1.z, lp2.z)));
        Float deltaZ = errorBound(3) * maxZt;
        Float maxXt = maxComponent(abs(Vector3f(lp0.x, lp1.x, lp2.x)));
        Float maxYt = maxComponent(abs(Vector3f(lp0.y, lp1.y, lp2.y)));
        Float deltaX = errorBound(5) * (maxXt + maxZt);
        Float deltaY = errorBound(5) * (maxYt + maxZt);

        Float deltaE = 2 * (errorBound(2) * maxXt * maxYt + deltaY * maxXt +
                          deltaX * maxYt);

        Float maxE = maxComponent(abs(Vector3f(edge0, edge1, edge2)));
        
        Float deltaT = 3 * (errorBound(3) * maxE * maxZt + deltaE * maxZt +
                            deltaZ * maxE) * abs(invDet);
        
        // Check the offset against the error bounds
        if (hitOffset <= deltaT)
            return false;

        Float xAbsSum = (abs(b0 * wp0.x) + abs(b1 * wp1.x) +
                        abs(b2 * wp2.x));
        Float yAbsSum = (abs(b0 * wp0.y) + abs(b1 * wp1.y) +
                            abs(b2 * wp2.y));
        Float zAbsSum = (abs(b0 * wp0.z) + abs(b1 * wp1.z) +
                            abs(b2 * wp2.z));
        Vector3f pError = errorBound(7) * Vector3f(xAbsSum, yAbsSum, zAbsSum);


        // Compute triangle partial derivatives
        Vector3f dpdu, dpdv;
        Point2f surfaceVertices[3];
        getSurfaceCoordinates(surfaceVertices);

        // Compute deltas for triangle partial derivatives
        Vector2f duv02 = surfaceVertices[0] - surfaceVertices[2];
        Vector2f duv12 = surfaceVertices[1] - surfaceVertices[2];
        Vector3f dp02 = wp0 - wp2;
        Vector3f dp12 = wp1 - wp2;

        Float determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];

        if (abs(determinant) >= FLOAT_ERROR_MARGIN)
        {
            Float invDet = 1 / determinant;
            dpdu = (duv12[1] * dp02 - duv02[1] * dp12) * invDet;
            dpdv = (-duv12[0] * dp02 + duv02[0] * dp12) * invDet;
        }
        else
        {
            // Handle degenerate triangle
            generateCoordinateSystem(normalize(cross(wp2 - wp0, wp1 - wp0)), dpdu, dpdv);
        }

        // Interpolate hit coordinates using barycentric formula 
        //  (more precise than the ray parametric equation)
        Point2f surfacePoint = b0 * surfaceVertices[0] + 
                                b1 * surfaceVertices[1] + 
                                b2 * surfaceVertices[2];

        // Evaluate alpha texture if present
        if (testAlphaTexture
            && !evalAlphaTexture(dpdu, dpdv, surfacePoint))
        {
            return false;
        }

        return true;
    }


    __host__ __device__
    Float area() const
    {
        const Point3f &p0 = mesh->vertices[vertexIndices[0]];
        const Point3f &p1 = mesh->vertices[vertexIndices[1]];
        const Point3f &p2 = mesh->vertices[vertexIndices[2]];

        return 0.5 * cross(p1 - p0, p2 - p0).length();
    }

    __host__ __device__
    Point3f centroid() const
    {
        const Point3f &p0 = mesh->vertices[vertexIndices[0]];
        const Point3f &p1 = mesh->vertices[vertexIndices[1]];
        const Point3f &p2 = mesh->vertices[vertexIndices[2]];

        return (p0 + p1 + p2) / 3;
    }

    __host__ __device__
    void setMeshData(TriangleMeshData *_mesh)
    {
        mesh = _mesh;
    }

    __host__ __device__
    TriangleMeshData* getMeshData() const
    {
        return mesh;
    }

    void setMaterial(Material *_material)
    {
        material = _material;
        isTransparent = material->isTransparent();
    }

    Material* getMaterial() const
    {
        return material;
    }


    // Operator <<
    friend std::ostream& operator<<(std::ostream &out, const Triangle &triangle)
    {
        out << "Triangle: " << std::endl;
        out << "Vertex 0: " << triangle.mesh->vertices[triangle.vertexIndices[0]] << std::endl;
        out << "Vertex 1: " << triangle.mesh->vertices[triangle.vertexIndices[1]] << std::endl;
        out << "Vertex 2: " << triangle.mesh->vertices[triangle.vertexIndices[2]] << std::endl;

        return out;
    }
};


class triangleWorldBoundFunctor
{
    public:

    __host__ __device__
    triangleWorldBoundFunctor() {}

    __host__ __device__
    inline Bound3f operator() (const Triangle &triangle) const
    {
        return triangle.worldBound();
    }
};


class Morton3DFunctor
{
    public:
    Bound3f worldBounding;

    __host__ __device__
    Morton3DFunctor(const Bound3f& worldBounding) 
    : worldBounding(worldBounding) 
    {}

    __host__ __device__
    unsigned int operator()(const Triangle &entity)
    {
        Point3f p = worldBounding.getUnitCubeCoordinates(entity.centroid());
        return morton3D(p.x, p.y, p.z);
    }
};

// Sorts the data array using the Morton codes.
void sort_by_morton(const thrust::device_vector<Triangle> &initial_data, 
                    thrust::device_vector<unsigned int> &morton_positions,
                    thrust::device_vector<unsigned int> &sorted_element_ids)
{
    // Transform triangles into bounding boxes
    thrust::device_vector<Bound3f> bounding_boxes(initial_data.size());
    thrust::transform(initial_data.begin(), initial_data.end(), bounding_boxes.begin(), triangleWorldBoundFunctor());

    // Compute sum of all the bounding boxes
    Bound3f worldBounding = thrust::reduce(bounding_boxes.begin(), bounding_boxes.end(), Bound3f());
    Morton3DFunctor morton_functor(worldBounding);

    bounding_boxes.clear();


    // Calculate the Morton positions for each element.
    thrust::transform(initial_data.begin(), 
                    initial_data.end(), 
                    morton_positions.begin(), 
                    morton_functor);

    // Initialize element ids
    thrust::sequence(sorted_element_ids.begin(), sorted_element_ids.end());

    // Create a zip_iterator for both Morton codes and element IDs
    thrust::device_vector<thrust::tuple<unsigned int, unsigned int>> zipped_data(initial_data.size());
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(morton_positions.begin(), sorted_element_ids.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(morton_positions.end(), sorted_element_ids.end())),
                    zipped_data.begin(),
                    thrust::identity<thrust::tuple<unsigned int, unsigned int>>());

    // Sort using the zipped data (Morton positions and element IDs)
    thrust::sort(zipped_data.begin(), zipped_data.end());

    // Unzip the sorted data back into Morton codes and element IDs
    thrust::transform(zipped_data.begin(),
                    zipped_data.end(),
                    thrust::make_zip_iterator(thrust::make_tuple(morton_positions.begin(), sorted_element_ids.begin())),
                    thrust::identity<thrust::tuple<unsigned int, unsigned int>>());
}
