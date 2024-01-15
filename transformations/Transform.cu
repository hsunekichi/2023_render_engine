#pragma once

#include "../cuda_libraries/Matrix4x4.cu"
#include "../cuda_libraries/math.cu"
#include "../cuda_libraries/types.h"

#include "../geometry/Vector.cu"
#include "../geometry/Point.cu"
#include "../geometry/Normal.cu"
#include "../geometry/Ray.cu"
#include "../geometry/RayDifferential.cu"
#include "../geometry/BoundingBox.cu"
#include "../light/Photon.cu"

class Transform
{
    public:
    Matrix4x4f m;

    //friend class AnimatedTransform;
    //friend struct Quaternion;

    public:

    __host__ __device__
    Transform() 
    { }

    __host__ __device__
    Transform(const Matrix4x4f &m) 
        : m(m)
    { }

    __host__ __device__
    Transform(const Matrix4x4f &m, const Matrix4x4f &mInv) 
        : m(m)
    { }

    Transform* toGPU() const
    {
        // Allocate memory on gpu
        Transform *gpu_transform;
        cudaMalloc(&gpu_transform, sizeof(Transform));

        // Copy the data to the gpu
        cudaMemcpy(gpu_transform, this, sizeof(Transform), cudaMemcpyHostToDevice);

        return gpu_transform;
    }

    friend std::ostream &operator<<(std::ostream &os, const Transform &t) 
    {
        Matrix4x4f temp = t.getMatrix();

        // Puts 0 instead of very small numbers
        for (int i = 0; i < 4; ++i) 
        {
            for (int j = 0; j < 4; ++j) 
            {
                if (abs(temp[i][j]) < FLOAT_ERROR_MARGIN) 
                    temp[i][j] = 0.0f;
            }
        }

        os << "[ " << temp[0][0] << " " << temp[0][1] << " " << temp[0][2] << " " << temp[0][3] << " ]\n"
           << "[ " << temp[1][0] << " " << temp[1][1] << " " << temp[1][2] << " " << temp[1][3] << " ]\n"
           << "[ " << temp[2][0] << " " << temp[2][1] << " " << temp[2][2] << " " << temp[2][3] << " ]\n"
           << "[ " << temp[3][0] << " " << temp[3][1] << " " << temp[3][2] << " " << temp[3][3] << " ]";
        
        return os;
    }

    __host__ __device__
    void print() const
    {
        Matrix4x4f temp = m;

        // Puts 0 instead of very small numbers
        for (int i = 0; i < 4; ++i) 
        {
            for (int j = 0; j < 4; ++j) 
            {
                if (abs(temp[i][j]) < FLOAT_ERROR_MARGIN) 
                    temp[i][j] = 0.0f;
            }
        }

        printf("[ %f %f %f %f ]\n", temp[0][0], temp[0][1], temp[0][2], temp[0][3]);
        printf("[ %f %f %f %f ]\n", temp[1][0], temp[1][1], temp[1][2], temp[1][3]);
        printf("[ %f %f %f %f ]\n", temp[2][0], temp[2][1], temp[2][2], temp[2][3]);
        printf("[ %f %f %f %f ]\n", temp[3][0], temp[3][1], temp[3][2], temp[3][3]);
    }


    __host__ __device__
    friend Transform inverse(const Transform &t);

    __host__ __device__
    friend Transform transpose(const Transform &t);

    __host__ __device__
    bool operator==(const Transform &t) const {
        return t.m == m;
    }

    __host__ __device__
    bool operator!=(const Transform &t) const {
        return t.m != m;
    }

    __host__ __device__
    bool operator<(const Transform &t2) const 
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++) 
            {
                if (m[i][j] < t2.m[i][j]) 
                    return true;

                if (m[i][j] > t2.m[i][j]) 
                    return false;
            }
        }

        return false;
    }

    __host__ __device__
    bool isIdentity() const 
    {
        return (m[0][0] == 1.0f && m[0][1] == 0.0f &&
                m[0][2] == 0.0f && m[0][3] == 0.0f &&
                m[1][0] == 0.0f && m[1][1] == 1.0f &&
                m[1][2] == 0.0f && m[1][3] == 0.0f &&
                m[2][0] == 0.0f && m[2][1] == 0.0f &&
                m[2][2] == 1.0f && m[2][3] == 0.0f &&
                m[3][0] == 0.0f && m[3][1] == 0.0f &&
                m[3][2] == 0.0f && m[3][3] == 1.0f);
    }

    __host__ __device__
    const Matrix4x4f &getMatrix() const { 
        return m; 
    }


    __host__ __device__
    bool hasScale() const 
    {
        Vector3f x (1, 0, 0);
        Vector3f y (0, 1, 0);
        Vector3f z (0, 0, 1);

        const Transform &transform = *this;

        // Transform the base vectors to check for scaling.
        // Length squared is used, since if the vector is still 1
        //  the square will not do anything
        Float x_transformed = transform(x).lengthSquared();
        Float y_transformed = transform(y).lengthSquared();
        Float z_transformed = transform(z).lengthSquared();
        
        // If any of the vectors have a length different from 1
        return (abs(x_transformed - 1.0f) > FLOAT_ERROR_MARGIN ||
                abs(y_transformed - 1.0f) > FLOAT_ERROR_MARGIN ||
                abs(z_transformed - 1.0f) > FLOAT_ERROR_MARGIN); 
               
    }

    template <typename T> 
    inline 
    __host__ __device__
    Point3<T> operator()(const Point3<T> &p) const 
    {
        T x = p.x, y = p.y, z = p.z;
        T xp = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
        T yp = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
        T zp = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
        T wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];

        return Point3<T>(xp, yp, zp) / wp;
    }

    template <typename T> 
    inline 
    __host__ __device__
    Vector3<T> operator()(const Vector3<T> &v) const
    {
        T x = v.x, y = v.y, z = v.z;
        T xp = m[0][0] * x + m[0][1] * y + m[0][2] * z;
        T yp = m[1][0] * x + m[1][1] * y + m[1][2] * z;
        T zp = m[2][0] * x + m[2][1] * y + m[2][2] * z;

        return Vector3<T>(xp, yp, zp);
    }

    template <typename T> 
    __host__ __device__
    inline Normal3<T> operator()(const Normal3<T> &n) const 
    { 
        T x = n.x, y = n.y, z = n.z;

        Matrix4x4 mInv = inverse(m);

        return Normal3<T>(mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z,
                            mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z,
                            mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z);
    }

    //template <typename T> 
    //__host__ __device__
    //inline void operator()(const Normal3<T> &, Normal3<T> *nt) const {};

    __host__ __device__
    inline Ray operator()(const Ray &ray) const 
    {
        // Transform ray origin and direction
        Vector3f oError;
        Point3f origin = operator()(ray.origin, oError);
        Vector3f direction = operator()(ray.direction);

        /************ Fix float rounding errors ************/
        // Offset ray origin to edge of error bounds and compute tMax
        Float lengthSquared = direction.lengthSquared();
        Float maximumOffset = ray.maximumOffset;

        // If the ray is not null, we need to compensate the error
        if (lengthSquared > 0) 
        {
            Float dt = dot(abs(direction), oError) / lengthSquared;
            origin += direction * dt;
            maximumOffset -= dt;
        }

        return Ray(origin, direction, maximumOffset, ray.time, ray.medium, ray.cameraRay);
    }

    template <typename T>
    __host__ __device__
    inline Point3<T> operator()(const Point3<T> &p, Vector3<T> &pError)
    {
        T x = p.x, y = p.y, z = p.z;
        T xp = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
        T yp = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
        T zp = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
        T wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];

        // Compute absolute error for transformed point
        T xAbsSum = (abs(m[0][0] * x) + abs(m[0][1] * y) +
                        abs(m[0][2] * z) + abs(m[0][3]));

        T yAbsSum = (abs(m[1][0] * x) + abs(m[1][1] * y) +
                        abs(m[1][2] * z) + abs(m[1][3]));

        T zAbsSum = (abs(m[2][0] * x) + abs(m[2][1] * y) +
                        abs(m[2][2] * z) + abs(m[2][3]));

        pError += Vector3<T>(xAbsSum, yAbsSum, zAbsSum) * errorBound(4) * abs(wp);

        return Point3<T>(xp, yp, zp) / wp;
    }

    
    __host__ __device__
    inline RayDifferential operator()(const RayDifferential &rayDiff) const 
    {   
        // Transforms the ray differential as a normal ray
        Ray tr = operator()(Ray(rayDiff));
        RayDifferential ret(tr.origin, tr.direction, tr.maximumOffset, tr.time, tr.medium);

        // Transforms the differentials
        ret.hasDifferentials = rayDiff.hasDifferentials;
        ret.rxOrigin = operator()(rayDiff.rxOrigin);
        ret.ryOrigin = operator()(rayDiff.ryOrigin);
        ret.rxDirection = operator()(rayDiff.rxDirection);
        ret.ryDirection = operator()(rayDiff.ryDirection);

        return ret;
    }
    

    __host__ __device__
    Bound3f operator()(const Bound3f &b) const 
    {
        const Transform &transform = *this;

        // Transforms each corner and adds it to the bounding box
        Bound3f ret (transform(Point3f(b.pMin.x, b.pMin.y, b.pMin.z)));    

        ret += transform(Point3f(b.pMax.x, b.pMin.y, b.pMin.z));
        ret += transform(Point3f(b.pMin.x, b.pMax.y, b.pMin.z));
        ret += transform(Point3f(b.pMin.x, b.pMin.y, b.pMax.z));
        ret += transform(Point3f(b.pMin.x, b.pMax.y, b.pMax.z));
        ret += transform(Point3f(b.pMax.x, b.pMax.y, b.pMin.z));
        ret += transform(Point3f(b.pMax.x, b.pMin.y, b.pMax.z));
        ret += transform(Point3f(b.pMax.x, b.pMax.y, b.pMax.z));

        return ret;
    }

    __host__ __device__
    Transform operator*(const Transform &t2) const 
    {
        Matrix4x4f m1 = m * t2.m;

        return Transform(m1);
    }


    __host__ __device__
    Transform operator()(const Transform &b) const 
    {
        const Transform &a = (*this);
        return a * b; 
    }


    __host__ __device__
    bool swapsHandedness() const 
    {
        // A transformation swaps handedness when the determinant is negative
        Float det = 
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        return det < 0; 
    }

    //__host__ __device__
    //SurfaceInteraction *operator()(const SurfaceInteraction &si) const {return nullptr;}

    template <typename T> 
    __host__ __device__
    inline Point3<T> operator()(const Point3<T> &point, Vector3<T> &absError) const
    {
        T xAbsSum = (abs(m[0][0] * point.x) + abs(m[0][1] * point.y) +
                        abs(m[0][2] * point.z) + abs(m[0][3]));

        T yAbsSum = (abs(m[1][0] * point.x) + abs(m[1][1] * point.y) +
                        abs(m[1][2] * point.z) + abs(m[1][3]));

        T zAbsSum = (abs(m[2][0] * point.x) + abs(m[2][1] * point.y) +
                        abs(m[2][2] * point.z) + abs(m[2][3]));

        absError = Vector3<T>(xAbsSum, yAbsSum, zAbsSum) * errorBound((3));

        return operator()(point);
    }

    template <typename T> 
    __host__ __device__
    inline Point3<T> operator()(const Point3<T> &p, const Vector3<T> &pError,
                Vector3<T> &pTransError) const 
    {
        throw std::runtime_error("Transform doesn't support pError transformation");
    }

    template <typename T> 
    __host__ __device__ 
    inline Vector3<T> operator()(const Vector3<T> &v, Vector3<T> &vTransError) const 
    {
        T xAbsSum = (abs(m[0][0] * v.x) + abs(m[0][1] * v.y) +
                        abs(m[0][2] * v.z));

        T yAbsSum = (abs(m[1][0] * v.x) + abs(m[1][1] * v.y) +
                        abs(m[1][2] * v.z));

        T zAbsSum = (abs(m[2][0] * v.x) + abs(m[2][1] * v.y) +
                        abs(m[2][2] * v.z));

        vTransError = Vector3<T>(xAbsSum, yAbsSum, zAbsSum) * (errorBound(3));

        return operator()(v);
    }

    template <typename T> 
    __host__ __device__ 
    inline Vector3<T> operator()(const Vector3<T> &v, const Vector3<T> &vError,
                Vector3<T> &vTransError) const 
    {
        throw std::runtime_error("Transform doesn't support vError transformation");
    }

    __host__ __device__
    inline Ray operator()(const Ray &ray, Vector3f &oError,
        Vector3f &dError) const 
    {
        Point3f origin = operator()(ray.origin, oError);
        Vector3f direction = operator()(ray.direction, dError);

        Float tMax = ray.maximumOffset;
        Float lengthSquared = direction.lengthSquared();

        if (lengthSquared > 0) 
        {
            Float dt = dot(abs(direction), oError) / lengthSquared;
            origin += direction * dt;
            tMax -= dt;
        }

        return Ray(origin, direction, tMax, ray.time, ray.medium);
    }

    __host__ __device__
    inline Photon operator()(const Photon &photon) const 
    {
        Point3f location = operator()(photon.location);
        Ray ray = operator()(photon.ray);
        Vector3f normal = operator()(photon.surfaceNormal);

        return Photon(ray, location, normal, photon.radiance, photon.shapeId, photon.surfacePoint);
    }


    __host__ __device__
    inline Ray operator()(const Ray &ray, const Vector3f &oErrorIn,
                            const Vector3f &dErrorIn, Vector3f &oErrorOut,
                            Vector3f &dErrorOut) const 
    {
        Point3f origin = operator()(ray.origin, oErrorOut);
        Vector3f direction = operator()(ray.direction, dErrorOut);

        Float tMax = ray.maximumOffset;
        Float lengthSquared = direction.lengthSquared();

        if (lengthSquared > 0) 
        {
            Float dt = dot(abs(direction), oErrorIn) / lengthSquared;
            origin += direction * dt;
            tMax -= dt;
        }

        return Ray(origin, direction, tMax, ray.time, ray.medium);
    }

    __host__ __device__
    inline Transform* operator()(const Transform *t) const
    {
        Transform *ret = new Transform((*this)(*t));
        return ret;
    }
};

// inverse
__host__ __device__
Transform inverse(const Transform &t)
{
    return Transform(inverse(t.m));
}

__host__ __device__
Transform* inverse(const Transform *t)
{
    return new Transform(inverse(t->m));
}


__host__ __device__
Transform transpose(const Transform &t)
{
    return Transform(transpose(t.m));
}

__host__ __device__
Transform* transpose(const Transform *t)
{
    return new Transform(transpose(t->m));
}

namespace Transformations
{
__host__ __device__
Transform translate(const Vector3f &delta)
{
    Matrix4x4f m(1, 0, 0, delta.x,
                0, 1, 0, delta.y,
                0, 0, 1, delta.z,
                0, 0, 0, 1);

    Matrix4x4f mInv(1, 0, 0, -delta.x,
                    0, 1, 0, -delta.y,
                    0, 0, 1, -delta.z,
                    0, 0, 0, 1);

    return Transform(m, mInv);
}

__host__ __device__
Transform scale(Float x, Float y, Float z)
{
    Matrix4x4f m(x, 0, 0, 0,
                0, y, 0, 0,
                0, 0, z, 0,
                0, 0, 0, 1);

    Matrix4x4f mInv(1.0f / x, 0, 0, 0,
                    0, 1.0f / y, 0, 0,
                    0, 0, 1.0f / z, 0,
                    0, 0, 0, 1);

    return Transform(m, mInv);
}

__host__ __device__
Transform rotateX(Float angle)
{
    Float sinTheta = sin(radians(angle));
    Float cosTheta = cos(radians(angle));

    Matrix4x4f m(1, 0, 0, 0,
                0, cosTheta, -sinTheta, 0,
                0, sinTheta, cosTheta, 0,
                0, 0, 0, 1);

    return Transform(m, transpose(m));
}

__host__ __device__
Transform rotateY(Float angle)
{
    Float sinTheta = sin(radians(angle));
    Float cosTheta = cos(radians(angle));

    Matrix4x4f m(cosTheta, 0, sinTheta, 0,
                0, 1, 0, 0,
                -sinTheta, 0, cosTheta, 0,
                0, 0, 0, 1);

    return Transform(m, transpose(m));
}


__host__ __device__
Transform rotateZ(Float angle)
{
    Float sinTheta = sin(radians(angle));
    Float cosTheta = cos(radians(angle));

    Matrix4x4f m(cosTheta, -sinTheta, 0, 0,
                sinTheta, cosTheta, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1);

    return Transform(m, transpose(m));
}


__host__ __device__
Transform baseChange(const Point3f &o, 
        const Vector3f &u, 
        const Vector3f &v, 
        const Vector3f &w)
{
    Matrix4x4f m(u.x, v.x, w.x, o.x,
                 u.y, v.y, w.y, o.y,
                 u.z, v.z, w.z, o.z,
                 0.0f, 0.0f, 0.0f, 1.0f);

    return Transform(m);
}

__host__ __device__
Transform* newBaseChange(const Point3f &o, 
        const Vector3f &u, 
        const Vector3f &v, 
        const Vector3f &w)
{
    Matrix4x4f m(u.x, v.x, w.x, o.x,
                 u.y, v.y, w.y, o.y,
                 u.z, v.z, w.z, o.z,
                 0.0f, 0.0f, 0.0f, 1.0f);

    return new Transform(m);
}


__host__ __device__
Transform rotate(Float theta, const Vector3f &axis) 
{
    Vector3f a = normalize(axis);
    Float sinTheta = sin(radians(theta));
    Float cosTheta = cos(radians(theta));

    Matrix4x4f matrix;
    
    // Compute rotation of first basis vector
    matrix[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
    matrix[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
    matrix[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
    matrix[0][3] = 0;

    // Second basis vector
    matrix[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
    matrix[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
    matrix[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
    matrix[1][3] = 0;
    
    // Third basis vector
    matrix[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
    matrix[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
    matrix[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
    matrix[2][3] = 0;

    return Transform(matrix, transpose(matrix));
}


__host__ __device__ 
Transform lookAt(const Point3f &pos, 
        const Point3f &look,
        const Vector3f &up) 
{
    Matrix4x4f cameraToWorld;

    // Initialize fourth column of viewing matrix
    cameraToWorld[0][3] = pos.x;
    cameraToWorld[1][3] = pos.y;
    cameraToWorld[2][3] = pos.z;
    cameraToWorld[3][3] = 1;

    // Initialize first three columns of viewing matrix
    Vector3f dir = normalize(look - pos);   // Look direction vector
    Vector3f right = normalize(cross(normalize(up), dir));
    Vector3f newUp = cross(dir, right);

    cameraToWorld[0][0] = right.x;
    cameraToWorld[1][0] = right.y;
    cameraToWorld[2][0] = right.z;
    cameraToWorld[3][0] = 0.;
    cameraToWorld[0][1] = newUp.x;
    cameraToWorld[1][1] = newUp.y;
    cameraToWorld[2][1] = newUp.z;
    cameraToWorld[3][1] = 0.;
    cameraToWorld[0][2] = dir.x;
    cameraToWorld[1][2] = dir.y;
    cameraToWorld[2][2] = dir.z;
    cameraToWorld[3][2] = 0.;

    return Transform(inverse(cameraToWorld), cameraToWorld);
}

};
