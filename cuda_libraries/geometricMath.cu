#pragma once

#include "../geometry/Point.cu"
#include "../geometry/Vector.cu"
#include "../geometry/BoundingBox.cu"
#include "../light/Photon.cu"
#include "../transformations/Transform.cu"
#include "../geometry/Complex.cu"

#include "types.h"
#include "math.cu"


// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__host__ __device__
inline unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__host__ __device__
inline unsigned int morton3D(Float x, Float y, Float z)
{
    x = min(max(x * 1024.0, 0.0), 1023.0);
    y = min(max(y * 1024.0, 0.0), 1023.0);
    z = min(max(z * 1024.0, 0.0), 1023.0);

    float fx = x;
    float fy = y;
    float fz = z;

    unsigned int xx = expandBits((unsigned int)fx);
    unsigned int yy = expandBits((unsigned int)fy);
    unsigned int zz = expandBits((unsigned int)fz);

    return xx * 4 + yy * 2 + zz;
}


// Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)
__host__ __device__
inline void to_spherical(Point3f p, Float &radius, Float &theta, Float &phi) 
{   
    radius = distance(p, Point3f(0, 0, 0));
    theta = atan2(sqrt(p.x * p.x + p.y * p.y), p.z);
    phi = atan2(p.y, p.x);

    theta = degrees(theta);
    phi = degrees(phi);
}

// Converts a spherical coordinate (radius, theta, phi) into a cartesian one (x, y, z)
__host__ __device__
inline Point3f to_cartesian(const Float &radius, const Float &theta, const Float &phi)
{
    Point3f cartesian;

    Float th = radians(theta);
    Float ph = radians(phi);

    cartesian.x = radius * cos(ph) * sin(th);
    cartesian.y = radius * sin(ph) * sin(th);
    cartesian.z = radius * cos(th);
    
    return cartesian;
}

__host__ __device__
inline Vector3f rotateVector(const Vector3f &v, const Float &theta, const Float &phi)
{
    Vector3f cartesian;

    Float th = degrees(theta);
    Float ph = degrees(phi);

    Transform thRotation = Transformations::rotate(th, Vector3f(1, 0, 0));
    Transform phRotation = Transformations::rotate(ph, Vector3f(0, 1, 0));

    cartesian = thRotation(phRotation(v));

    return cartesian;
}


inline Vector3f sphericalDirection(Float sinTheta, Float cosTheta, Float phi) 
{
    return Vector3f(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// Returns the theta (longitude) direction in spherical coordinates
__host__ __device__
inline Float sphericalTheta(const Vector3f &p) 
{
    return acos(clamp(p.z, -1.0, 1.0));
}

// Returns the phi (azimut) direction in spherical coordinates
__host__ __device__
inline Float sphericalPhi(const Vector3f &p) 
{
    Float phi = atan2(p.y, p.x);
    return (phi < 0.0) ? phi + 2.0 * PI : phi;
}

template <typename T>
__host__ __device__
inline Point3<T> clamp (Point3<T> p, Point3<T> a, Point3<T> b)
{
    return Point3<T>(clamp(p.x, a.x, b.x), clamp(p.y, a.y, b.y), clamp(p.z, a.z, b.z));
}

template <typename T>
__host__ __device__
inline Point2<T> clamp (Point2<T> p, Point2<T> a, Point2<T> b)
{
    return Point2<T>(clamp(p.x, a.x, b.x), clamp(p.y, a.y, b.y));
}

__host__ __device__
inline Spectrum clamp(Spectrum s, double min, double max)
{
    return Spectrum(clamp(s.getR(), min, max), clamp(s.getG(), min, max), clamp(s.getB(), min, max));
}

__host__ __device__
inline bool sameHemisphere(const Vector3f &v1, const Vector3f &v2) 
{
    return v1.z * v2.z > 0.0;
}

__host__ __device__
inline Float cosine(Vector3f v1, Vector3f v2)
{
    return dot(v1, v2) / (v1.length() * v2.length());
}

__host__ __device__
inline Float absCosine(Vector3f v1, Vector3f v2)
{
    return abs(dot(v1, v2)) / (v1.length() * v2.length());
}

__host__ __device__
inline Float absCosine(Vector3f v)
{
    return abs(v.z / v.length());
}

__host__ __device__
inline Float cosine(Vector3f v)
{
    return v.z / v.length();
}

__host__ __device__
inline Float lightDecay(const Float lengthSquared)
{
    return 1 / lengthSquared;
    
    // This alternative to 1/d^2 does not have a singularity at d=0
    Float length = sqrt(lengthSquared);
    return (1 - length/sqrt(lengthSquared+4))/2;
}

template <typename T> 
__host__ __device__
inline Normal3<T> faceforward(const Normal3<T> &n, const Normal3<T> &v) 
{
    if (dot(n, v) < 0.f) 
        return -n;
    else
        return n;
}

__host__ __device__
inline Vector3f refract(Vector3f direction, Vector3f normal, Float etaI, Float etaT)
{
    Float cosThetaI = dot(normal, direction);
    Float sin2ThetaI = max((Float)0, 1 - cosThetaI * cosThetaI);
    Float sin2ThetaT = etaI * etaI / (etaT * etaT) * sin2ThetaI;

    if (sin2ThetaT >= 1)
        return Vector3f(0, 0, 0);

    Float cosThetaT = sqrt(1 - sin2ThetaT);

    return etaI * direction + (etaI * cosThetaI - cosThetaT) * normal;
}

__host__ __device__
Vector3f reflect(Vector3f cameraDir, Vector3f n) {
    return -cameraDir + 2 * dot(cameraDir, n) * n;
}

__host__ __device__
Float fresnelDielectric(Float cosThetaOrigin, Float refrIndexOrigin, Float refrIndexDestiny) 
{
    /***** Check for entering or exiting ****/
    cosThetaOrigin = clamp(cosThetaOrigin, -1, 1);
    bool isEntering = cosThetaOrigin > 0;

    if (!isEntering) 
    {
        swap(refrIndexOrigin, refrIndexOrigin);
        cosThetaOrigin = abs(cosThetaOrigin);
    }


    /***** Get cosThetaDestiny with snells law ****/
    Float sinThetaOrigin = sqrt(max((Float)0, 1 - cosThetaOrigin * cosThetaOrigin));
    Float sinThetaDestiny = refrIndexOrigin / refrIndexDestiny * sinThetaOrigin;

    // Check for total internal reflection
    if (sinThetaOrigin >= 1)
        return 1;

    Float cosThetaDestiny = sqrt(max((Float)0, 1 - sinThetaDestiny * sinThetaDestiny));


    /***** Compute Fresnel coefficients ****/

    Float rParallel = ((refrIndexDestiny * cosThetaOrigin) - (refrIndexDestiny * cosThetaDestiny)) 
                        /
                        ((refrIndexDestiny * cosThetaOrigin) + (refrIndexDestiny * cosThetaDestiny));

    Float rPerpendicular = ((refrIndexDestiny * cosThetaOrigin) - (refrIndexDestiny * cosThetaDestiny)) 
                            /
                            ((refrIndexDestiny * cosThetaOrigin) + (refrIndexDestiny * cosThetaDestiny));

    Float coefficient = (rParallel*rParallel + rPerpendicular*rPerpendicular) / 2*abs(cosThetaOrigin);

    return coefficient;
}

__host__ __device__
Float fresnelConductor(Float cosThetaI, Complex eta)
{
    cosThetaI = clamp(cosThetaI, 0, 1);

    Float sin2ThetaI = 1 - pow2(cosThetaI);

    Complex sin2ThetaT = sin2ThetaI / pow2(eta);
    Complex cosThetaT = sqrt(1 - sin2ThetaT);

    Complex r_parl = (eta * Complex(cosThetaI) - cosThetaT) /
                     (eta * Complex(cosThetaI) + cosThetaT);

    Complex r_perp = (cosThetaI - eta * cosThetaT) /
                     (cosThetaI + eta * cosThetaT);

    Float result = (norm(r_parl) + norm(r_perp)) / 2;

    return result;
}

__host__ __device__
Spectrum fresnelConductor(Float cosThetaI, Spectrum eta, Spectrum k)
{
    Float fresnelR = fresnelConductor(cosThetaI, Complex(eta.getR(), k.getR()));
    Float fresnelG = fresnelConductor(cosThetaI, Complex(eta.getG(), k.getG()));
    Float fresnelB = fresnelConductor(cosThetaI, Complex(eta.getB(), k.getB()));

    return Spectrum(fresnelR, fresnelG, fresnelB);
}




__host__ __device__
Float cos2Theta(const Vector3f &w) 
{
    return w.z * w.z;
}

__host__ __device__
Float sin2Theta(const Vector3f &w) 
{
    return max(0, 1 - cos2Theta(w));
}


__host__ __device__
Float sinTheta(const Vector3f &w) 
{
    return sqrt(sin2Theta(w));
}

__host__ __device__
Float sinPhi(Vector3f w) 
{
    Float sinTh = sinTheta(w);
    return (sinTh == 0) ? 0 : clamp(w.y / sinTh, -1, 1);
}

__host__ __device__
Float sin2Phi(const Vector3f &w) 
{
    return sinPhi(w) * sinPhi(w);
}

__host__ __device__
Float absSin2Phi(const Vector3f &w) 
{
    return abs(sin2Phi(w));
}

__host__ __device__
Float absSinPhi(const Vector3f &w) 
{
    return abs(sinPhi(w));
}


__host__ __device__
Float cosPhi(Vector3f w) 
{
    Float sinTh = sinTheta(w);
    return (sinTh == 0) ? 1 : clamp(w.x / sinTh, -1, 1);
}

__host__ __device__
Float absCosPhi(const Vector3f &w) 
{
    return abs(cosPhi(w));
}

__host__ __device__
Float cos2Phi(const Vector3f &w) 
{
    return cosPhi(w) * cosPhi(w);
}

__host__ __device__
Float absCos2Phi(const Vector3f &w) 
{
    return abs(cos2Phi(w));
}


__host__ __device__
Float tan2Theta(const Vector3f &w) 
{
    return sin2Theta(w) / cos2Theta(w);
}

__host__ __device__
Float tanTheta(const Vector3f &w) 
{
    return sinTheta(w) / cosine(w);
}



__host__ __device__
Spectrum schlickFresnel(Float cosTheta, Spectrum ks) 
{
    return ks + pow5(1 - cosTheta) * (Spectrum(1) - ks);
}

Float geometricAttenuation(Float lambda1, Float lambda2) 
{
    return 1 / (1 + lambda1 + lambda2);
}

Float geometricAttenuation(Float lambda)
{
    return 1 / (1 + lambda);
}

Float beckmannDistribution(Vector3f halfVector, Float alphaX, Float alphaY) 
{
    Float tan2Th = tan2Theta(halfVector);

    if (isinf(tan2Th)) 
        return 0;

    Float cos4Theta = cos2Theta(halfVector) * cos2Theta(halfVector);
    Float denom = (PI * alphaX * alphaY * cos4Theta);

    Float p1 = absCos2Phi(halfVector) / pow2(alphaX);
    Float p2 = absSin2Phi(halfVector) / pow2(alphaY);
    Float p = p1 + p2;

    Float result = exp(-tan2Th * p) / denom;

    return result;
}

Float beckmannDistributionLambda(const Vector3f &rayDir, 
        Float alphax, Float alphay) 
{
    Float absTanTheta = abs(tanTheta(rayDir));
    
    if (isinf(absTanTheta)) 
        return 0;

    Float sin2Ph = absSin2Phi(rayDir);
    Float cos2Ph = absCos2Phi(rayDir);

    // Compute alpha direction for rayDir
    Float alpha = sqrt(cos2Ph * alphax * alphax +
                               sin2Ph * alphay * alphay);

    Float a = 1 / (alpha * absTanTheta);

    if (a >= 1.6)
        return 0;

    return (1 - 1.259 * a + 0.396 * a * a) /
           (3.535 * a + 2.181 * a * a);
}


__host__ __device__
Spectrum dipoleDiffusionAproximation(Point3f point, Photon photon, Spectrum sigmaS, Spectrum sigmaA, Float g, Float eta)
{
    // Compute isotropic phase function
    Spectrum _sigmaS = (1 - g) * sigmaS;
    Spectrum _sigmaT = _sigmaS + sigmaA;
    Spectrum _alpha = _sigmaS / _sigmaT;

    // Effective transport coefficient
    Spectrum sigmaTr = sqrt(3 * sigmaA * _sigmaT);
    
    // Aproximation for the diffuse reflectance (fresnel)
    Float Fdr = (-1.440 / pow2(eta)) + (0.710 / eta) + 0.668 + 0.0636 * eta;
    Float A = (1 + Fdr) / (1 - Fdr);    // Boundary condition for the change between refraction indexes

    Float r = distance(point, photon.location);
    Spectrum lu = 1 / _sigmaT;
    Spectrum zr = lu;
    Spectrum zv = lu * (1 + 4/3 * A);

    Spectrum distanceR = sqrt(pow2(r) + pow2(zr)); 
    Spectrum distanceV = sqrt(pow2(r) + pow2(zv)); 

    // Compute main formula
    Spectrum C1 = zr * (sigmaTr + 1/distanceR);
    Spectrum C2 = zv * (sigmaTr + 1/distanceV);

    Spectrum m2 = C1 * exp(-sigmaTr * distanceR) / pow2(distanceR) + C2 * exp(-sigmaTr * distanceV) / pow2(distanceV);
    Spectrum result = _alpha * INV_FOUR_PI * m2;

    return result;
}

__host__ __device__
Spectrum impPaper_dipoleDiffusionAproximation(Point3f point, Photon photon, Spectrum sigmaS, Spectrum sigmaA, Float g, Float eta)
{
    // Compute isotropic phase function
    Spectrum _sigmaS = (1 - g) * sigmaS;
    Spectrum _sigmaT = _sigmaS + sigmaA;
    Spectrum _alpha = _sigmaS / _sigmaT;

    // Effective transport coefficient
    Spectrum sigmaTr = sqrt(3 * sigmaA * _sigmaT);
    
    // Aproximation for the diffuse reflectance (fresnel)
    Float Fdr = (-1.440 / pow2(eta)) + (0.710 / eta) + 0.668 + 0.0636 * eta;
    Float A = (1 + Fdr) / (1 - Fdr);    // Boundary condition for the change between refraction indexes

    Float r = distance(point, photon.location);
    Spectrum zr = sqrt(3*(1-_alpha)) / sigmaTr;
    Spectrum zv = zr * A;

    Spectrum distanceR = sqrt(pow2(r) + pow2(zr)); 
    Spectrum distanceV = sqrt(pow2(r) + pow2(zv)); 
    Spectrum sigmaTrDr = sigmaTr * distanceR;
    Spectrum sigmaTrDv = sigmaTr * distanceV;

    Spectrum Rd = (sigmaTrDr+1) * exp(-sigmaTrDr) * zr/pow3(distanceR)
                    +
                    (sigmaTrDv+1) * exp(-sigmaTrDv) * zv/pow3(distanceV);

    Spectrum C1 = zr * (sigmaTr + 1/distanceR);

    Spectrum result = C1 * Rd * (1-Fdr) * _alpha;

    return result;
}

__host__ __device__
Spectrum dipoleDiffusionAproximation(Point3f point, Photon photon, Spectrum sigmaT, Float g, Float eta)
{
    Spectrum sigmaS = sigmaT * (1 - g);
    Spectrum sigmaA = sigmaT - sigmaS;

    return dipoleDiffusionAproximation(point, photon, sigmaS, sigmaA, g, eta);
}


__host__ __device__
Spectrum diffusionTermBSSRDF(
        Point3f cameraP, 
        Vector3f cameraDir, 
        Vector3f cameraNormal,
        Photon photon, 
        Spectrum sigmaS, Spectrum sigmaA, 
        Float g, Float eta)
{
    // Assuming Rd is precomputed
    Spectrum Rd = impPaper_dipoleDiffusionAproximation(cameraP, photon, sigmaS, sigmaA, g, eta);

    Float cosWi = cosine(photon.surfaceNormal, -photon.ray.direction);
    Float cosWo = cosine(cameraNormal, cameraDir);

    // Assuming Ft is a function that returns the Fresnel term
    double Ft_o = 1 - fresnelDielectric(cosWo, 1.0, eta);
    double Ft_i = 1 - fresnelDielectric(cosWi, 1.0, eta);

    // Compute the diffusion term
    Spectrum Sd = INV_PI * Ft_i * Rd * Ft_o;

    return Sd;
}