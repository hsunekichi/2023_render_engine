#pragma once

#include "types.h"
#include "../geometry/Vector.cu"
#include "../radiometry/Spectrum.cu"

template <typename T>
class Matrix3x3
{
public:

    T m00, m01, m02;
    T m10, m11, m12;
    T m20, m21, m22;

public:

    Matrix3x3() = default;

    Matrix3x3(T m00, T m01, T m02,
              T m10, T m11, T m12,
              T m20, T m21, T m22)
        : m00(m00), m01(m01), m02(m02),
          m10(m10), m11(m11), m12(m12),
          m20(m20), m21(m21), m22(m22)
    {
    }

    Matrix3x3(const Matrix3x3<T>& other)
        : m00(other.m00), m01(other.m01), m02(other.m02),
          m10(other.m10), m11(other.m11), m12(other.m12),
          m20(other.m20), m21(other.m21), m22(other.m22)
    {
    }

    Matrix3x3(Matrix3x3<T>&& other)
        : m00(other.m00), m01(other.m01), m02(other.m02),
          m10(other.m10), m11(other.m11), m12(other.m12),
          m20(other.m20), m21(other.m21), m22(other.m22)
    {
    }

    Matrix3x3<T>& operator=(const Matrix3x3<T>& other)
    {
        m00 = other.m00;
        m01 = other.m01;
        m02 = other.m02;
        m10 = other.m10;
        m11 = other.m11;
        m12 = other.m12;
        m20 = other.m20;
        m21 = other.m21;
        m22 = other.m22;
        return *this;
    }

    Matrix3x3<T>& operator=(Matrix3x3<T>&& other)
    {
        m00 = other.m00;
        m01 = other.m01;
        m02 = other.m02;
        m10 = other.m10;
        m11 = other.m11;
        m12 = other.m12;
        m20 = other.m20;
        m21 = other.m21;
        m22 = other.m22;
        return *this;
    }

    Matrix3x3<T> operator*(const Matrix3x3<T>& other) const
    {
        return Matrix3x3<T>(
            m00 * other.m00 + m01 * other.m10 + m02 * other.m20,
            m00 * other.m01 + m01 * other.m11 + m02 * other.m21,
            m00 * other.m02 + m01 * other.m12 + m02 * other.m22,
            m10 * other.m00 + m11 * other.m10 + m12 * other.m20,
            m10 * other.m01 + m11 * other.m11 + m12 * other.m21,
            m10 * other.m02 + m11 * other.m12 + m12 * other.m22,
            m20 * other.m00 + m21 * other.m10 + m22 * other.m20,
            m20 * other.m01 + m21 * other.m11 + m22 * other.m21,
            m20 * other.m02 + m21 * other.m12 + m22 * other.m22);
    }

    Matrix3x3<T>& operator*=(const Matrix3x3<T>& other)
    {
        *this = *this * other;
        return *this;
    }

    Matrix3x3<T> operator*(T scalar) const
    {
        return Matrix3x3<T>(
            m00 * scalar, m01 * scalar, m02 * scalar,
            m10 * scalar, m11 * scalar, m12 * scalar,
            m20 * scalar, m21 * scalar, m22 * scalar);
    }

    Matrix3x3<T>& operator*=(T scalar)
    {
        *this = *this * scalar;
        return *this;
    }

    Matrix3x3<T> operator+(const Matrix3x3<T>& other) const
    {
        return Matrix3x3<T>(
            m00 + other.m00, m01 + other.m01, m02 + other.m02,
            m10 + other.m10, m11 + other.m11, m12 + other.m12,
            m20 + other.m20, m21 + other.m21, m22 + other.m22);
    }

    Matrix3x3<T>& operator+=(const Matrix3x3<T>& other)
    {
        *this = *this + other;
        return *this;
    }

    Matrix3x3<T> operator-(const Matrix3x3<T>& other) const
    {
        return Matrix3x3<T>(
            m00 - other.m00, m01 - other.m01, m02 - other.m02,
            m10 - other.m10, m11 - other.m11, m12 - other.m12,
            m20 - other.m20, m21 - other.m21, m22 - other.m22);
    }

    // Multiply Vector3f
    Vector3<T> operator*(const Vector3<T>& vector) const
    {
        return Vector3<T>(
            m00 * vector.x + m01 * vector.y + m02 * vector.z,
            m10 * vector.x + m11 * vector.y + m12 * vector.z,
            m20 * vector.x + m21 * vector.y + m22 * vector.z);
    }

    // Multiply spectrum
    Spectrum operator*(const Spectrum& v) const
    {
        double x = m00 * v.getR() + m01 * v.getG() + m02 * v.getB();
        double y = m10 * v.getR() + m11 * v.getG() + m12 * v.getB();
        double z = m20 * v.getR() + m21 * v.getG() + m22 * v.getB();
        return Spectrum(x, y, z);
    }

};