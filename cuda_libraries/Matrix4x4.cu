#pragma once

#include "types.h"

template <typename T>
class Matrix4x4
{
private:

    T m[4][4];

public:

    __host__ __device__
    Matrix4x4() // Creates the identity matrix
    {
        m[0][0] = 1; m[0][1] = 0; m[0][2] = 0; m[0][3] = 0;
        m[1][0] = 0; m[1][1] = 1; m[1][2] = 0; m[1][3] = 0;
        m[2][0] = 0; m[2][1] = 0; m[2][2] = 1; m[2][3] = 0;
        m[3][0] = 0; m[3][1] = 0; m[3][2] = 0; m[3][3] = 1;
    }

    // Initialize with array
    __host__ __device__
    Matrix4x4 (T other[])
    {
        m[0][0] = other[0]; m[0][1] = other[1]; m[0][2] = other[2]; m[0][3] = other[3];
        m[1][0] = other[4]; m[1][1] = other[5]; m[1][2] = other[6]; m[1][3] = other[7];
        m[2][0] = other[8]; m[2][1] = other[9]; m[2][2] = other[10]; m[2][3] = other[11];
        m[3][0] = other[12]; m[3][1] = other[13]; m[3][2] = other[14]; m[3][3] = other[15];
    }

    __host__ __device__
    Matrix4x4(T m00, T m01, T m02, T m03,
              T m10, T m11, T m12, T m13,
              T m20, T m21, T m22, T m23,
              T m30, T m31, T m32, T m33)
    {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
    }

    __host__ __device__
    Matrix4x4(const Matrix4x4<T>& other)
    {
        m[0][0] = other.m[0][0]; m[0][1] = other.m[0][1]; m[0][2] = other.m[0][2]; m[0][3] = other.m[0][3];
        m[1][0] = other.m[1][0]; m[1][1] = other.m[1][1]; m[1][2] = other.m[1][2]; m[1][3] = other.m[1][3];
        m[2][0] = other.m[2][0]; m[2][1] = other.m[2][1]; m[2][2] = other.m[2][2]; m[2][3] = other.m[2][3];
        m[3][0] = other.m[3][0]; m[3][1] = other.m[3][1]; m[3][2] = other.m[3][2]; m[3][3] = other.m[3][3];
    }


    __host__ __device__
    Matrix4x4<T>& operator=(const Matrix4x4<T>& other)
    {
        m[0][0] = other.m[0][0]; m[0][1] = other.m[0][1]; m[0][2] = other.m[0][2]; m[0][3] = other.m[0][3];
        m[1][0] = other.m[1][0]; m[1][1] = other.m[1][1]; m[1][2] = other.m[1][2]; m[1][3] = other.m[1][3];
        m[2][0] = other.m[2][0]; m[2][1] = other.m[2][1]; m[2][2] = other.m[2][2]; m[2][3] = other.m[2][3];
        m[3][0] = other.m[3][0]; m[3][1] = other.m[3][1]; m[3][2] = other.m[3][2]; m[3][3] = other.m[3][3];
        
        return *this;
    }


    // Matrix multiplication
    __host__ __device__
    Matrix4x4<T> operator*(Matrix4x4<T> const& obj) const
    {
        Matrix4x4<T> result;

        for (int i = 0; i < 4; i++) // row
        {
            for (int j = 0; j < 4; j++) // column
            {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++)
                {
                    result.m[i][j] += m[i][k] * obj.m[k][j];
                }
            }
        }

        return result;
    }

    __host__ __device__
    Matrix4x4<T> operator*(T const& obj) const 
    {
        Matrix4x4<T> result;

        for (int i = 0; i < 4; i++) // row
        {
            for (int j = 0; j < 4; j++) // column
            {
                result.m[i][j] = m[i][j] * obj;
            }
        }

        return result;
    }

    __host__ __device__
    Matrix4x4<T> operator+(Matrix4x4<T> const& obj) const
    {
        Matrix4x4<T> result;

        for (int i = 0; i < 4; i++) // row
        {
            for (int j = 0; j < 4; j++) // column
            {
                result.m[i][j] = m[i][j] + obj.m[i][j];
            }
        }

        return result;
    }

    __host__ __device__
    Matrix4x4<T> operator+(T const& obj) const
    {
        Matrix4x4<T> result;

        for (int i = 0; i < 4; i++) // row
        {
            for (int j = 0; j < 4; j++) // column
            {
                result.m[i][j] = m[i][j] + obj;
            }
        }

        return result;
    }

    __host__ __device__
    Matrix4x4<T> operator-(Matrix4x4<T> const& obj) const
    {
        Matrix4x4<T> result;

        for (int i = 0; i < 4; i++) // row
        {
            for (int j = 0; j < 4; j++) // column
            {
                result.m[i][j] = m[i][j] - obj.m[i][j];
            }
        }

        return result;
    }

    __host__ __device__
    Matrix4x4<T> operator-(T const& obj) const
    {
        Matrix4x4<T> result;

        for (int i = 0; i < 4; i++) // row
        {
            for (int j = 0; j < 4; j++) // column
            {
                result.m[i][j] = m[i][j] - obj;
            }
        }

        return result;
    }

    __host__ __device__
    Matrix4x4<T> operator/(T const& obj) const
    {
        Matrix4x4<T> result;

        for (int i = 0; i < 4; i++) // row
        {
            for (int j = 0; j < 4; j++) // column
            {
                result.m[i][j] = m[i][j] / obj;
            }
        }

        return result;
    }

    __host__ __device__
    Matrix4x4<T> operator+=(Matrix4x4<T> const& obj) const
    {
        *this = *this + obj;
        return *this;
    }

    __host__ __device__
    Matrix4x4<T> operator+=(T const& obj) const
    {
        *this = *this + obj;
        return *this;
    }

    __host__ __device__
    Matrix4x4<T> operator-=(Matrix4x4<T> const& obj) const
    {
        *this = *this - obj;
        return *this;
    }

    __host__ __device__
    Matrix4x4<T> operator-=(T const& obj) const
    {
        *this = *this - obj;
        return *this;
    }

    __host__ __device__
    Matrix4x4<T> operator*=(Matrix4x4<T> const& obj) const
    {
        *this = *this * obj;
        return *this;
    }

    __host__ __device__
    Matrix4x4<T> operator*=(T const& obj) const
    {
        *this = *this * obj;
        return *this;
    }

    __host__ __device__
    Matrix4x4<T> operator/=(T const& obj) const
    {
        *this = *this / obj;
        return *this;
    }

    __host__ __device__
    T* operator[](int i)
    {
        return m[i];
    }

    __host__ __device__
    T const* operator[](int i) const
    {
        return m[i];
    }

    __host__ __device__
    bool operator==(Matrix4x4<T> const& obj) const
    {
        bool result = true;

        for (int i = 0; i < 4; i++) // row
        {
            for (int j = 0; j < 4; j++) // column
            {
                if (m[i][j] != obj.m[i][j])
                {
                    result = false;
                    break;
                }
            }
        }

        return result;
    }

    __host__ __device__
    bool operator!=(Matrix4x4<T> const& obj) const
    {
        return !(*this == obj);
    }

    
    // Fills the array with the matrix data
    __host__ __device__
    void getArray(T array[]) const
    {
        array[0] = m[0][0]; array[1] = m[0][1]; array[2] = m[0][2]; array[3] = m[0][3];
        array[4] = m[1][0]; array[5] = m[1][1]; array[6] = m[1][2]; array[7] = m[1][3];
        array[8] = m[2][0]; array[9] = m[2][1]; array[10] = m[2][2]; array[11] = m[2][3];
        array[12] = m[3][0]; array[13] = m[3][1]; array[14] = m[3][2]; array[15] = m[3][3];
    }
    
};


template <typename T>
__host__ __device__
Matrix4x4<T> transpose(const Matrix4x4<T> &obj)
{
    Matrix4x4<T> result;

    for (int i = 0; i < 4; i++) // row
    {
        for (int j = 0; j < 4; j++) // column
        {
            result[i][j] = obj[j][i];
        }
    }

    return result;
}


// From the MESA implementation of the GLU library
template <typename T>
__host__ __device__
Matrix4x4<T> inverse(const Matrix4x4<T> &matrix)
{
    T m[16];
    matrix.getArray(m);

    T invOut[16];
    T inv[16], det;


    inv[0] = m[5]  * m[10] * m[15] - 
             m[5]  * m[11] * m[14] - 
             m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] - 
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] + 
              m[4]  * m[11] * m[14] + 
              m[8]  * m[6]  * m[15] - 
              m[8]  * m[7]  * m[14] - 
              m[12] * m[6]  * m[11] + 
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] - 
             m[4]  * m[11] * m[13] - 
             m[8]  * m[5] * m[15] + 
             m[8]  * m[7] * m[13] + 
             m[12] * m[5] * m[11] - 
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] + 
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] - 
               m[8]  * m[6] * m[13] - 
               m[12] * m[5] * m[10] + 
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] + 
              m[1]  * m[11] * m[14] + 
              m[9]  * m[2] * m[15] - 
              m[9]  * m[3] * m[14] - 
              m[13] * m[2] * m[11] + 
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] - 
             m[0]  * m[11] * m[14] - 
             m[8]  * m[2] * m[15] + 
             m[8]  * m[3] * m[14] + 
             m[12] * m[2] * m[11] - 
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] + 
              m[0]  * m[11] * m[13] + 
              m[8]  * m[1] * m[15] - 
              m[8]  * m[3] * m[13] - 
              m[12] * m[1] * m[11] + 
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] - 
              m[0]  * m[10] * m[13] - 
              m[8]  * m[1] * m[14] + 
              m[8]  * m[2] * m[13] + 
              m[12] * m[1] * m[10] - 
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] - 
             m[1]  * m[7] * m[14] - 
             m[5]  * m[2] * m[15] + 
             m[5]  * m[3] * m[14] + 
             m[13] * m[2] * m[7] - 
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] + 
              m[0]  * m[7] * m[14] + 
              m[4]  * m[2] * m[15] - 
              m[4]  * m[3] * m[14] - 
              m[12] * m[2] * m[7] + 
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] - 
              m[0]  * m[7] * m[13] - 
              m[4]  * m[1] * m[15] + 
              m[4]  * m[3] * m[13] + 
              m[12] * m[1] * m[7] - 
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] + 
               m[0]  * m[6] * m[13] + 
               m[4]  * m[1] * m[14] - 
               m[4]  * m[2] * m[13] - 
               m[12] * m[1] * m[6] + 
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + 
              m[1] * m[7] * m[10] + 
              m[5] * m[2] * m[11] - 
              m[5] * m[3] * m[10] - 
              m[9] * m[2] * m[7] + 
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - 
             m[0] * m[7] * m[10] - 
             m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + 
             m[8] * m[2] * m[7] - 
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + 
               m[0] * m[7] * m[9] + 
               m[4] * m[1] * m[11] - 
               m[4] * m[3] * m[9] - 
               m[8] * m[1] * m[7] + 
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - 
              m[0] * m[6] * m[9] - 
              m[4] * m[1] * m[10] + 
              m[4] * m[2] * m[9] + 
              m[8] * m[1] * m[6] - 
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return Matrix4x4<T>();

    det = 1.0 / det;

    for (int i = 0; i < 16; i++) {
        invOut[i] = inv[i] * det;
    }

    return Matrix4x4<T>(invOut);
};

typedef Matrix4x4<Float> Matrix4x4f;
typedef Matrix4x4<int> Matrix4x4i;
