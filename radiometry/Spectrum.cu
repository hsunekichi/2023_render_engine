#pragma once

#include "../cuda_libraries/types.h"
#include "../cuda_libraries/math.cu"


struct RGB
{
    unsigned char r, g, b;
};

class Spectrum
{
    protected:
    double r, g, b;

    public:

    __host__ __device__
    Spectrum() : r(0), g(0), b(0) {}
    
    __host__ __device__
    Spectrum(double r, double g, double b) : r(r), g(g), b(b) {}
    
    __host__ __device__
    Spectrum(double v) : r(v), g(v), b(v) {}

    
    __host__ __device__
    double getR() const { return r; }
    
    __host__ __device__
    double getG() const { return g; }
    
    __host__ __device__
    double getB() const { return b; }

    __host__ __device__
    void setR(double r) { this->r = r; }

    __host__ __device__
    void setG(double g) { this->g = g; }

    __host__ __device__
    void setB(double b) { this->b = b; }

    __host__ __device__
    bool isBlack() const
    {
        return r == 0 && g == 0 && b == 0;
    }

    __host__ __device__
    RGB toRGB() const
    {
        // Escalate to 255
        double r = this->r * 255;
        double g = this->g * 255;
        double b = this->b * 255;

        // Truncate pixels to 255 and convert to unsigned char
        return RGB{ (unsigned char)(clamp(r, 0.0, 255.0)), 
                    (unsigned char)(clamp(g, 0.0, 255.0)), 
                    (unsigned char)(clamp(b, 0.0, 255.0)) };
    }

    __host__ __device__
    double brightness() const
    {
        const double redWeight   =  0.212671f;
        const double greenWeight =  0.715160f;
        const double blueWeight  =  0.072169f;
        
        return redWeight * r + greenWeight * g + blueWeight * b;
    }

    __host__ __device__
    Spectrum operator+(const Spectrum& s) const
    {
        return Spectrum(r + s.r, g + s.g, b + s.b);
    }

    __host__ __device__
    Spectrum operator-(const Spectrum& s) const
    {
        return Spectrum(r - s.r, g - s.g, b - s.b);
    }

    __host__ __device__
    Spectrum operator*(const Spectrum& s) const
    {
        return Spectrum(r * s.r, g * s.g, b * s.b);
    }

    __host__ __device__
    Spectrum operator*(double s) const
    {
        return Spectrum(r * s, g * s, b * s);
    }

    __host__ __device__
    Spectrum operator/(const Spectrum& s) const
    {
        return Spectrum(r / s.r, g / s.g, b / s.b);
    }

    __host__ __device__
    Spectrum operator/(double s) const
    {
        return Spectrum(r / s, g / s, b / s);
    }

    __host__ __device__
    Spectrum& operator+=(const Spectrum& s)
    {
        r += s.r;
        g += s.g;
        b += s.b;
        return *this;
    }

    __host__ __device__
    Spectrum& operator-=(const Spectrum& s)
    {
        r -= s.r;
        g -= s.g;
        b -= s.b;
        return *this;
    }

    __host__ __device__
    Spectrum& operator*=(const Spectrum& s)
    {
        r *= s.r;
        g *= s.g;
        b *= s.b;
        return *this;
    }

    __host__ __device__
    Spectrum& operator*=(double s)
    {
        r *= s;
        g *= s;
        b *= s;
        return *this;
    }

    __host__ __device__
    Spectrum& operator/=(const Spectrum& s)
    {
        r /= s.r;
        g /= s.g;
        b /= s.b;
        return *this;
    }

    __host__ __device__
    Spectrum& operator/=(double s)
    {
        r /= s;
        g /= s;
        b /= s;
        return *this;
    }

    __host__ __device__
    bool operator==(const Spectrum& s) const
    {
        return r == s.r && g == s.g && b == s.b;
    }

    __host__ __device__
    bool operator!=(const Spectrum& s) const
    {
        return r != s.r || g != s.g || b != s.b;
    }

    // Operator <<
    friend std::ostream& operator<<(std::ostream& os, const Spectrum& s)
    {
        os << "(" << s.r << ", " << s.g << ", " << s.b << ")";
        return os;
    }

    inline double maxComponent() const
    {
        return max(r, max(g, b));
    }

    __host__ __device__
    bool hasNaNs() const
    {
        return isnan(r) || isnan(g) || isnan(b);
    }

    __host__ __device__
    bool hasInf() const
    {
        return isinf(r) || isinf(g) || isinf(b);
    }

    __host__ __device__
    Float squaredNorm() const
    {
        return r * r + g * g + b * b;
    }

    __host__ __device__
    Float norm() const
    {
        return sqrt(squaredNorm());
    }
};

// Inverse int - operator
__host__ __device__
inline Spectrum operator-(int s, const Spectrum& v)
{
    return Spectrum(s) - v;
}


//Inverse float * operator
__host__ __device__
inline Spectrum operator*(Float s, const Spectrum& v)
{
    return v * s;
}

//Inverse float + operator
__host__ __device__
inline Spectrum operator+(Float s, const Spectrum& v)
{
    return v + s;
}

__host__ __device__
inline Spectrum pow2(Spectrum s)
{
    return Spectrum(s.getR() * s.getR(), s.getG() * s.getG(), s.getB() * s.getB());
}

__host__ __device__
inline Spectrum pow3(Spectrum s)
{
    return Spectrum(s.getR() * s.getR() * s.getR(), s.getG() * s.getG() * s.getG(), s.getB() * s.getB() * s.getB());
}

// Inverse / operator (ej: 1 / Spectrum)
__host__ __device__
inline Spectrum operator/(double s, const Spectrum& v)
{
    return Spectrum(s) / v;
}

// Sqrt
__host__ __device__
inline Spectrum sqrt(const Spectrum& s)
{
    return Spectrum(sqrt(s.getR()), sqrt(s.getG()), sqrt(s.getB()));
}

// exp
__host__ __device__
inline Spectrum exp(const Spectrum& s)
{
    return Spectrum(exp(s.getR()), exp(s.getG()), exp(s.getB()));
}

// - operand
__host__ __device__
inline Spectrum operator-(const Spectrum& s)
{
    return Spectrum(-s.getR(), -s.getG(), -s.getB());
}

// pow
__host__ __device__
inline Spectrum pow(const Spectrum& s, double e)
{
    return Spectrum(pow(s.getR(), e), pow(s.getG(), e), pow(s.getB(), e));
}

__host__ __device__
inline Float sum(const Spectrum& s)
{
    return s.getR() + s.getG() + s.getB();
}

__host__ __device__
inline Spectrum abs(const Spectrum& s)
{
    return Spectrum(abs(s.getR()), abs(s.getG()), abs(s.getB()));
}