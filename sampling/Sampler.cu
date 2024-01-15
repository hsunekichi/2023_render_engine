#pragma once

#include "../geometry/Point.cu"
#include "../camera/Camera.cu"
#include "../cuda_libraries/math.cu"
#include "../cuda_libraries/geometricMath.cu"

#include <random>

namespace Sampling
{
    __host__ __device__
    Point2f concentricSampleDisk (const Point2f &p) 
    {
        Point2f uOffset = 2.0 * p - Vector2f(1, 1);

        // Handle the origin 
        if (uOffset.x == 0 && uOffset.y == 0)
            return Point2f(0, 0);

        // Apply concentric mapping to point 
        Float theta, r;

        if (abs(uOffset.x) > abs(uOffset.y)) 
        {
            r = uOffset.x;
            theta = PIquarter * (uOffset.y / uOffset.x);
        } 
        else 
        {
            r = uOffset.y;
            theta = PIhalf - PIquarter * (uOffset.x / uOffset.y);
        }

        return r * Point2f(cos(theta), sin(theta));
    }

    __host__ __device__
    Point2f uniformSampleDisk (const Point2f &p) 
    {
        Float r = sqrt(abs(p.x));
        Float theta = 2 * PI * p.y;
        return Point2f(r * cos(theta), r * sin(theta));
    }

    __host__ __device__
    Vector3f uniformSampleSphere(const Point2f &p)
    {
        Float phi = 2 * PI * p.x;
        Float theta = acos(2 * p.y - 1);
        return Vector3f(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    }

    Vector3f uniformSampleHemisphere(const Point2f &p)
    {
        Float phi = 2 * PI * p.x;
        Float theta = acos(p.y);
        return Vector3f(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    }

    __host__ __device__
    Float uniformHemispherePdf()
    {
        return 1/(2*PI);
    }

    __host__ __device__
    Float uniformSpherePdf()
    {
        return INV_FOUR_PI;
    }

    __host__ __device__
    Vector3f cosineSampleHemisphere(Point2f &p)
    {
        /*
        Float theta = acos(sqrt(p.x));
        Float phi = 2*PI*p.y;

        Float x = sin(theta) * cos(phi);
        Float y = sin(theta) * sin(phi);
        Float z = cos(theta);

        return Vector3f(x, y, z);
        */

        Point2f d = concentricSampleDisk(p);
        Float z = sqrt(max(0, 1 - d.x * d.x - d.y * d.y));
        return Vector3f(d.x, d.y, z);
    }

    __host__ __device__
    Float cosineHemispherePdf(Vector3f incomingRay)
    {
        return cosine(incomingRay) * INV_PI;
    }

    // Function to compute the sample glossy direction
    __host__ __device__
    Vector3f sampleGlossyDirection(const Vector3f& cameraDirection, Float ns, Point2f p) 
    {
        Point2f polarOffset;
        Vector3f perfectDir = Vector3f(-cameraDirection.x, -cameraDirection.y, cameraDirection.z);

        polarOffset.x = acos(pow(p.x, 1/(ns+1)));
        polarOffset.y = TAU * p.y;
        Vector3f sampleDir = rotateVector(perfectDir, polarOffset.x, polarOffset.y);

        return sampleDir;
    }

    __host__ __device__
    Float sampleGlossyPdf(Vector3f cameraDir, Vector3f sampleDir, Float ns)
    {
        Vector3f perfectSpecularDir = Vector3f(-cameraDir.x, -cameraDir.y, cameraDir.z);
        Float cosineFactor = pow(max(0, cosine(sampleDir, perfectSpecularDir)), ns);

        return ((ns+2)/TAU) * cosineFactor;
    }


};

class Sampler
{   
    public:
    int seed;
    curandGenerator_t cudaGenerator;

    std::mt19937 rng;
    std::uniform_real_distribution<Float> randomDistribution;

    #define N_RANDOM_SAMPLES 10000

    Float *randomNumbers = nullptr;
    int randomNumbersIndex = 0;

    __host__
    Sampler() 
    {
        // Initialize random distribution with random seed
        auto sd = std::random_device{}();
        rng = std::mt19937(sd);
        randomDistribution = std::uniform_real_distribution<Float>(0.0, 1.0);
    }

    // Copy constructor
    __host__ __device__
    Sampler (const Sampler &sampler) 
    {
        this->seed = sampler.seed;
        this->rng = sampler.rng;
        this->randomDistribution = sampler.randomDistribution;
    }

    // Destructor
    __host__ __device__
    ~Sampler () 
    {
    }




    __host__ __device__
    CameraSample getCameraSample(Point2i pRaster) 
    {
        CameraSample cameraSample;

        cameraSample.pFilm = (Point2f)pRaster + get2Dsample();

        cameraSample.pFilm = clamp(cameraSample.pFilm, 
                                    Point2f(0, 0), 
                                    Point2f(resolution.x, resolution.y));

        //cs.time = Get1D();

        cameraSample.pLens = get2Dsample();

        return cameraSample;
    }

    __global__
    friend
    void createGPUSampler(Sampler** gpuSampler, Float* gpuRandomNumbers);


    Sampler* toGPU() const
    {
        // Copy random numbers
        Float* gpuRandomNumbers;

        cudaMalloc(&gpuRandomNumbers, N_RANDOM_SAMPLES * sizeof(Float));
        cudaMemcpy(gpuRandomNumbers, this->randomNumbers, N_RANDOM_SAMPLES * sizeof(Float), cudaMemcpyHostToDevice);

        Sampler** gpuSampler;        
        cudaMalloc(&gpuSampler, sizeof(Sampler*));
        createGPUSampler<<<1, 1>>>(gpuSampler, gpuRandomNumbers);

        Sampler* gpuSamplerPtr;
        cudaMemcpy(&gpuSamplerPtr, gpuSampler, sizeof(Sampler*), cudaMemcpyDeviceToHost);

        // Free random numbers memory
        cudaFree(gpuRandomNumbers);

        return gpuSamplerPtr;
    }

    __host__ __device__
    inline Float randFloat()
    {
        #ifndef __CUDA_ARCH__
            return randomDistribution(rng);
        #else
            int threadId = threadIdx.x + blockIdx.x * blockDim.x;
            curandState state;
            curand_init(clock64(), threadId, 0, &state); // Initialize the random state
            return curand_uniform(&state);   // Generate a random float between 0 and 1
        #endif
    }

    __host__ __device__
    Point3f get3Dsample()
    {
        return Point3f(randFloat(), randFloat(), randFloat());
    }

    __host__ __device__
    Point2f get2Dsample()
    {
        return Point2f(randFloat(), randFloat());
    }

    __host__ __device__
    Float get1Dsample()
    {
        return randFloat();
    }
};

/*
__global__
void createGPUSampler(Sampler** gpuSampler, Float* gpuRandomNumbers)
{
    *gpuSampler = new Sampler();
    
    (*gpuSampler)->randomNumbers = new Float[N_RANDOM_SAMPLES];

    for (int i = 0; i < N_RANDOM_SAMPLES; i++)
    {
        (*gpuSampler)->randomNumbers[i] = gpuRandomNumbers[i];
    }

    (*gpuSampler)->randomNumbersIndex = 0;
}
*/


__global__
void printSamplerGPU(Sampler* sampler)
{
    for (int i = 0; i < 100; i++)
    {
        printf("%f\n", sampler->randFloat());
    }
}