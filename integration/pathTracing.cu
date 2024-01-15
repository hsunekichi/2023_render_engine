#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../sampling/Sampler.cu"
#include "../camera/Camera.cu"
#include "Scene.cu"

#include "../transformations/SurfaceInteraction.cu"

#include "../geometry/Point.cu"
#include "../cuda_libraries/types.h"
#include "../cuda_libraries/geometricMath.cu"
#include "../sampling/ShapeLightDistribution.cu"

#include <time.h>
#include <thread>
#include <random>
#include <vector>
#include <atomic>



struct ShapeLightningSamples
{
    std::vector<std::vector<Float>> samplesX;
    std::vector<Spectrum> samplesY;
};



void sampleIntersections(thrust::host_vector<SampleState> &samples,
        Sampler *sampler, int iMin, int iMax)
{
    for (int i = iMin; i < iMax && i < samples.size(); i++)
    {
        SampleState &current_sample = samples[i];

        // If the sample is not finished
        if (!current_sample.finished)
        {
            // Intersect the ray with the scene
            Spectrum currentScatteredLight = 
                        samples[i].interaction->sampleRay(
                                sampler,
                                samples[i].pathRay);

            Spectrum emittedLight = samples[i].interaction->getEmittedLight();
            samples[i].totalLight += samples[i].pathLightFactor * emittedLight;
            
            // Accumulates to the factor the scattered light of the surface, the cosine term and the pdf
            samples[i].pathLightFactor *= currentScatteredLight;

            // If the scattered light is black, there will be no further contributions
            if (currentScatteredLight.isBlack())
                samples[i].finished = true;
        }
    }
}

void finishStep(
        int threadId,
        thrust::host_vector<SampleState> &samples,
        int nSamples, 
        const Scene *scene, 
        Sampler *sampler,
        std::atomic<bool> &allFinished,
        bool photon_first_pass) 
{      
    if (threadId < nSamples && !samples[threadId].finished)
    {
        SampleState &current_sample = samples[threadId];

        Spectrum &pathLightFactor = current_sample.pathLightFactor;
        Interaction *&interaction = current_sample.interaction;

        bool &lastBunceWasSpecular = current_sample.lastBunceWasSpecular;
        int &depth = current_sample.depth;

        if (interaction->isScatteringMaterial()) 
        {
            /*********** Stop the path with Russian Roulette ***********/

            if (!photon_first_pass 
                && (RENDER_MODE == PHOTON_MAPPING || RENDER_MODE == MACHINE_LEARNING)
                && (!interaction->isSpecularMaterial() || !interaction->isSubsurfaceScattering())) {
                current_sample.finished = true;
            }

            if (depth > 3) 
            {
                // The more brightness, the less probability of stopping the path
                Float q = max((Float)0.05, 1 - pathLightFactor.brightness());

                // If the path is not stopped, 
                //  the path light factor is updated with the probability of continuing
                if (sampler->get1Dsample() < q)
                    current_sample.finished = true;
                else
                    pathLightFactor /= 1 - q;
            }
        }

        if (interaction->isSpecularMaterial())
            lastBunceWasSpecular = true;


        // The path is not finished
        if (!current_sample.finished)
        {
            depth++;
            allFinished = false;
        }
    }

    if (threadId < nSamples && samples[threadId].interaction != nullptr)
    {
        // Delete interaction data
        delete samples[threadId].interaction;
        samples[threadId].interaction = nullptr;
    }
}

void finishStepsArray(
        thrust::host_vector<SampleState> &samples,
        int nSamples, 
        const Scene *scene, 
        Sampler *sampler,
        std::atomic<bool> &allFinished,
        int iMin, int iMax,
        bool photon_first_pass) 
{    
    for (int i = iMin; i < iMax && i < samples.size(); i++)
    {
        finishStep(i, samples, nSamples, scene, sampler, allFinished, photon_first_pass);
    }
}


void generateInteraction(
        int threadId,
        thrust::host_vector<SampleState> &samples,
        int nSamples, 
        const Scene *scene, 
        Sampler *sampler)
{
    if (!samples[threadId].finished)
    {
        SampleState &current_sample = samples[threadId];

        //Spectrum &pathLightFactor = current_sample.pathLightFactor;
        Interaction *&interaction = current_sample.interaction;
        int &depth = current_sample.depth;
        bool &intersected = current_sample.intersected;

        if (depth == 0 && intersected)
        {
            // If it is an instance of surfaceInteraction
            if (interaction->isSurfaceInteraction())
            {
                current_sample.firstShapeId = 
                        ((SurfaceInteraction*)interaction)->getShapeId();

                current_sample.firstSampleSurfacePoint = 
                        ((SurfaceInteraction*)interaction)->getSurfacePoint();
            }
        }
    }
}


void computeDirectLightning(
        thrust::host_vector<SampleState> &samples,
        const thrust::host_vector<Spectrum> &incomingLights,
        int nSamples, 
        const Scene *scene, 
        Sampler *sampler,
        int iMin,
        int iMax)
{
    if (incomingLights.size() == 0)
        return;

    for (int i = iMin; i < iMax && i < samples.size(); i++)
    {
        if (!samples[i].finished && samples[i].interaction->isScatteringMaterial())
        {
            /*********** Direct light sampling ***********/
            const Spectrum incomingLight = incomingLights[i];
            const Spectrum &pathLightFactor = samples[i].pathLightFactor;
            Spectrum &totalLight = samples[i].totalLight;

            Spectrum ambient;

            if (AMBIENT_LIGHT)
            {
                ambient = samples[i].interaction->getAmbientScattering() 
                    * scene->getAmbientLight(samples[i].interaction, sampler);
            }

            totalLight += pathLightFactor * (incomingLight + ambient);
        }
    }
}

void th_computeShapesIntersection(
        thrust::host_vector<SampleState> &samples,
        thrust::host_vector<int> &triangleIntersections,
        thrust::host_vector<int> &interactionIds,
        int nSamples, 
        const Scene *scene, 
        Sampler *sampler,
        std::atomic<bool> &allFinished,
        int iMin, int iMax)
{
    scene->intersectShapes(samples, 
            triangleIntersections, 
            interactionIds,
            iMin,
            iMax);
}

void th_computeBounce(
        thrust::host_vector<SampleState> &samples,
        thrust::host_vector<int> &triangleIntersections,
        thrust::host_vector<Spectrum> &incomingLights,
        thrust::host_vector<int> &interactionIds,
        int nSamples, 
        const Scene *scene, 
        Sampler *sampler,
        std::atomic<bool> &allFinished,
        int iMin, int iMax)
{
    computeDirectLightning(
            samples,
            incomingLights,
            nSamples, 
            scene, 
            sampler,
            iMin,
            iMax);

    sampleIntersections(samples,
            sampler,
            iMin,
            iMax);

    finishStepsArray(samples,
            nSamples, 
            scene, 
            sampler,
            allFinished,
            iMin,
            iMax,
            false);
}

/*
__global__
void g_generateInteraction(SampleState *samples, int nSamples, const Scene *scene, Sampler *sampler)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    generateInteraction(threadId, samples, nSamples, scene, sampler);
}
*/

void printGPUMemoryUsage()
{
    // Print gpu memory
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    printf("GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n",
        used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}



void computeSamples(const Scene *scene,
                    thrust::host_vector<SampleState> &samples, 
                    int nSamples, Sampler *sampler,
                    std::vector<ShapeLightningSamples> &shapeSamples)
{
    std::atomic<bool> allFinished (false);

    std::vector<std::thread> workers(N_THREADS);
    int samplesPerThread = ceil((Float)nSamples / (Float)N_THREADS);
    
    int RECURSION_DEPTH = MAX_RAY_RECURSION_DEPTH;

    for (int d = 0; d < RECURSION_DEPTH && !allFinished; d++)
    {   
        //if (d % 5 == 0)
        //    std::cout << "Computing depth " << d << std::endl;
        
        allFinished = true;
        thrust::host_vector<int> interactionIds(nSamples, -1);

        /**************************** Intersect the samples with the scene ******************************/
        auto triangleIntersections = scene->intersectTriangles(samples, interactionIds);
        
        int runningSamples = 0;
        for (int i = 0; i < N_THREADS; i++) 
        {
            int iMin = i * samplesPerThread;
            int iMax = min((i + 1) * samplesPerThread, nSamples);

            workers[i] = std::thread(th_computeShapesIntersection, 
                                    std::ref(samples),
                                    std::ref(triangleIntersections),
                                    std::ref(interactionIds),
                                    nSamples, 
                                    scene, 
                                    sampler,
                                    std::ref(allFinished),
                                    iMin,
                                    iMax);
            
            runningSamples += iMax - iMin;

            if (runningSamples >= SAMPLES_PER_PIXEL*MAX_CONCURRENT_PIXELS)
            {
                for (int j = 0; j < N_THREADS; j++) 
                {
                    if (workers[j].joinable())
                        workers[j].join();
                }

                runningSamples = 0;
            }
        }

        for (int i = 0; i < N_THREADS; i++) 
        {
            if (workers[i].joinable())
                workers[i].join();
        }

        /******************************** Compute direct lightning and bounce ********************************/
        auto incomingLights = scene->sampleDirectLight(samples, sampler);

        for (int i = 0; i < N_THREADS; i++) 
        {
            int iMin = i * samplesPerThread;
            int iMax = min((i + 1) * samplesPerThread, nSamples);

            workers[i] = std::thread(
                th_computeBounce, 
                    std::ref(samples),
                    std::ref(triangleIntersections),
                    std::ref(incomingLights),
                    std::ref(interactionIds),
                    nSamples, 
                    scene, 
                    sampler,
                    std::ref(allFinished),
                    iMin,
                    iMax);
            
            runningSamples += iMax - iMin;

            if (runningSamples >= SAMPLES_PER_PIXEL*MAX_CONCURRENT_PIXELS)
            {
                for (int j = 0; j < N_THREADS; j++) 
                {
                    if (workers[j].joinable())
                        workers[j].join();
                }

                runningSamples = 0;
            }
        }

        for (int i = 0; i < N_THREADS; i++) 
        {
            if (workers[i].joinable())
                workers[i].join();
        }
    }

/*
    // Add samples for denoising training
    for (int i = 0; i < nSamples; i++)
    {
        SampleState &current_sample = samples[i];

        // If the sample intersected
        if (current_sample.depth > 0)
        {
            unsigned int firstShapeId = current_sample.firstShapeId;
            Point2f firstSurfacePoint = current_sample.firstSampleSurfacePoint;

            // Add X and Y training data
            shapeSamples[firstShapeId].samplesX.push_back(
                    std::vector<Float>({firstSurfacePoint.x, 
                                        firstSurfacePoint.y}));

            shapeSamples[firstShapeId].samplesY.push_back(
                    current_sample.indirectLight);
        }
    }
    
    for (int i = 0; i < nSamples; i++)
    {
        // Add the first shape albedo interaction
        samples[i].indirectLight *= samples[i].firstStepLightFactor;
    }
    */
}


void generatePixelSamples(Sampler *sampler, Camera **pp_camera, 
                    SampleState *samples, u_int64_t nSamples,
                    unsigned int pixelSamplesOffset,
                    Point2i pixelId)
{
    Camera *camera = *pp_camera;

    // Get pixel samples
    for (int i = 0; i < nSamples; i++)
    {
        SampleState &currentSample = samples[pixelSamplesOffset + i];
        currentSample = SampleState();

        // Get sample point on the pixel
        CameraSample cameraSample = sampler->getCameraSample(Point2i(pixelId.x, pixelId.y));   

        // Generate camera ray for pixel
        Ray sampleCameraRay;
        camera->generateRay(cameraSample, sampleCameraRay);

        // Initialize sample path state
        currentSample.filmOrigin = pixelId;
        currentSample.pathRay = sampleCameraRay;
    }
}

/*
__global__
void k_fillCameraSample(Sampler sampler, Camera **pp_camera, 
                    SampleState *samples, u_int64_t nSamples, 
                    int tileIdX, int tileIdY, int sizeX, int sizeY)
{
    // Get pixel id
    int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;

    fillCameraSample(&sampler, pp_camera, samples, nSamples, tileIdX, tileIdY, sizeX, sizeY, thread_id_x, thread_id_y);
}
*/

inline void groupPixelContributions(SampleState *samples, u_int64_t nSamples, Camera **pp_camera,
                            int tileIdX, int tileIdY, int sizeX, int sizeY,
                            int thread_id_x, int thread_id_y)
{
    Camera *camera = *pp_camera;

    Point2f pixelId = Point2f(thread_id_x + tileIdX * TILE_SIDE_X, thread_id_y + tileIdY * TILE_SIDE_Y);

    if (thread_id_x < sizeX && thread_id_y < sizeY)
    {
        unsigned int pixelSamplesOffset = (thread_id_y * sizeX + thread_id_x)*SAMPLES_PER_PIXEL;

        // Get pixel contribution
        Spectrum pixelContribution = Spectrum(0.0);
        int invalidSamples = 0;

        const long long nSamples = SAMPLES_PER_PIXEL;
        for (int i = 0; i < nSamples; i++)
        {
            SampleState &currentSample = samples[pixelSamplesOffset+i];

            // Add a sentinel in case a sample has found a NaN
            if (!currentSample.totalLight.hasNaNs() && !currentSample.totalLight.hasInf())
                pixelContribution += currentSample.totalLight;
            else
                invalidSamples++;
        }

        Spectrum avgValidContribution = pixelContribution / (nSamples - invalidSamples);
        pixelContribution += invalidSamples * avgValidContribution;

        // Add pixel contribution to image
        camera->film->addSample(pixelId, pixelContribution, nSamples);
    }
}


/*
__global__
void copyImageInGPU(Camera **pp_camera, Spectrum *image)
{
    // Get pixel id
    int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (thread_id_x < resolution.x && thread_id_y < resolution.y)
    {
        Camera *camera = *pp_camera;
        Spectrum *gpuPixels = camera->film->getPixels();
        image[thread_id_y * resolution.x + thread_id_x] = gpuPixels[thread_id_y * resolution.x + thread_id_x];
    }
}
*/