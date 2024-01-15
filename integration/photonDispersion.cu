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

#include "pathTracing.cu"

#include <time.h>
#include <thread>
#include <random>
#include <vector>
#include <atomic>


void samplePhotonIntersections(
        thrust::host_vector<SampleState> &samples,
        Sampler *sampler, Scene *scene, 
        int iMin, int iMax)
{
    thrust::host_vector<Photon> photons;

    for (int i = iMin; i < iMax && i < samples.size(); i++)
    {
        SampleState &current_sample = samples[i];

        // If the sample is not finished
        if (!current_sample.finished)
        {
            bool isScattering = current_sample.interaction->isScatteringMaterial();
            
            if (isScattering)
            {
                SurfaceInteraction *interaction = (SurfaceInteraction*)current_sample.interaction;

                //Float cos = absCosine(interaction->shading.normal.toVector(), interaction->worldToShading(-current_sample.pathRay.direction));

                photons.push_back(Photon(current_sample.pathRay,  
                        current_sample.interaction->worldPoint, 
                        interaction->shading.normal.toVector(),
                        current_sample.totalLight*current_sample.pathLightFactor,
                        interaction->getShapeId(),
                        interaction->surfacePoint));
            }

            // Intersect the ray with the scene
            Spectrum currentScatteredLight = 
                        current_sample.interaction->sampleRay(
                                sampler,
                                current_sample.pathRay);


            /****************** Compute bounce scattering *********************/
            Spectrum emittedLight = current_sample.interaction->getEmittedLight();
            current_sample.totalLight += current_sample.pathLightFactor * emittedLight;
            
            // Accumulates to the factor the scattered light of the surface, the cosine term and the pdf
            current_sample.pathLightFactor *= currentScatteredLight;

            // If the scattered light is black, there will be no further contributions
            if (currentScatteredLight.isBlack())
                current_sample.finished = true;
        }
    }

    scene->storePhotons(photons);
}


void photonComputingThread(
        thrust::host_vector<SampleState> &samples,
        thrust::host_vector<int> &triangleIntersections,
        thrust::host_vector<int> &interactionIds,
        int nSamples, 
        Scene *scene, 
        Sampler *sampler,
        std::atomic<bool> &allFinished,
        int iMin, int iMax)
{
    scene->intersectShapes(samples, 
            triangleIntersections, 
            interactionIds,
            iMin,
            iMax);

    samplePhotonIntersections(samples,
            sampler,
            scene,
            iMin,
            iMax);

    // Apply roussian roulette, delete interactions
    //  and store temporal data of the bounce (like isSpecular)
    finishStepsArray(samples,
            nSamples, 
            scene, 
            sampler,
            allFinished,
            iMin,
            iMax,
            true);
}

void computePhotons(Scene *scene,
                    thrust::host_vector<SampleState> &samples, 
                    Sampler *sampler)
{
    unsigned int nSamples = samples.size();
    std::atomic<bool> allFinished (false);

    std::vector<std::thread> workers(N_THREADS);
    int samplesPerThread = ceil((Float)nSamples / (Float)N_THREADS);
    
    for (int d = 0; d < MAX_PHOTON_RECURSION_DEPTH && !allFinished; d++)
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

            workers[i] = std::thread(photonComputingThread, 
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
    }
}


void generatePhotons(Scene *scene, Sampler *sampler)
{
    if (scene->nLights() > 0)
    {
        auto photons = scene->generatePhotons(sampler, N_PHOTON_SAMPLES);
        thrust::host_vector<SampleState> samples(N_PHOTON_SAMPLES);

        // Initialize samples
        for (int i = 0; i < N_PHOTON_SAMPLES; i++)
        {
            samples[i].pathRay = photons[i].ray;
            samples[i].totalLight = photons[i].radiance;
        }

        computePhotons(scene, samples, sampler);

        scene->buildPhotonStructures();
    }
}