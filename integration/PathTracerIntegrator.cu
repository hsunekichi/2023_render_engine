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
#include "photonDispersion.cu"
#include "pathTracing.cu"

#include <time.h>
#include <thread>
#include <random>
#include <vector>
#include <atomic>


void renderTile(const Scene *cpuScene, 
        int iTileX, int iTileY, 
        Sampler &cpuSampler, Camera *cpuCamera,
        std::vector<ShapeLightningSamples> &shapeSamples)
{
    int sizeX = min(TILE_SIDE_X, resolution.x - iTileX * TILE_SIDE_X);
    int sizeY = min(TILE_SIDE_Y, resolution.y - iTileY * TILE_SIDE_Y);

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks((sizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (sizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    int nSamples = sizeX * sizeY * SAMPLES_PER_PIXEL;


    thrust::host_vector<SampleState> samplesArrayCPU (nSamples);

    // Generate sample rays from the camera
    for (int i = 0; i < sizeX*sizeY; i++)
    {
        int thread_id_x = i % sizeX;
        int thread_id_y = i / sizeX;

        Point2i pixelId = Point2i(thread_id_x + iTileX * TILE_SIDE_X, thread_id_y + iTileY * TILE_SIDE_Y);
        unsigned int pixelSamplesOffset = (thread_id_y * sizeX + thread_id_x)*SAMPLES_PER_PIXEL;

        generatePixelSamples(&cpuSampler, &cpuCamera, 
                    samplesArrayCPU.data(), SAMPLES_PER_PIXEL,
                    pixelSamplesOffset, pixelId);
    }

    // Compute samples
    computeSamples(cpuScene, samplesArrayCPU, nSamples, &cpuSampler, shapeSamples);

    // Compute pixels from samples
    for (int i = 0; i < sizeX*sizeY; i++)
    {
        int thread_id_x = i % sizeX;
        int thread_id_y = i / sizeX;

        groupPixelContributions(samplesArrayCPU.data(), nSamples, &cpuCamera, 
                                                            iTileX, iTileY, sizeX, sizeY, thread_id_x, thread_id_y);
    }   

}

void mergeShapeSamples(std::vector<std::vector<ShapeLightningSamples>> &shapeSamples,
                        std::vector<ShapeLightningSamples> &mergedSamples,
                        unsigned int shapeId)
{
    // For each tile
    for (int i = 0; i < shapeSamples.size(); i++)
    {
        std::vector<std::vector<Float>> &tileShapeSamplesX = shapeSamples[i][shapeId].samplesX;
        std::vector<std::vector<Float>> &mergedShapeSamplesX = mergedSamples[shapeId].samplesX;

        std::vector<Spectrum> &tileShapeSamplesY = shapeSamples[i][shapeId].samplesY;
        std::vector<Spectrum> &mergedShapeSamplesY = mergedSamples[shapeId].samplesY;

        // Add x samples
        mergedShapeSamplesX.insert(mergedShapeSamplesX.end(), tileShapeSamplesX.begin(), tileShapeSamplesX.end());

        // Add y samples
        mergedShapeSamplesY.insert(mergedShapeSamplesY.end(), tileShapeSamplesY.begin(), tileShapeSamplesY.end());
    }
}


void renderAllTiles(const Scene *cpuScene,
                    Camera* cpuCamera,
                    std::vector<std::vector<ShapeLightningSamples>> &shapeSamples,
                    int nTilesX, int nTilesY)
{
    // Render tiles
    for (int iTileX = 0; iTileX < nTilesX; iTileX++)
    {
        for (int iTileY = 0; iTileY < nTilesY; iTileY++)
        { 
            // Generates a new sampler
            Sampler reprocessSampler = Sampler();

            // Get time
            auto start = std::chrono::high_resolution_clock::now();

            renderTile(cpuScene, 
                iTileX, iTileY, 
                reprocessSampler, cpuCamera,
                shapeSamples[iTileX * nTilesY + iTileY]);

            // Get time
            auto finish = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = finish - start;

            if (elapsed.count() < 60)
            {
                std::cout << "Finished " << iTileX * nTilesY + iTileY + 1 << "/" 
                        << nTilesX * nTilesY << " tiles in "
                        << elapsed.count() << "s, "
                        << "estimated time left: "
                        << elapsed.count() * (nTilesX * nTilesY - (iTileX * nTilesY + iTileY)) / 60 << "m" << std::endl;
            }
            else
            {
                std::cout << "Finished " << iTileX * nTilesY + iTileY + 1 << "/" 
                        << nTilesX * nTilesY << " tiles in "
                        << elapsed.count() / 60 << "m, "
                        << "estimated time left: "
                        << elapsed.count() * (nTilesX * nTilesY - (iTileX * nTilesY + iTileY)) / 60 << "m" << std::endl;
            }
        }
    }   
}

class PathTracerIntegrator
{
private: 
    Camera *cpuCamera;

public:
    PathTracerIntegrator(Camera *camera) 
    { 
        this->cpuCamera = camera;
    }

    ~PathTracerIntegrator() {}

    void render(Scene *cpuScene)
    {
        printf("Rendering scene\n");

        if (RENDER_MODE == PHOTON_MAPPING || RENDER_MODE == HYBRID || RENDER_MODE == MACHINE_LEARNING || CHECK_SUBSURFACE_SCATTERING)
        {
            Sampler *sampler = new Sampler();
            generatePhotons(cpuScene, sampler);
        }
        
        std::vector<ShapeLightDistribution> shapeLightDistributions(cpuScene->getNShapes());

        int nTilesX = max(1, ceil((Float)resolution.x / (Float)TILE_SIDE_X));
        int nTilesY = max(1, ceil((Float)resolution.y / (Float)TILE_SIDE_Y));

        // Generate samples for denoising
        std::vector<std::vector<ShapeLightningSamples>> shapeSamples;
        //std::vector<ShapeLightningSamples> mergedShapeSamples;

        for (int i_reprocess = 0; i_reprocess < N_IMAGE_REPROCESSES; i_reprocess++)
        {
            //shapeSamples.resize(nTilesX * nTilesY);
            //mergedShapeSamples.resize(cpuScene->getNShapes());

            // Get high res time
            auto start = std::chrono::high_resolution_clock::now();

            renderAllTiles(cpuScene, cpuCamera, 
                            shapeSamples,
                            nTilesX, nTilesY);

            //mergeAllTileShapeSamples(shapeSamples, mergedShapeSamples, nTilesX, nTilesY);
            //shapeSamples.clear();

            //std::cout << "Training denoisers\n";
            
            if (DENOISER_SAMPLES_PER_PIXEL > 0)
            {
                //trainDenoisers(mergedShapeSamples, shapeLightDistributions);
            }

            //mergedShapeSamples.clear(); 

            // Get time
            auto finish = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = finish - start;
            double elapsed_mins = floor(elapsed.count() / 60);
            double elapsed_secs = elapsed.count() - elapsed_mins * 60;

            std::cout << "Image reprocess " << i_reprocess << " finished in " 
                    << elapsed_mins << "m, " << elapsed_secs << "s" << std::endl;
        }

        //denoiseFilter(cpuScene, 
        //        cpuCamera,
        //        shapeLightDistributions);
    }

};

