
void mergeAllTileShapeSamples(std::vector<std::vector<ShapeLightningSamples>> &shapeSamples,
                                std::vector<ShapeLightningSamples> &mergedSamples,
                                int nTilesX, int nTilesY)
{
    std::thread shapesMergeThreads[mergedSamples.size()];

    // Merge each tiles shape samples
    for (int iShape = 0; iShape < mergedSamples.size(); iShape++)
    {
        shapesMergeThreads[iShape] = std::thread(mergeShapeSamples, 
                                            std::ref(shapeSamples),
                                            std::ref(mergedSamples),
                                            iShape);

        //mergeShapeSamples(shapeSamples, mergedSamples, iShape);
    }

    for (int i = 0; i < mergedSamples.size(); i++)
    {
        shapesMergeThreads[i].join();
    }
}

void trainDenoiser(ShapeLightningSamples &samples, 
                    ShapeLightDistribution &denoiser)
{
    std::vector<std::vector<Float>> &shapeSamplesX = samples.samplesX;
    std::vector<Spectrum> &shapeSamplesY = samples.samplesY;

    denoiser.train(shapeSamplesX, shapeSamplesY);
}

void computeDenoiserSamples(const Scene *cpuScene,
                    Camera* cpuCamera,
                    thrust::host_vector<SampleState> &samplesArrayCPU,
                    int nTilesX, int nTilesY)
{
    // Generate samples for the pixel
    Sampler cpuSampler = Sampler();
    int nPixeles = resolution.x * resolution.y;
    long long nTotalSamples = nPixeles * DENOISER_SAMPLES_PER_PIXEL;

    // Generate samples with the lightning models
    for (int iPixel = 0; iPixel < nPixeles; iPixel++)
    {
        Point2i pixelId(iPixel % resolution.x, iPixel / resolution.x);
        unsigned int pixelSamplesOffset = iPixel*DENOISER_SAMPLES_PER_PIXEL;

        generatePixelSamples(&cpuSampler, &cpuCamera, 
                samplesArrayCPU.data(), DENOISER_SAMPLES_PER_PIXEL,
                pixelSamplesOffset, 
                pixelId);
    }

    /********************** Compute denoiser samples ***********************/
    for (int iSample = 0; iSample < nTotalSamples; iSample++)
    {
        /************ Compute first interaction **************/
        generateInteraction(iSample, samplesArrayCPU, 
                nTotalSamples, cpuScene, &cpuSampler);

        if (!samplesArrayCPU[iSample].finished)
        {
            // Samples a new ray from the interaction point
            samplesArrayCPU[iSample].currentScatteredLight = 
                    samplesArrayCPU[iSample].interaction->sampleRay(
                            &cpuSampler,
                            samplesArrayCPU[iSample].pathRay);
        }

        std::atomic<bool> allFinished = false;

        computeBounce(iSample, samplesArrayCPU,
                nTotalSamples, cpuScene, &cpuSampler, allFinished);
    }
}

void denoiseFilter(const Scene *cpuScene, 
                    Camera* cpuCamera, 
                    std::vector<ShapeLightDistribution> &shapeLightDistributions)
{
    int nPixeles = resolution.x * resolution.y;
    long long nTotalSamples = nPixeles * DENOISER_SAMPLES_PER_PIXEL;

    int nTilesX = max(1, ceil((Float)resolution.x / (Float)TILE_SIDE_X));
    int nTilesY = max(1, ceil((Float)resolution.y / (Float)TILE_SIDE_Y));

    if (DENOISER_SAMPLES_PER_PIXEL > 0)
    {
        thrust::host_vector<SampleState> samplesArrayCPU (nTotalSamples);

        computeDenoiserSamples(cpuScene, cpuCamera, 
                                samplesArrayCPU, nTilesX, nTilesY);

        std::vector<std::vector<DenoiserSample>> denoiserSamplesPerShape(cpuScene->getNShapes());

        for (int iSample = 0; iSample < nTotalSamples; iSample++)
        {
            SampleState &currentSample = samplesArrayCPU[iSample];

            if (!currentSample.finished)
            {
                unsigned int shapeId = currentSample.firstShapeId;

                denoiserSamplesPerShape[shapeId].push_back(
                        DenoiserSample({currentSample.firstSampleSurfacePoint, 
                                        currentSample.indirectLight,
                                        iSample,
                                        currentSample.firstStepLightFactor})
                );
            }
        }

        /************ Estimate indirect light with the denoiser *************/              
        for (int iDenoiser = 0; iDenoiser < shapeLightDistributions.size(); iDenoiser++)
        {
            // Get light model
            ShapeLightDistribution &shapeLightDistribution = shapeLightDistributions[iDenoiser];

            std::vector<DenoiserSample> &currentShapeSamples = denoiserSamplesPerShape[iDenoiser];

            // Get the lightning model prediction
            shapeLightDistribution.predict(currentShapeSamples);

            for (int iSample = 0; iSample < denoiserSamplesPerShape[iDenoiser].size(); iSample++)
            {
                DenoiserSample &currentDenoiserSample = denoiserSamplesPerShape[iDenoiser][iSample];
                SampleState &currentSample = samplesArrayCPU[currentDenoiserSample.sampleId];

                // Add the sample to the pixel samples
                currentSample.indirectLight = 
                        currentSample.firstStepLightFactor * currentDenoiserSample.indirectLight;
            }
        }


        /***************** Add denoiser samples to film ******************/
        for (int iPixel = 0; iPixel < nPixeles; iPixel++)
        {
            Spectrum directPixelContribution = Spectrum(0.0);
            Spectrum indirectPixelContribution = Spectrum(0.0);

            long long pixelSamplesOffset = iPixel*DENOISER_SAMPLES_PER_PIXEL;

            for (int iSample = pixelSamplesOffset; 
                    iSample < pixelSamplesOffset+DENOISER_SAMPLES_PER_PIXEL;
                    iSample++)
            {
                SampleState &currentSample = samplesArrayCPU[iSample];
                directPixelContribution += currentSample.directLight;
                indirectPixelContribution += currentSample.indirectLight;
            }

            Point2f pixelId = Point2f(iPixel % resolution.x, iPixel / resolution.x);
            // Add pixel contribution to image
            cpuCamera->film->addSample(pixelId, 
                                        directPixelContribution, 
                                        indirectPixelContribution, DENOISER_SAMPLES_PER_PIXEL);
        }
    }
}


void trainDenoisers(std::vector<ShapeLightningSamples> &mergedShapeSamples,
                    std::vector<ShapeLightDistribution> &shapeLightDistributions)
{
    std::thread denoiserThreads[mergedShapeSamples.size()];

    // Train a denoiser for each shape
    for (int iShape = 0; iShape < mergedShapeSamples.size(); iShape++)
    {
        denoiserThreads[iShape] = std::thread(trainDenoiser, 
                                            std::ref(mergedShapeSamples[iShape]),
                                            std::ref(shapeLightDistributions[iShape]));

        //trainDenoiser(mergedShapeSamples[iShape], shapeLightDistributions[iShape]);
    }

    for (int iShape = 0; iShape < mergedShapeSamples.size(); iShape++)
    {
        denoiserThreads[iShape].join();
    }
}
