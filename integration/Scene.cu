#pragma once

#include <thrust/host_vector.h>
#include <vector>
#include <thread>
#include <mutex>

#include "../sampling/ShapeLightDistribution.cu"
#include "../transformations/SurfaceInteraction.cu"
#include "../transformations/Interaction.cu"
#include "../geometry/Point.cu"
#include "../geometry/Vector.cu"
#include "../geometry/Ray.cu"

#include "../cuda_libraries/BVH.cu"
#include "../cuda_libraries/kdtree.h"

#include "../light/Light.cu"
#include "../sampling/Sampler.cu"

class Scene
{

protected:
    std::vector<Shape*> shapes;
    std::vector<Light*> lights;
    std::vector<Triangle> triangles;
    std::vector<CpuTriangleMesh> cpuMeshes;
    std::vector<TriangleMeshData> meshData;

    thrust::host_vector<Photon> photons; 
    std::mutex photonMutex;
    BVH *bvh = nullptr;
    PhotonMap *photonMap = nullptr;

    std::vector<ShapeLightDistribution> shapeLightDistributions;

public:

    Scene() {};

    ~Scene() 
    {
        delete bvh;
        delete photonMap;
    }

    void copyMesh(std::vector<TriangleMesh> &meshes)
    {
        for (int i = 0; i < meshes.size(); i++)
        {
            cpuMeshes.push_back(CpuTriangleMesh(meshes[i]));
        }
    }

    void build(const std::vector<Shape*> &shapes,
            const std::vector<Light*> &lights,
            std::vector<std::vector<Triangle>> &initTriangles,
            std::shared_ptr<std::vector<TriangleMesh>> &triangleMesh)
    {
        this->shapes = shapes;
        this->lights = lights;

        if (initTriangles.size() > 0)
        {
            copyMesh(*triangleMesh);
        
            for (int i = 0; i < initTriangles.size(); i++)
            {
                // Init mesh data for this triangle mesh
                TriangleMeshData data;
                data.vertices = cpuMeshes[i].vertices.data();

                if (cpuMeshes[i].normals.size() > 0)
                    data.normals = cpuMeshes[i].normals.data();
                
                if (cpuMeshes[i].textureCoords.size() > 0) 
                {
                    data.textureCoords = cpuMeshes[i].textureCoords.data();
                }

                meshData.push_back(data);

                // Add triangles to the scene
                for (int j = 0; j < initTriangles[i].size(); j++)
                {
                    initTriangles[i][j].mesh = &meshData[i];
                    this->triangles.push_back(initTriangles[i][j]);
                }
            }

            bvh = new BVH(triangleMesh, initTriangles, meshData, triangles);
            //bvh = new BVH(triangleMesh, initTriangles);
        }
    }


    bool intersects (const Ray &ray, Float &hitOffset) const
    {
        thrust::host_vector<Ray> rays(1, ray);
        thrust::host_vector<Float> hitOffsets(1, INFINITY);
        bool intersected = intersects(rays, hitOffsets)[0];
        hitOffset = hitOffsets[0];
        return intersected;
    }


    thrust::host_vector<bool> intersects(
            const thrust::host_vector<Ray> &rays,
            thrust::host_vector<Float> &hitOffsets,
            bool checkShadow=false,
            cudaStream_t streamId = 0) const
    {
        thrust::host_vector<int> triangleIds;
        thrust::host_vector<bool> v_intersected(rays.size(), false);

        if (bvh != nullptr)
        {
            bvh->intersectRays(rays, triangleIds, checkShadow, streamId);
        }

        for (int i = 0; i < rays.size(); i++)
        {
            /******************* Store triangle intersection data ******************/
            const Ray &ray = rays[i];
            bool &intersected = v_intersected[i];
            Float &tempOffset = hitOffsets[i];
            tempOffset = INFINITY;

            // Intersect with the triangle returned by the BVH
            if (bvh != nullptr)
            {
                int tId = triangleIds[i];

                if (tId != -1)
                {
                    intersected = triangles[tId].intersects(ray, tempOffset, checkShadow);
                }
            }

            /*************** Intersect ray with normal shapes ****************/
            for (int j = 0; j < shapes.size(); j++)
            {                
                bool hasIntersected = shapes[j]->intersects(ray, tempOffset, checkShadow);
                intersected = intersected || hasIntersected;
            }
        }

        return v_intersected;
    }


    void intersect (SampleState &sample) const
    {
        thrust::host_vector<SampleState> samples(1, sample);
        thrust::host_vector<int> interactionIds(samples.size());

        auto triangleIntersections = intersectTriangles(samples, interactionIds);
        intersectShapes(samples, triangleIntersections, 
            interactionIds, 0, samples.size());

        sample = samples[0];
    }

    thrust::host_vector<int> 
        intersectTriangles(thrust::host_vector<SampleState> &samples,
                thrust::host_vector<int> &interactionIds) const
    {
        thrust::host_vector<int> triangleInteractions;

        if (bvh != nullptr)
        {
            thrust::host_vector<Ray> samplesCopy;

            for (int i = 0; i < samples.size(); i++)
            {
                if (!samples[i].finished) 
                {
                    samplesCopy.push_back(samples[i].pathRay);
                    interactionIds[i] = samplesCopy.size() - 1;
                }
            }

            bvh->intersectRays(samplesCopy, triangleInteractions);
        }

        return triangleInteractions;
    }

    void intersectShapes(thrust::host_vector<SampleState> &samples,
            thrust::host_vector<int> &triangleIds,
            thrust::host_vector<int> &interactionIds,
            int iMin, int iMax) const
    {
        for (int i = iMin; i < iMax; i++)
        {
            if (!samples[i].finished)
            {
                /******************* Store triangle intersection data ******************/
                Ray &ray = samples[i].pathRay;
                bool &intersected = samples[i].intersected;
                int interactionIndex = interactionIds[i];
                Float tempOffset = INFINITY;
                intersected = false;

                SurfaceInteraction interaction;

                // Intersect with the triangle returned by the BVH
                if (bvh != nullptr && interactionIndex != -1)
                {
                    int tId = triangleIds[interactionIndex];

                    if (tId != -1)
                    {
                        intersected = triangles[tId].intersect(ray, tempOffset, 
                            interaction, shapes.size()+tId, true);
                    }
                }

                /*************** Intersect ray with normal shapes ****************/
                for (int j = 0; j < shapes.size(); j++)
                {                
                    bool hasIntersected = shapes[j]->intersect(ray, tempOffset,
                                                    interaction, j, true);

                    intersected = intersected || hasIntersected;
                }

                samples[i].lastBunceWasSpecular = false;
                samples[i].finished = !intersected;

                if (!samples[i].finished)
                    samples[i].interaction = new SurfaceInteraction(interaction);
            }
        }
    }

    thrust::host_vector<bool> isVisible(
            const thrust::host_vector<SampleState> &samples, 
            const thrust::host_vector<Point3f> &lightPoint,
            thrust::host_vector<Spectrum> &scatteringWeights,
            cudaStream_t streamId = 0) const
    {
        thrust::host_vector<bool> visible(samples.size(), false);
        thrust::host_vector<Float> hitOffsets(samples.size());
        thrust::host_vector<Ray> shadowRays;

        for (int i = 0; i < samples.size(); i++)
        {
            if (!samples[i].finished && !scatteringWeights[i].isBlack())
            {
                Vector3f pointToLight = lightPoint[i] - samples[i].interaction->worldPoint;
                pointToLight = normalize(pointToLight);

                Point3f worldPoint = samples[i].interaction->worldPoint;
                shadowRays.push_back(samples[i].interaction->spawnRay(worldPoint, pointToLight));
            }
        }
        
        thrust::host_vector<bool> haveIntersected = intersects(shadowRays, hitOffsets, true, streamId);

        int iIntersections = 0;
        for (int i = 0; i < samples.size(); i++)
        {
            if (!samples[i].finished && !scatteringWeights[i].isBlack())
            {
                Vector3f pointToLight = lightPoint[i] - samples[i].interaction->worldPoint;
                Float distance = pointToLight.length();
                
                visible[i] = !haveIntersected[iIntersections] || hitOffsets[iIntersections] > distance;
                iIntersections++;
            }
        }

        return visible;
    }
    
    // Computes the light scattered over the interaction point,
    //  using the light distribution
    Spectrum computeLightScattering (Interaction *interaction, 
            Vector3f surfaceToLight, 
            Float &scatteringPdf) const
    {
        Spectrum scatteringWeight;

        if (interaction->isSurfaceInteraction()) 
        {
            // Compute BSDF 
            const SurfaceInteraction *surfaceInteraction = (const SurfaceInteraction*)interaction;

            // If the light is not in the same hemisphere as the surface, there is no visibility
            if (!surfaceInteraction->isOrientedTo(surfaceToLight)) {
                scatteringWeight = Spectrum(0.f);
            }
            else 
            {
                scatteringWeight = surfaceInteraction->lightScattered(surfaceToLight);
                scatteringPdf = surfaceInteraction->pdf(surfaceToLight);
            }
        } 
        else 
        {
            /*
            // Compute light interaction with the medium
            const MediumInteraction *mediumInteraction = (const MediumInteraction*)interaction;
            Float phase = mediumInteraction->phase->phase(mediumInteraction.cameraRay.direction, incidentDirection);
            scatteringWeight = Spectrum(phase);
            scatteringPdf = phase;
            */
        }

        return scatteringWeight;
    }

    void addLightContribution(
            const Interaction *interaction,
            Spectrum scatteringWeight, 
            Float &scatteringPdf,
            Float &lightPdf,
            Spectrum &estimateLight,
            Spectrum &lightSample,
            Point3f lightPoint) const
    {
        // If the light scattered is not black, computes its visibility
        if (!scatteringWeight.isBlack() && !lightSample.isBlack()) 
        {
            // Add light contribution
            //if (light->isDelta())
            //    Ld += f * Li;
            //else {
            Float lightWeight = powerHeuristic(1, lightPdf, 1, scatteringPdf);
            estimateLight += scatteringWeight * lightSample * lightWeight;
            //}
        }
    }

    void checkVisibility(
            thrust::host_vector<SampleState> &samples,
            thrust::host_vector<Point3f> &lightPoints,
            thrust::host_vector<Spectrum> &lightSamples,
            thrust::host_vector<Spectrum> &scatteringWeights,
            cudaStream_t streamId = 0) const
    {
        auto areVisible = this->isVisible(samples, lightPoints, scatteringWeights, streamId);

        for (int i = 0; i < samples.size(); i++)
        {
            // If the light scattered is not black, computes its visibility
            if (!scatteringWeights[i].isBlack()) 
            {
                // Check visibility
                //if (checkMediaAttenuation)
                    //lightSample *= visibility.Tr(scene, sampler); // Also computes visibility with all entities
                // else
                if (!areVisible[i]) {     // Only checks visibility with surfaces
                    lightSamples[i] = Spectrum(0.f);
                }
            }
        }
    }
    
    // Computes estimated light over the interaction point,
    //  using multiple importance sampling 
    thrust::host_vector<Spectrum> estimateOneDirectLight(
            thrust::host_vector<SampleState> &samples, 
            Sampler *sampler, 
            bool checkMediaAttenuation) const
    {
        thrust::host_vector<Spectrum> recievedLight(samples.size());
        thrust::host_vector<Vector3f> surfaceToLight(samples.size());
        thrust::host_vector<Float> lightPdf(samples.size(), 0);
        thrust::host_vector<Float> interactionScatteringPdf(samples.size(), 0);
        thrust::host_vector<Point3f> lightPoint(samples.size());
        thrust::host_vector<Spectrum> lightSamples(samples.size());
        thrust::host_vector<Spectrum> scatteringWeights(samples.size(), Spectrum(0));


        for (int i = 0; i < samples.size(); i++)
        {
            if(!samples[i].finished)
            {           
                const int nLights = lights.size();
                const int lightId = selectOneRandomLight(sampler);
                
                // Sample with multiple importance
                lightSamples[i] = lights[lightId]->sampleLight(samples[i].interaction, 
                                                sampler->get2Dsample(), 
                                                surfaceToLight[i], 
                                                lightPdf[i], 
                                                lightPoint[i]);

                lightSamples[i] /= lightSelectionPdf(lightId);

                if (samples[i].interaction->isSurfaceInteraction()) {
                    ((SurfaceInteraction*)samples[i].interaction)->L = surfaceToLight[i];
                }

                if (lightPdf[i] > 0 && !lightSamples[i].isBlack()) 
                {   
                    // Compute light scattering
                    scatteringWeights[i] = computeLightScattering(samples[i].interaction, 
                                                        surfaceToLight[i], 
                                                        interactionScatteringPdf[i]);
                }
            }
        }

        // Check visibility
        checkVisibility(samples, lightPoint, lightSamples, scatteringWeights);

        for(int i = 0; i < samples.size(); i++)
        {
            if(!samples[i].finished)
            {
                // Add light contribution
                addLightContribution(samples[i].interaction, scatteringWeights[i], 
                                    interactionScatteringPdf[i], lightPdf[i], 
                                    recievedLight[i], lightSamples[i], lightPoint[i]);
                
                // Get a sample from the BSDF distribution
            }  
        }

        return recievedLight;
    }

    int selectOneRandomLight(Sampler *sampler) const
    {
        const int nLights = lights.size();
        Float sceneIntensity = getSceneLightIntensity();       
        Float rand1 = sampler->get1Dsample();

        //return int(rand1 * nLights);

        // Select a light
        for (int i = 0; i < nLights; i++) 
        {
            rand1 -= sum(lights[i]->power()) / sceneIntensity;
            if (rand1 <= 0) {
                return i;
            }
        }

        return -1;
    }

    Float lightSelectionPdf(int lightId) const
    {
        //return 1.0 / lights.size();

        Float sceneIntensity = getSceneLightIntensity();
        return sum(lights[lightId]->power()) / sceneIntensity;
    }

    Float getSceneLightIntensity() const
    {
        Float sceneIntensity = 0;

        for (int i = 0; i < lights.size(); i++) {
            sceneIntensity += sum(lights[i]->power());
        }

        return sceneIntensity;
    }

    // Computes estimated light over the interaction point,
    //  using multiple importance sampling 
    void estimateAllDirectLights(
            thrust::host_vector<SampleState> &samples, 
            int lightId,
            Sampler *sampler, 
            bool checkMediaAttenuation,
            thrust::host_vector<Spectrum> &recievedLight) const
    {
        thrust::host_vector<Vector3f> surfaceToLight(samples.size());
        thrust::host_vector<Float> lightPdf(samples.size(), 0);
        thrust::host_vector<Float> interactionScatteringPdf(samples.size(), 0);
        thrust::host_vector<Point3f> lightPoint(samples.size());
        thrust::host_vector<Spectrum> lightSamples(samples.size());
        thrust::host_vector<Spectrum> scatteringWeights(samples.size(), Spectrum(0));

        for (int i = 0; i < samples.size(); i++)
        {
            if(!samples[i].finished)
            {           
                // Sample with multiple importance
                lightSamples[i] = lights[lightId]->sampleLight(samples[i].interaction, 
                                                sampler->get2Dsample(), 
                                                surfaceToLight[i], 
                                                lightPdf[i], 
                                                lightPoint[i]);

                if (samples[i].interaction->isSurfaceInteraction()) {
                    ((SurfaceInteraction*)samples[i].interaction)->L = surfaceToLight[i];
                }

                if (lightPdf[i] > 0 && !lightSamples[i].isBlack()) 
                {   
                    // Compute light scattering
                    scatteringWeights[i] = computeLightScattering(samples[i].interaction, 
                                                        surfaceToLight[i], 
                                                        interactionScatteringPdf[i]);
                }
            }
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Check visibility
        checkVisibility(samples, lightPoint, lightSamples, scatteringWeights, stream);

        for(int i = 0; i < samples.size(); i++)
        {
            if(!samples[i].finished)
            {
                // Add light contribution
                addLightContribution(samples[i].interaction, scatteringWeights[i], 
                                    interactionScatteringPdf[i], lightPdf[i], 
                                    recievedLight[i], lightSamples[i], lightPoint[i]);
                
                // Get a sample from the BSDF distribution
            }  
        }
    }

    thrust::host_vector<Spectrum> sampleOneDirectLight(
            thrust::host_vector<SampleState> &samples, 
            Sampler *sampler, 
            bool checkMediaAttenuation=false) const
    {
        int nLights = lights.size();

        // If there are no lights, return black
        if (nLights == 0) 
            return thrust::host_vector<Spectrum>(0);

        // The expected light contribution is statistically the sample light 
        //  weigthed with its probability of beying chosen. Since we have a uniform
        //  distribution, the probability is 1 / nLights and hence 
        //  ExpectedLight = sampleLight / (1/nLights) = sampleLight * nLights
        auto recievedLight = estimateOneDirectLight(samples,
                            sampler, checkMediaAttenuation);

        return recievedLight;
    } 

    thrust::host_vector<Spectrum> sampleAllDirectLight(
            thrust::host_vector<SampleState> &samples, 
            Sampler *sampler, 
            bool checkMediaAttenuation=false) const
    {
        thrust::host_vector<Spectrum> recievedLight(samples.size());
        thrust::host_vector<thrust::host_vector<Spectrum>> lightSamples(lights.size(), 
            thrust::host_vector<Spectrum>(samples.size(), Spectrum(0)));

        std::vector<std::thread> threads(lights.size());

        for (int i = 0; i < lights.size(); i++)
        {
            //estimateAllDirectLights(samples, i, 
            //                sampler, checkMediaAttenuation,
            //                lightSamples[i]);

            threads[i] = std::thread(&Scene::estimateAllDirectLights, this, 
                                std::ref(samples), i, 
                                sampler, checkMediaAttenuation,
                                std::ref(lightSamples[i]));
        }

        for (int i = 0; i < lights.size(); i++)
        {
            threads[i].join();
        }

        for (int i = 0; i < lights.size(); i++)
        {
            for (int j = 0; j < samples.size(); j++)
            {
                recievedLight[j] += lightSamples[i][j];
            }
        }

        return recievedLight;
    }

    Spectrum getAmbientLight(Interaction *interaction,
            Sampler *sampler) const
    {
        Spectrum ambient = Spectrum(0.0);

        for (int i = 0; i < lights.size(); i++)
        {
            Vector3f pointToLight;
            Float lightPdf;
            Point3f point;
            Spectrum contribution = lights[i]->sampleLight(interaction, sampler->get2Dsample(), 
                pointToLight, lightPdf, point);

            if (lightPdf > 0 && !contribution.isBlack())
                ambient += contribution;
        }

        return 0.5 * ambient / lights.size();
    }

    Spectrum combinePhotonContributions(SurfaceInteraction &si, 
            thrust::host_vector<Photon> &photons) const
    {
        // Subsurface scattering is accounted earlier
        if (si.isSubsurfaceScattering())
            return Spectrum(0);

        // Density estimation
        Spectrum contribution = Spectrum(0.0);
        Float maxDistance = 0;

        if (photons.size() == 0)
            return Spectrum(0);

        for (int i = 0; i < photons.size(); i++)
        {
            Photon &photon = photons[i];
            Float d = distance(photon.location, si.worldPoint);
            maxDistance = max(maxDistance, d);

            Spectrum scattering = si.lightScattered(-photon.ray.direction);
            contribution += photon.radiance * scattering;
            //contribution += Spectrum(0.1);
        }

        Float kernel = 1 / (PI * maxDistance * maxDistance);

        return contribution * kernel;
    }

    class diffusionAproxFunctor
    {
        private:
            Point3f p;
            Vector3f cameraDir;
            Vector3f cameraNormal;

            Spectrum sigmaS;
            Spectrum sigmaA;
            Float g;
            Float eta;

        public:
            diffusionAproxFunctor(Point3f p, 
                    Vector3f cameraDir, Vector3f cameraNormal,
                    Spectrum sigmaS, Spectrum sigmaA, 
                    Float g, Float eta) 
            : p(p), sigmaS(sigmaS), sigmaA(sigmaA), g(g), eta(eta), 
                cameraDir(cameraDir),
                cameraNormal(cameraNormal)
            {}

            __host__ __device__ 
            Spectrum operator()(Photon ph)
            {
                Spectrum factor = diffusionTermBSSRDF(p, cameraDir, cameraNormal, ph, sigmaS, sigmaA, g, eta);
                return ph.radiance * factor;
            }
    };
    
    Spectrum combinePhotonsSubsurface(Point3f p, 
            SurfaceInteraction &interaction,
            thrust::host_vector<Photon> &photons) const
    {
        Spectrum sigmaS;
        Spectrum sigmaA;
        Spectrum kd;
        Float g;
        Float eta;

        Vector3f cameraDir = -interaction.worldCameraRay.direction;
        Vector3f cameraNormal = interaction.shading.normal.toVector(); 
        interaction.getMaterialBSSDFparameters(kd, sigmaS, sigmaA, g, eta);

        thrust::host_vector<Spectrum> contributions(photons.size());
        thrust::transform(photons.begin(), photons.end(), contributions.begin(), 
            diffusionAproxFunctor(p, cameraDir, cameraNormal, sigmaS, sigmaA, g, eta));

        Spectrum contribution = Spectrum(0.0);

        Float maxD = 0;

        for (int i = 0; i < photons.size(); i++)
        {
            contribution += contributions[i] * kd;

            Float d = distance(photons[i].location, p);
            maxD = max(maxD, d);
        }

        contribution /= (2*PI*maxD*maxD);
        //std::cout << contribution << std::endl;

        return contribution;
    }


    thrust::host_vector<Photon> search_nearest(Point3f position, 
            Float radius, 
            unsigned long nPhotons_estimate,
            int shapeId=-1) const
    {
        if (photonMap == nullptr)
            throw std::runtime_error("Photon map not built");

        // nearest is the nearest photons returned by the KDTree
        std::vector<Photon> nearest = photonMap->nearest_neighbors(position,
                                            nPhotons_estimate,
                                            radius,
                                            shapeId);


        thrust::host_vector<Photon> thrust_nearest(nearest);

        return thrust_nearest;
    }

    Spectrum getPhotonMappingContribution(
            Interaction *interaction, 
            Sampler *sampler) const
    {
        if (!interaction->isSurfaceInteraction())
            throw std::runtime_error("Photon mapping is only supported for surface interactions");
    
        SurfaceInteraction &surfaceInteraction = *(SurfaceInteraction*)interaction;
        Point3f point = surfaceInteraction.worldPoint;
        Spectrum contribution;

        if (surfaceInteraction.isScatteringMaterial())
        {
            // KDTree
            auto contributingPhotons = search_nearest(point, 
                    PHOTON_SEARCH_RADIUS, 
                    MAX_NEIGHBOURS, 
                    surfaceInteraction.getShapeId());

            contribution += combinePhotonContributions(surfaceInteraction, contributingPhotons);
        }

        return contribution;
    }
    
    class FilterPhotonSurfaceId
    {
        private:
            int shapeId;
            GPUVector<Photon> *result;

        public:
            FilterPhotonSurfaceId(int shapeId, GPUVector<Photon> &result) 
            : shapeId(shapeId), result(&result) 
            {}

            __host__ __device__ 
            void operator()(Photon ph)
            {
                if (ph.shapeId == shapeId)
                    result->push_back(ph);
            }
    };

    void th_photonMappingContributions(thrust::host_vector<SampleState> &samples, 
        thrust::host_vector<Spectrum> &output,
        int iMin, int iMax) const
    {
        Sampler sampler;

        for (int i = iMin; i < iMax; i++)
        {
            if (!samples[i].finished)
            {
                Spectrum contribution = getPhotonMappingContribution(samples[i].interaction, &sampler);
                output[i] = contribution;
            }
        }
    }

    void th_photonSubsurfaceContributions(thrust::host_vector<SampleState> &samples, 
        thrust::host_vector<Spectrum> &output,
        int iMin, int iMax) const
    {
        Sampler sampler;
        //thrust::host_vector<Photon> filtered_photons;

        for (int i = iMin; i < iMax; i++)
        {
            if (!samples[i].finished)
            {
                Interaction *interaction = samples[i].interaction;

                if (!interaction->isSurfaceInteraction())
                    throw std::runtime_error("Photon mapping is only supported for surface interactions");
            
                SurfaceInteraction &surfaceInteraction = *(SurfaceInteraction*)interaction;
                Point3f point = surfaceInteraction.worldPoint;
                
                Float searchRadius = 0.3;

                // Only camera rays account for sss
                if (surfaceInteraction.isSubsurfaceScattering() && surfaceInteraction.worldCameraRay.isCameraRay()) // Only directly visible bounces
                {
                    int shapeId = surfaceInteraction.getShapeId();

                    auto contributingPhotons = search_nearest(point, searchRadius, (ulong)-1, shapeId);

                    Spectrum contribution = combinePhotonsSubsurface(point, 
                            surfaceInteraction,
                            contributingPhotons);

                    output[i] = contribution;
                }
            }
        }
    }

    thrust::host_vector<Spectrum> getPhotonContributions(
            thrust::host_vector<SampleState> &samples, 
            Sampler *sampler) const
    {
        thrust::host_vector<Spectrum> recievedLight(samples.size());
        std::vector<std::thread> threads(N_THREADS);

        for (int i = 0; i < N_THREADS; i++)
        {
            threads[i] = std::thread(&Scene::th_photonMappingContributions, this, 
                                std::ref(samples), std::ref(recievedLight),
                                i * samples.size() / N_THREADS, 
                                (i+1) * samples.size() / N_THREADS);
        }

        for (int i = 0; i < N_THREADS; i++)
        {
            threads[i].join();
        }


        return recievedLight;
    }

    thrust::host_vector<Spectrum> getSubsurfaceContributions(
            thrust::host_vector<SampleState> &samples, 
            Sampler *sampler) const
    {
        thrust::host_vector<Spectrum> recievedLight(samples.size());
        std::vector<std::thread> threads(N_THREADS);

        for (int i = 0; i < N_THREADS; i++)
        {
            threads[i] = std::thread(&Scene::th_photonSubsurfaceContributions, this, 
                                std::ref(samples), std::ref(recievedLight),
                                i * samples.size() / N_THREADS, 
                                (i+1) * samples.size() / N_THREADS);
        }

        for (int i = 0; i < N_THREADS; i++)
        {
            threads[i].join();
        }
    
        return recievedLight;
    }

    thrust::host_vector<Spectrum> predictPhotons(
            thrust::host_vector<SampleState> &samples) const
    {
        thrust::host_vector<Spectrum> recievedLight(samples.size());
        thrust::host_vector<thrust::host_vector<Point2f>> shapePoints(shapes.size());
        thrust::host_vector<thrust::host_vector<Spectrum>> shapePredictions(shapes.size());
        thrust::host_vector<thrust::host_vector<int>> shapeSampleIds(shapes.size());

        for (int i = 0; i < samples.size(); i++)
        {
            if (!samples[i].finished && samples[i].interaction->isSurfaceInteraction())
            {
                SurfaceInteraction *interaction = (SurfaceInteraction*)samples[i].interaction;
                Point2f surfacePoint = interaction->getSurfacePoint();
                shapePoints[interaction->getShapeId()].push_back(shapes[interaction->getShapeId()]->surfacePointTo01(surfacePoint));
                shapeSampleIds[interaction->getShapeId()].push_back(i);
            }
        }

        for (int i = 0; i < shapes.size(); i++)
        {
            if (shapePoints[i].size() > 0)
            {
                shapePredictions[i] = shapeLightDistributions[i].predict(shapePoints[i]);
            
                for (int j = 0; j < shapePredictions[i].size(); j++)
                {
                    recievedLight[shapeSampleIds[i][j]] += shapePredictions[i][j];
                }
            }
        }

        for (int i = 0; i < samples.size(); i++)
        {
            if (samples[i].interaction != nullptr)
            {
                SurfaceInteraction *interaction = (SurfaceInteraction*)samples[i].interaction;
                recievedLight[i] *= interaction->lightScattered();
            }
        }
        

        return recievedLight;
    }

    thrust::host_vector<Spectrum> sampleDirectLight(
            thrust::host_vector<SampleState> &samples, 
            Sampler *sampler, 
            bool checkMediaAttenuation=false) const
    {
        thrust::host_vector<Spectrum> recievedLight(samples.size());

        if (CHECK_SUBSURFACE_SCATTERING)
        {
            auto subsurface = getSubsurfaceContributions(samples, sampler);

            for (int i = 0; i < samples.size(); i++)
            {
                recievedLight[i] += subsurface[i];
            }
        }

        if (RENDER_MODE == PHOTON_MAPPING || RENDER_MODE == HYBRID)
        {
            auto photons = getPhotonContributions(samples, sampler);

            for (int i = 0; i < samples.size(); i++)
            {
                Spectrum &contribution = photons[i];
                recievedLight[i] += contribution;
            }
        }

        if (RENDER_MODE == MACHINE_LEARNING)
        {
            auto photons = predictPhotons(samples);

            for (int i = 0; i < samples.size(); i++)
            {
                Spectrum &contribution = photons[i];
                recievedLight[i] += contribution;
            }
        }

        if (RENDER_MODE == PATH_TRACING || RENDER_MODE == HYBRID)
        {
            auto direct = sampleAllDirectLight(samples, sampler, checkMediaAttenuation);

            for (int i = 0; i < samples.size(); i++)
            {
                recievedLight[i] += direct[i];
            }
        }

        if (RENDER_MODE == HYBRID)
        {
            for (int i = 0; i < samples.size(); i++)
            {
                recievedLight[i] /= 2;
            }
        }

        return recievedLight;
    }

    Spectrum getEnvironmentLight(const Ray &ray) const
    {
        return Spectrum(0.0);
    }

    unsigned int getNShapes() const
    {
        return shapes.size();
    }

    unsigned int nLights() const
    {
        return lights.size();
    }

    thrust::host_vector<Photon> generatePhotons (
            Sampler *sampler, 
            unsigned int nSamples) const
    {
        thrust::host_vector<Photon> photons;
        thrust::host_vector<int> samplesPerLight(lights.size(), 0);
        unsigned int nLights = lights.size();

        for (int i = 0; i < nSamples; i++)
        {
            int randomLight = selectOneRandomLight(sampler);
            samplesPerLight[randomLight]++;
        }

        for (int l = 0; l < nLights; l++)
        {
            Ray outgoingRay;
            unsigned int lightSamples = samplesPerLight[l];
            auto lightPhotons = lights[l]->sampleLight(sampler, lightSamples);
            
            // Accounts for light sampling pdf
            for (int sample = 0; sample < lightSamples; sample++) {
                lightPhotons[sample].radiance /= lightSelectionPdf(l);
            }
            
            photons.insert(photons.end(), lightPhotons.begin(), lightPhotons.end());
        }
        
        return photons;
    }

    void storePhotons(thrust::host_vector<Photon> &photons)
    {
        // Append photons
        photonMutex.lock();
        this->photons.insert(this->photons.end(), photons.begin(), photons.end());
        photonMutex.unlock();
    }

    void buildShapeDistribution(int shapeId)
    {
        ShapeLightDistribution distribution;
            
        // Get photons from the shape
        auto photons = search_nearest(shapes[shapeId]->worldBound().centroid(), 
                uint(-1), 
                uint(-1), 
                shapeId);


        // C++ random number generator
        Sampler sampler;

        // Generate k random point2f
        thrust::host_vector<Photon> photonSamples(N_ML_SAMPLES);

        for (int i = 0; i < N_ML_SAMPLES; i++)
        {
            photonSamples[i].surfacePoint = sampler.get2Dsample();
        }

        Float maxD = 0;

        for (int i = 0; i < photons.size(); i++)
        {
            for (int j = 0; j < N_ML_SAMPLES; j++)
            {
                // Apply density estimation
                Float d = distance(shapes[shapeId]->surfacePointTo01(photons[i].surfacePoint), photonSamples[j].surfacePoint);

                if (d < ML_DENSITY_RADIUS) {
                    photonSamples[j].radiance += photons[i].radiance;
                    //std::cout << "Added photon" << std::endl;
                }

                maxD = max(maxD, d);
            }
        }

        // Apply kernel
        for (int i = 0; i < N_ML_SAMPLES; i++) {
            photonSamples[i].radiance /= (PI * maxD * maxD);
        }

        distribution.train(photonSamples);

        shapeLightDistributions.push_back(distribution);
    }

    void buildShapeDistributions()
    {
        std::thread threads[shapes.size()];

        //buildShapeDistribution(0);
        //return;

        for (int i = 0; i < shapes.size(); i++)
        {
            threads[i] = std::thread(&Scene::buildShapeDistribution, this, i);
        }

        for (int i = 0; i < shapes.size(); i++)
        {
            threads[i].join();
        }
    }

    void clearPhotons()
    {
        photons.clear();
        delete photonMap;
    }

    void buildPhotonStructures()
    {
        // Build KDTree
        std::vector<Photon> std_photons(photons.begin(), photons.end());

        if (std_photons.size() > 0)
        {
            photonMap = new PhotonMap(std_photons);

            if (RENDER_MODE == MACHINE_LEARNING)
                buildShapeDistributions();
        }
    }
};