
#include "shapes/Sphere.cu"
#include "shapes/Plane.cu"
#include "shapes/Triangle.cu"

#include "integration/Scene.cu"
#include "integration/PathTracerIntegrator.cu"

#include "camera/ProjectiveCamera.cu"
#include "sampling/Sampler.cu"
#include "light/PointLight.cu"
#include "materials/LambertianMaterial.cu"
#include "materials/LightMaterial.cu"
#include "materials/GlassMaterial.cu"
#include "materials/MirrorMaterial.cu"

#include "textures/Texture.cu"
#include "textures/SphericalTexture.cu"

#include "cuda_libraries/FileImporting.cu"

int main(int argc, char *argv[])
{
    std::string fileName = "image";

    // Get file name
    if (argc >= 2)
    {
        fileName = argv[1];
    }

    // Get gpu id
    if (argc == 3)
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        int gpuId = atoi(argv[2]);

        if (gpuId >= deviceCount)
        {
            std::cout << "Invalid GPU id" << std::endl;
            return 0;
        }

        cudaSetDevice(gpuId);
    }


    std::shared_ptr<std::vector<TriangleMesh>> triangleMeshes = std::make_shared<std::vector<TriangleMesh>>();
    std::vector<std::vector<Triangle>> triangles;

    loadObj("./models/bath/bath.obj", triangleMeshes, triangles);
   
    // Create device array with the shape*
    std::vector<Shape*> shapes;
    std::vector<Light*> lights;
    Camera *camera;

    loadObejota("./models/bath/bath.obejota", lights, shapes, camera);

    Scene scene;
    scene.build(shapes, lights, triangles, triangleMeshes);
   
    //PathTracerIntegrator integrator(&camera, &sampler);
    PathTracerIntegrator integrator(camera);
    // Loads latest file

    if (N_RENDERS > 1)
    {
        int highestId = getNewHdrBmpId("./bathRenders") - 1;
        if (highestId >= 0) {
            std::cout << "Loading image " << highestId << std::endl;
            camera->film->loadContributionsFromFile("./bathRenders/bath_" + std::to_string(highestId));
        }

        for (int i = 0; i < N_RENDERS; i++)
        {
            integrator.render(&scene);
            Film *film = camera->getFilm();

            int newId = getNewHdrBmpId("./bathRenders");

            //film.filterIndirectLight();
            film->storeContributionsToFile("./bathRenders/bath_" + std::to_string(newId));
            std::cout << "Stored image " << newId << std::endl;

            film->writeToBMP("./bathRenders/bath_" + std::to_string(newId), ToneMappingType::reinhard);
        
            scene.clearPhotons();
        }
    }
    else
    {
        integrator.render(&scene);
    }
    
    camera->film->storeContributionsToFile("yenBath");
    camera->film->writeToBMP("yenBath", ToneMappingType::reinhard);
}