
#include "shapes/Sphere.cu"
#include "shapes/Plane.cu"
#include "shapes/Triangle.cu"

#include "integration/Scene.cu"
//#include "integration/PathTracerIntegrator.cu"
#include "integration/PhotonMapIntegrator.cu"

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



    // Create a sphere
    auto sphere1 = Sphere(Point3f(2.5, 1.25, 0), 
                        Vector3f(0, 0, 2), 
                        Point3f(2.5, 2.25, 0));

    auto sphere2 = Sphere(Point3f(2.5, 0, 0), 
                        Vector3f(0, 0, 2),
                        Point3f(1.5, 0, 0));

    auto sphereL = Sphere(Point3f(0, -3.5, 0), 
                        Vector3f(0, 0, 4),
                        Point3f(0, -5.5, 0));

    auto worldSphere = Sphere(Point3f(0, 0, 0), 20000);
    auto sky = Sphere(Point3f(0, 0, 0), 10000);


    auto plane1 = Plane(Point3f(3, -3, 0), 
                        Vector3f(0, 6, 0), 
                        Vector3f(0, 0, 4), 
                        Vector3f(-1, 0, 0));

    auto plane2 = Plane(Point3f(3, 3, 0), 
                        Vector3f(-6, 0, 0), 
                        Vector3f(0, 0, 4), 
                        Vector3f(0, -1, 0));

    auto plane3 = Plane(Point3f(-1000, -1000, 0), 
                        Vector3f(0, 2000, 0), 
                        Vector3f(0, 0, 2000), 
                        Vector3f(1, 0, 0));

    auto plane4 = Plane(Point3f(3, -3, 0), 
                        Vector3f(-6, 0, 0), 
                        Vector3f(0, 0, 4), 
                        Vector3f(0, 1, 0));

    auto plane5 = Plane(Point3f(-3, -3, 0), 
                        Vector3f(6, 0, 0), 
                        Vector3f(0, 6, 0), 
                        Vector3f(0, 0, 1));

    auto plane6 = Plane(Point3f(-10, -10, 6), 
                        Vector3f(20, 0, 0), 
                        Vector3f(0, 20, 0), 
                        Vector3f(0, 0, -1));

    LambertianMaterial blueMaterial(Spectrum(0.4f, 0.73f, 1.0f));
    LambertianMaterial redMaterial(Spectrum(1.0f, 0.2f, 0.3f));
    LambertianMaterial whiteMaterial(Spectrum(1.0f, 1.0f, 1.0f));
    LambertianMaterial greyMaterial(Spectrum(0.7f, 0.7f, 0.7f));
    LambertianMaterial skyMaterial(Spectrum(0.0f, 0.0f, 1.0f), Spectrum(0), nullptr, true);

    LightMaterial lightMaterial(Spectrum(1, 1, 1), 4);
    GlassMaterial glassMaterial(Spectrum(1, 1, 1));
    MirrorMaterial mirror(Spectrum(1, 1, 1));


    std::shared_ptr<std::vector<TriangleMesh>> triangleMeshes = std::make_shared<std::vector<TriangleMesh>>();
    std::vector<std::vector<Triangle>> triangles;

    loadObj("/home/ismael/pathtracing_renderer/models/lens/Focal_Lens_2_smooth.obj", triangleMeshes, triangles);
    //loadObj("/home/hsunekichi/Escritorio/pathtracing_renderer/models/kaerMorhen/kaerMorhen.obj", mesh1, mesh1Triangles);
    //loadObj("/export/d03/scratch/a816678/pathtracing_renderer/models/kaerMorhen/kaerMorhen.obj", mesh1, mesh1Triangles);


    //Texture *texture = new Texture("/home/hsunekichi/Escritorio/pathtracing_renderer/bricks.bmp");
    //Texture *sph_texture = new SphericalTexture("/home/hsunekichi/Escritorio/pathtracing_renderer/marsSPH.bmp");

    //Texture *texture = new Texture("/export/d03/scratch/a816678/pathtracing_renderer/bricks.bmp");
    //Texture *sph_texture = new SphericalTexture("/export/d03/scratch/a816678/pathtracing_renderer/marsSPH.bmp");


    //LambertianMaterial bricksMaterial(Spectrum(1), Spectrum(1), texture);
    //LambertianMaterial sph_bricksMaterial(Spectrum(1), Spectrum(1), sph_texture);

    //sphere1.setMaterial(&whiteMaterial);
    //sphere2.setMaterial(&glassMaterial);
    worldSphere.setMaterial(&lightMaterial);
    sky.setMaterial(&skyMaterial);


    //plane1.setMaterial(&bricksMaterial);
    //plane2.setMaterial(&bricksMaterial);
    //plane3.setMaterial(&lightMaterial);
    //plane4.setMaterial(&whiteMaterial);
    //plane5.setMaterial(&bricksMaterial);
    //plane6.setMaterial(&lightMaterial);

    //sphereL.setMaterial(&lightMaterial);

    // Create device array with the shape*
    std::vector<Shape*> shapes;
    //shapes.push_back(&sphere1);
    //shapes.push_back(&sphere2);
    //shapes.push_back(&plane2);
    //shapes.push_back(&worldSphere);
    //shapes.push_back(&sky);

    //shapes.push_back(&plane1);
    //shapes.push_back(&plane2);
    //shapes.push_back(&plane3);
    //shapes.push_back(&plane4);
    //shapes.push_back(&plane5);
    //shapes.push_back(&plane6);


    std::vector<Light*> lights;
    Camera *camera;

    loadObejota("/home/ismael/pathtracing_renderer/models/lens/Focal_Lens_2_smooth.obejota", lights, shapes, camera);
    //loadObejota("/home/hsunekichi/Escritorio/pathtracing_renderer/models/kaerMorhen/kaerMorhen.obejota", lights, shapes, camera);
    //loadObejota("/export/d03/scratch/a816678/pathtracing_renderer/models/kaerMorhen/kaerMorhen.obejota", lights, shapes, camera);

    Scene scene;
    scene.build(shapes, lights, triangles, triangleMeshes);
   
    //PathTracerIntegrator integrator(&camera, &sampler);
    PhotonMapIntegrator integrator(camera);

    integrator.render(&scene);

    //film.filterIndirectLight();
    //film.loadContributionsFromFile("test");
    //film.storeContributionsToFile("test");
    
    Film *film = camera->getFilm();
    if (RESCALE_IMAGE == true)
        film->rescale(3840, 2160);
        

    film->writeToBMP("focal_lens", ToneMappingType::reinhardJodie);
}