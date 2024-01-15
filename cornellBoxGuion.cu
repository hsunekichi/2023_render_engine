#include "shapes/Sphere.cu"
#include "shapes/Plane.cu"
#include "shapes/Triangle.cu"

#include "integration/Scene.cu"
#include "integration/PathTracerIntegrator.cu"
#include "integration/photonDispersion.cu"

#include "camera/ProjectiveCamera.cu"
#include "sampling/Sampler.cu"
#include "light/PointLight.cu"
#include "materials/LambertianMaterial.cu"
#include "materials/LightMaterial.cu"
#include "materials/GlassMaterial.cu"
#include "materials/MirrorMaterial.cu"
#include "materials/SSMaterial.cu"
#include "materials/PlasticUnrealMaterial.cu"

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
    auto sphere1 = Sphere(Point3f(-0.125, 0.25, -0.35), 0.15);
    auto sphere2 = Sphere(Point3f(0, 0, 0), 0.15);

    //auto sphere2 = Sphere(Point3f(0.125, -0.25, -0.35), 0.15);

    auto worldSphere = Sphere(Point3f(0, 0, 0), 100000);


    auto plane1 = Plane(Point3f(0.5, -0.5, -0.5), 
                        Vector3f(0, 1, 0), 
                        Vector3f(0, 0, 1), 
                        Vector3f(-0.5, 0, 0));

    auto plane2 = Plane(Point3f(0.5, 0.5, -0.5), 
                        Vector3f(-1, 0, 0), 
                        Vector3f(0, 0, 1), 
                        Vector3f(0, -0.5, 0));

    auto plane3 = Plane(Point3f(-0.5, -0.5, -0.5), 
                        Vector3f(0, 1, 0), 
                        Vector3f(0, 0, 1), 
                        Vector3f(0.5, 0, 0));

    auto plane4 = Plane(Point3f(0.5, -0.5, -0.5), 
                        Vector3f(-1, 0, 0), 
                        Vector3f(0, 0, 1), 
                        Vector3f(0, 0.5, 0));

    auto plane5 = Plane(Point3f(-0.5, -0.5, -0.5), 
                        Vector3f(1, 0, 0), 
                        Vector3f(0, 1, 0), 
                        Vector3f(0, 0, 0.5));

    auto plane6 = Plane(Point3f(-0.5, -0.5, 0.5), 
                        Vector3f(1, 0, 0), 
                        Vector3f(0, 1, 0), 
                        Vector3f(0, 0, -0.5));

    auto planeFront = Plane(Point3f(0.5, -0.5, -0.5), 
                        Vector3f(0, 1, 0), 
                        Vector3f(0, 0, 1), 
                        Vector3f(-0.5, 0, 0));

    LambertianMaterial blueMaterial(Spectrum(0.4f, 0.73f, 1.0f));
    LambertianMaterial greenMaterial(Spectrum(0.0f, 1.0f, 0.0f));
    LambertianMaterial redMaterial(Spectrum(1.0f, 0.0f, 0.0f));
    LambertianMaterial whiteMaterial(Spectrum(1.0f, 1.0f, 1.0f));
    LambertianMaterial greyMaterial(Spectrum(0.7f, 0.7f, 0.7f));
    SSMaterial marbleMaterial(Spectrum(0), Spectrum(0.83, 0.79, 0.75), Spectrum(0), 1.3, 0, nullptr, false, 
            Spectrum(2.19, 2.62, 3), Spectrum(0.0021, 0.00041, 0.0071), 0);
    LambertianMaterial marbleBRDF(Spectrum(0.83, 0.79, 0.75));

    LightMaterial lightMaterial(Spectrum(1, 1, 1), 4);
    GlassMaterial glassMaterial(Spectrum(1, 1, 1));
    MirrorMaterial mirror(Spectrum(1, 1, 1));
    PlasticUnrealMaterial plastic(Spectrum(1, 1, 1), 1);

    TrowbridgeMaterial beckmann(Spectrum(1, 1, 1), Spectrum(0.169, 0.421, 0.45), Spectrum(3.88, 2.343, 1.770), 0.8, 0.01);

    std::shared_ptr<std::vector<TriangleMesh>> triangleMeshes = std::make_shared<std::vector<TriangleMesh>>();
    std::vector<std::vector<Triangle>> triangles;
    TriangleMesh mesh1;
    std::vector<Triangle> mesh1Triangles;

    //loadObj("./models/basics/cube.obj", triangleMeshes, triangles);
    //loadObj("/export/d03/scratch/a816678/pathtracing_renderer/models/basics/cube.obj", triangleMeshes, triangles);


    //Texture *texture = new Texture("/home/hsunekichi/Escritorio/pathtracing_renderer/bricks.bmp");
    Texture *texture = new Texture("./bricks.bmp");
    
    //Texture *sph_texture = new SphericalTexture("/home/hsunekichi/Escritorio/pathtracing_renderer/marsSPH.bmp");
    //Texture *sph_texture = new SphericalTexture("/home/hsunekichi/Escritorio/pathtracing_renderer/marsSPH.bmp");

    LambertianMaterial bricksMaterial(Spectrum(1), Spectrum(1), texture);
    //LambertianMaterial marsMaterial(Spectrum(1), Spectrum(1), sph_texture);

    sphere1.setMaterial(&plastic);
    sphere2.setMaterial(&beckmann);
    worldSphere.setMaterial(&whiteMaterial);


    plane1.setMaterial(&whiteMaterial);
    plane2.setMaterial(&greenMaterial);
    plane3.setMaterial(&whiteMaterial);
    plane4.setMaterial(&redMaterial);
    plane5.setMaterial(&whiteMaterial);
    plane6.setMaterial(&whiteMaterial);
    planeFront.setMaterial(&glassMaterial);

    //sphereL.setMaterial(&lightMaterial);

    // Create device array with the shape*
    std::vector<Shape*> shapes;
    //shapes.push_back(&sphere1);
    //shapes.push_back(&sphere2);
    //shapes.push_back(&plane2);
    //shapes.push_back(&worldSphere);

    shapes.push_back(&plane1);
    shapes.push_back(&plane2);
    //shapes.push_back(&plane3);
    shapes.push_back(&plane4);
    shapes.push_back(&plane5);
    shapes.push_back(&plane6);
    //shapes.push_back(&planeFront);

    std::vector<Light*> lights;
    Camera *camera;

    //loadObejota("/home/hsunekichi/Escritorio/pathtracing_renderer/box.obejota", lights, shapes, camera);
    loadObejota("./cornellBoxGuion.obejota", lights, shapes, camera);

    Scene scene;
    scene.build(shapes, lights, triangles, triangleMeshes);
   
    //PathTracerIntegrator integrator(&camera, &sampler);
    PathTracerIntegrator integrator(camera);

    integrator.render(&scene);
    Film *film = camera->getFilm();

    //film.filterIndirectLight();
    //film.loadContributionsFromFile("test");
    film->storeContributionsToFile("cornellBox");
    
    if (RESCALE_IMAGE == true)
        film->rescale(3840, 2160);
        

    film->writeToBMP("cornellBoxGuion", ToneMappingType::gammaClamp);
}