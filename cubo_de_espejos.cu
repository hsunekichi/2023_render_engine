
#include "shapes/Sphere.cu"
#include "shapes/Plane.cu"
#include "shapes/Triangle.cu"

#include "integration/Scene.cu"
#include "integration/GPUPathTracerIntegrator.cu"

#include "camera/ProjectiveCamera.cu"
#include "sampling/Sampler.cu"
#include "light/PointLight.cu"
#include "materials/MatteMaterial.cu"
#include "materials/LightMaterial.cu"
#include "materials/BricksMaterial.cu"
#include "materials/GlassMaterial.cu"
#include "materials/MirrorMaterial.cu"

#include "textures/Texture.cu"
#include "textures/SphericalTexture.cu"

#include "cuda_libraries/FileImporting.cu"

int main()
{
    // Create a sphere
    auto sphere1 = Sphere(Point3f(4, 4, 4), 1);

    auto sphere2 = Sphere(Point3f(4, 4, -4), 1);

    auto sphereL = Sphere(Point3f(0, -3.5, 0), 
                        Vector3f(0, 0, 4),
                        Point3f(0, -5.5, 0));

    auto worldSphere = Sphere(Point3f(0, 0, 0), 100000);


    auto plane1 = Plane(Point3f(6, -6, -6), 
                        Vector3f(0, 12, 0), 
                        Vector3f(0, 0, 12), 
                        Vector3f(-1, 0, 0));

    auto plane2 = Plane(Point3f(6, 6, -6), 
                        Vector3f(-12, 0, 0), 
                        Vector3f(0, 0, 12), 
                        Vector3f(0, -1, 0));

    auto plane3 = Plane(Point3f(-6, -6, -6), 
                        Vector3f(0, 12, 0), 
                        Vector3f(0, 0, 12), 
                        Vector3f(1, 0, 0));

    auto plane4 = Plane(Point3f(6, -6, -6), 
                        Vector3f(-12, 0, 0), 
                        Vector3f(0, 0, 12), 
                        Vector3f(0, 1, 0));

    auto plane5 = Plane(Point3f(-6, -6, -6), 
                        Vector3f(12, 0, 0), 
                        Vector3f(0, 12, 0), 
                        Vector3f(0, 0, 1));

    auto plane6 = Plane(Point3f(-6, -6, 6), 
                        Vector3f(12, 0, 0), 
                        Vector3f(0, 12, 0), 
                        Vector3f(0, 0, -1));

    MatteMaterial blueMaterial(Spectrum(0.4f, 0.73f, 1.0f));
    MatteMaterial redMaterial(Spectrum(1.0f, 0.2f, 0.3f));
    MatteMaterial whiteMaterial(Spectrum(1.0f, 1.0f, 1.0f));
    MatteMaterial greyMaterial(Spectrum(0.7f, 0.7f, 0.7f));
    LightMaterial lightMaterial(Spectrum(1, 1, 1), 2);
    GlassMaterial glassMaterial(Spectrum(1, 1, 1));
    MirrorMaterial mirror(Spectrum(1, 1, 1));


    std::shared_ptr<std::vector<TriangleMesh>> triangleMeshes = std::make_shared<std::vector<TriangleMesh>>();
    std::vector<std::vector<Triangle>> triangles;
    TriangleMesh mesh1;
    std::vector<Triangle> mesh1Triangles;

    Texture *texture = new Texture("/home/hsunekichi/Escritorio/pathtracing_renderer/bricks.bmp");
    Texture *sph_texture = new SphericalTexture("/home/hsunekichi/Escritorio/pathtracing_renderer/marsSPH.bmp");

    //Texture *texture = new Texture("/export/d03/scratch/a816678/pathtracing_renderer/bricks.bmp");
    //Texture *sph_texture = new SphericalTexture("/export/d03/scratch/a816678/pathtracing_renderer/marsSPH.bmp");


    BricksMaterial bricksMaterial(texture);
    BricksMaterial sph_bricksMaterial(sph_texture);

    sphere1.setMaterial(&blueMaterial);
    sphere2.setMaterial(&blueMaterial);
    worldSphere.setMaterial(&lightMaterial);


    plane1.setMaterial(&bricksMaterial);
    plane2.setMaterial(&bricksMaterial);
    plane3.setMaterial(&whiteMaterial);
    plane4.setMaterial(&whiteMaterial);
    plane5.setMaterial(&bricksMaterial);
    plane6.setMaterial(&lightMaterial);

    //sphereL.setMaterial(&lightMaterial);

    // Create device array with the shape*
    std::vector<Shape*> shapes;
    shapes.push_back(&sphere1);
    shapes.push_back(&sphere2);
    //shapes.push_back(&plane2);
    //shapes.push_back(&worldSphere);

    shapes.push_back(&plane1);
    shapes.push_back(&plane2);
    shapes.push_back(&plane3);
    shapes.push_back(&plane4);
    shapes.push_back(&plane5);
    shapes.push_back(&plane6);


    Light *light1 = new PointLight(Transformations::translate(Vector3f(0, 3, 0)), 
                                    Spectrum(1.0f, 1.0f, 1.0f), 30);

    Light *light2 = new PointLight(Transformations::translate(Vector3f(4.5, -0, 0)), 
                                    Spectrum(1.0f, 1.0f, 1.0f), 7);

    std::vector<Light*> lights;
    //lights.push_back(light1);
    //lights.push_back(light2);

    Scene scene;
    scene.build(shapes, lights, triangles, triangleMeshes);
   
    Transform lookAt = Transformations::lookAt(Point3f(0, 0, 0), Point3f(1, 0, 0), Vector3f(0, 0, 1));
    Transform cameraToScreen = Transformations::translate(Vector3f(0, 0, -1));

    Film film;

    Float verticalFrameSize = 2;
    verticalFrameSize /= 2;
    Float horizontalFrameSize = (verticalFrameSize/resolutionY) * resolutionX;
    
    ProjectiveCamera camera (inverse(lookAt), cameraToScreen, 
                Bound2f(Point2f(-horizontalFrameSize, -verticalFrameSize), 
                        Point2f(horizontalFrameSize, verticalFrameSize)),
                0, 0, &film, nullptr);

    //PathTracerIntegrator integrator(&camera, &sampler);
    GPUPathTracerIntegrator integrator(&camera);

    integrator.render(&scene);

    //film.filterIndirectLight();

    //film.loadContributionsFromFile("test");
    
    film.storeContributionsToFile("test");
    
    if (RESCALE_IMAGE == true)
        film.rescale(3840, 2160);
        
    film.writeToBMP("test", ToneMappingType::reinhardJodie);
}