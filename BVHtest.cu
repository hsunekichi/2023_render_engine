
#include <random>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>

#include "cuda_libraries/BVH.cu"
#include "geometry/Ray.cu"
#include "shapes/Box.cu"
#include "shapes/Triangle.cu"
#include "geometry/Vector.cu"
#include "cuda_libraries/GPUVector.cu"
#include "cuda_libraries/GPUPair.cu"

#include <chrono>
#include <thrust/random.h>



// Functor to generate random Bound3f boxes
struct RandomBound3f {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist;

    __host__ __device__
    void operator()(const int& seed) {
        rng.discard(2);  // Discard a few values to avoid correlations
        
        Point3f p1 = Point3f(dist(rng), dist(rng), dist(rng));
        Point3f p2 = Point3f(dist(rng), dist(rng), dist(rng));
        Point3f p3 = Point3f(dist(rng), dist(rng), dist(rng));

        //return Triangle(p1, p2, p3);
    }
};

// Functor to generate random Ray objects
struct RandomRay {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist;

    __host__ __device__
    Ray operator()(const int& seed) {
        rng.discard(2);  // Discard a few values to avoid correlations
        Point3f origin(dist(rng), dist(rng), dist(rng));
        Vector3f direction(dist(rng), dist(rng), dist(rng));

        return Ray(origin, direction);
    }
};



int nRays = 1;
int nBoxes = 2;

int main() 
{
    /*
    thrust::device_vector<Ray> rays(nRays);
    thrust::device_vector<int> raySeeds(nRays);

    thrust::device_vector<Bound3f> boxes(nBoxes);
    thrust::device_vector<int> boxSeeds(nBoxes);


    // Create a sequence of seeds for random number generation
    thrust::sequence(raySeeds.begin(), raySeeds.end());
    thrust::sequence(boxSeeds.begin(), boxSeeds.end());

    // Use thrust::transform to generate random Ray objects
    thrust::transform(raySeeds.begin(), raySeeds.end(), rays.begin(), RandomRay());
    thrust::transform(boxSeeds.begin(), boxSeeds.end(), boxes.begin(), RandomBound3f());

    raySeeds.clear();
    boxSeeds.clear();
*/

    std::shared_ptr<TriangleMesh> triangleMesh (new TriangleMesh());
    std::vector<Triangle> triangles;

    triangleMesh->vertices.push_back(Point3f(2, -1, -1));
    triangleMesh->vertices.push_back(Point3f(2, 0, 1));
    triangleMesh->vertices.push_back(Point3f(2, 1, -1));
    triangleMesh->vertices.push_back(Point3f(2, 2, 1));

    int vertices1[3] = {0, 1, 2};
    int vertices2[3] = {1, 2, 3};
    triangles.push_back(Triangle(vertices1));
    triangles.push_back(Triangle(vertices2));

    std::cout << "Samples generated" << std::endl;

    // Create BVH
    auto startBuild = std::chrono::high_resolution_clock::now();
    BVH bvh (triangleMesh, triangles);
    auto endBuild = std::chrono::high_resolution_clock::now();

    // Get time in microseconds
    auto durationBuild = std::chrono::duration_cast<std::chrono::microseconds>(endBuild - startBuild);

    std::cout << "Time taken to build BVH: " << durationBuild.count()/1000 << " ms" << std::endl;
    std::cout << "Time per box: " << durationBuild.count()/nBoxes << " microseconds" << std::endl;

    std::vector<Ray> host_rays;
    host_rays.push_back(Ray(Point3f(0, 1.5, 0), Vector3f(1, 0, 0)));

    
    // Get time in microseconds
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<SurfaceInteraction> collisionIds = bvh.intersectRays(host_rays);
    auto end = std::chrono::high_resolution_clock::now();

    // Get time in microseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Time per ray: " << duration.count()/nRays << " microseconds" << std::endl;
    std::cout << "Time per box: " << duration.count()/nBoxes << " microseconds" << std::endl;
    std::cout << "Time per ray per box: " << duration.count()/(nRays*nBoxes) << " microseconds" << std::endl;

    // Print collisions
    for (int i = 0; i < collisionIds.size(); i++) 
    {
        SurfaceInteraction s = collisionIds[i];
        
        if (s.shapeId != -1)
        {
            std::cout << "Ray " << i << " collided with object: " << s.shapeId << std::endl;
        }
        else {
            std::cout << "Ray " << i << " did not collide" << std::endl;
        }
    }
        

    return 0;
}