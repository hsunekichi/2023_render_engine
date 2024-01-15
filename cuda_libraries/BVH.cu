#pragma once

#include "GPUVector.cu"
#include "GPUStack.cu"
#include "GPUPair.cu"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <queue>

#include "math.cu"
#include "geometricMath.cu"
#include "../geometry/Ray.cu"
#include "../geometry/BoundingBox.cu"
#include "../shapes/Triangle.cu"

struct SampleState
{
    Point2i filmOrigin;
    Ray pathRay;

    int depth = 0;
    bool finished = false;
    bool lastBunceWasSpecular = false;
    Spectrum totalLight = Spectrum(0.0);
    Spectrum pathLightFactor = 1;
    bool intersected = false;

    Interaction *interaction = nullptr;

    // Data for lightning denoising
    unsigned int firstShapeId;
    Point2f firstSampleSurfacePoint;
};


__global__
void printTriangles(const Triangle *t, int nT)
{
    for (int i = 0; i < nT; i++)
    {
        printf("Triangle %d\n", i);
        printf("Triangle mesh: %p\n", t[i].mesh);
        printf("Triangle vertices: %p\n", t[i].mesh->vertices);
        
        printf("Vertex id0: %d\n", t[i].vertexIndices[0]);
        printf("Vertex id1: %d\n", t[i].vertexIndices[1]);
        printf("Vertex id2: %d\n", t[i].vertexIndices[2]);

        printf("Vertex 0: %f %f %f\n", t[i].mesh->vertices[t[i].vertexIndices[0]].x, t[i].mesh->vertices[t[i].vertexIndices[0]].y, t[i].mesh->vertices[t[i].vertexIndices[0]].z);
        printf("Vertex 1: %f %f %f\n", t[i].mesh->vertices[t[i].vertexIndices[1]].x, t[i].mesh->vertices[t[i].vertexIndices[1]].y, t[i].mesh->vertices[t[i].vertexIndices[1]].z);
        printf("Vertex 2: %f %f %f\n", t[i].mesh->vertices[t[i].vertexIndices[2]].x, t[i].mesh->vertices[t[i].vertexIndices[2]].y, t[i].mesh->vertices[t[i].vertexIndices[2]].z);


        if (t[i].mesh->vertices[t[i].vertexIndices[0]].hasInf()
            || t[i].mesh->vertices[t[i].vertexIndices[1]].hasInf()
            || t[i].mesh->vertices[t[i].vertexIndices[2]].hasInf())
        {
            printf("Triangle %d\n", i);
        }

        if (t[i].mesh->vertices[t[i].vertexIndices[0]].hasInf())
            printf("Vertex 0: %f %f %f\n", t[i].mesh->vertices[t[i].vertexIndices[0]].x, t[i].mesh->vertices[t[i].vertexIndices[0]].y, t[i].mesh->vertices[t[i].vertexIndices[0]].z);
        
        if (t[i].mesh->vertices[t[i].vertexIndices[1]].hasInf())
            printf("Vertex 1: %f %f %f\n", t[i].mesh->vertices[t[i].vertexIndices[1]].x, t[i].mesh->vertices[t[i].vertexIndices[1]].y, t[i].mesh->vertices[t[i].vertexIndices[1]].z);
        
        if (t[i].mesh->vertices[t[i].vertexIndices[2]].hasInf())
            printf("Vertex 2: %f %f %f\n", t[i].mesh->vertices[t[i].vertexIndices[2]].x, t[i].mesh->vertices[t[i].vertexIndices[2]].y, t[i].mesh->vertices[t[i].vertexIndices[2]].z);
    }
}

__global__
void printMeshData (TriangleMeshData *meshData)
{
    int nV = 2;

    printf("Mesh data\n");
    printf("Mesh direction: %p\n", meshData);
    printf("Vertices: %p\n", meshData->vertices);
    printf("Normals: %p\n", meshData->normals);
    printf("UVs: %p\n", meshData->textureCoords);

    for (int i = 0; i < nV; i++)
    {
        printf("Vertex %d: %f %f %f\n", i, meshData->vertices[i].x, meshData->vertices[i].y, meshData->vertices[i].z);
    }
}

class BVH 
{
    public:

    struct Node
    {
        unsigned int parent;
        unsigned int leftChild;
        unsigned int rightChild;
        
        Bound3f boundingBox;
        int objectId = -1;
        unsigned int readyToInit = 0;

        __host__ __device__
        inline bool isLeaf() const
        {
            return objectId != -1;
        }
    };

    unsigned int numLeaves;

    unsigned int root;
    thrust::device_vector<Node> d_treeNodes;

    thrust::device_vector<Triangle> d_triangles;
    thrust::device_vector<TriangleMeshData> d_gpuMeshData;
    const std::shared_ptr<std::vector<TriangleMesh>> cpuMeshes;

    thrust::host_vector<Node> treeNodes;
    thrust::host_vector<Triangle> triangles;
    

    __host__ __device__
    static void initializeLeaf(unsigned int idx, 
            Node *leafNodes, 
            const Triangle *triangles,
            unsigned int* sortedObjectIDs)
    {
        unsigned int objectID = sortedObjectIDs[idx];
        leafNodes[idx].boundingBox = triangles[objectID].worldBound();
        leafNodes[idx].objectId = objectID;
    }

    void initializeLeafs(thrust::host_vector<unsigned int> &sortedMortonCodes,
        thrust::host_vector<unsigned int> &sortedObjectIds,
        thrust::host_vector<Node> &treeNodes,
        unsigned int numObjects)
    {
        thrust::device_vector<unsigned int> d_sortedObjectIds(sortedObjectIds);
        thrust::device_vector<Node> d_treeNodes(treeNodes);

        // Initialize leaf nodes in parallel
        thrust::for_each(
            thrust::counting_iterator<unsigned int>(0), 
            thrust::counting_iterator<unsigned int>(numObjects), 
            IntitializeLeafFunctor(d_sortedObjectIds.data().get(), d_treeNodes, d_triangles));
    
        treeNodes = d_treeNodes;
    }

    unsigned int generateHierarchy(thrust::host_vector<unsigned int> &sortedMortonCodes,
        thrust::host_vector<unsigned int> &sortedObjectIds,
        thrust::host_vector<Node> &treeNodes,
        unsigned int numObjects)
    {
        std::queue<unsigned int> freeIds;

        for (int i = 0; i < numObjects - 1; i++)
        {
            freeIds.push(numObjects + i);
        }

        // First numObjects elements are the leaves
        for (int i = 0; i < numObjects; i++)
        {
            unsigned int parentId = freeIds.front();
            treeNodes[i].parent = parentId;

            if (i % 2 == 0)
            {
                treeNodes[parentId].leftChild = i;
            }            
            else
            { 
                treeNodes[parentId].rightChild = i;
                freeIds.pop();
            }
        }

        // All internal nodes except the root
        for (int i = numObjects; i < numObjects*2 - 2; i++)
        {
            unsigned int parentId = freeIds.front();
            treeNodes[i].parent = parentId;

            if (i % 2 == 0)
            {
                treeNodes[parentId].leftChild = i;
            }            
            else
            { 
                treeNodes[parentId].rightChild = i;
                freeIds.pop();
            }
        }

        // Root
        return numObjects*2 - 2;
    }

    void computeBoundingBoxes(
            thrust::host_vector<Node> &treeNodes, 
            unsigned int numObjects)
    {
        for (int i = numObjects; i < treeNodes.size(); i++)
        {
            Node *node = &treeNodes[i];
            boundingBoxNode(node, treeNodes.data());
        }
    }


    // Builds the tree from the sorted data.
    unsigned int generateTree(
        thrust::host_vector<unsigned int> &sortedMortonCodes,
        thrust::host_vector<unsigned int> &sortedObjectIds,
        unsigned int numObjects)
    {
        // Allocate nodes
        treeNodes = thrust::host_vector<Node> (numObjects*2 - 1);

        initializeLeafs(sortedMortonCodes, sortedObjectIds, treeNodes, numObjects);

        unsigned int root = generateHierarchy(sortedMortonCodes, sortedObjectIds, treeNodes, numObjects);

        // Compute bounding boxes
        computeBoundingBoxes(treeNodes, numObjects);

        d_treeNodes = treeNodes;

        return root;
    }

    __host__ __device__
    static void boundingBoxNode (Node *node, Node *treeNodes)
    {
        Node *leftChild = &treeNodes[node->leftChild];
        Node *rightChild = &treeNodes[node->rightChild];

        Bound3f &leftBox = leftChild->boundingBox;
        Bound3f &rightBox = rightChild->boundingBox;

        node->boundingBox = leftBox + rightBox;
    }

    thrust::device_vector<TriangleMeshData> copyMeshData(std::vector<TriangleMesh> &meshes)
    {
        thrust::device_vector<TriangleMeshData> gpuMeshes(meshes.size());

        for (int i = 0; i < meshes.size(); i++)
        {
            TriangleMeshData tData;

            tData.vertices = meshes[i].vertices.data().get();

            if (meshes[i].normals.size() > 0)
                tData.normals = meshes[i].normals.data().get();
            
            if (meshes[i].textureCoords.size() > 0)
                tData.textureCoords = meshes[i].textureCoords.data().get();

            gpuMeshes[i] = tData;
        }

        return gpuMeshes;
    }

    void initializeTriangles(std::vector<std::vector<Triangle>> &cpuTriangles, 
        thrust::device_vector<TriangleMeshData> &gpuMeshData)
    {
        for (int i = 0; i < gpuMeshData.size(); i++)
        {
            auto &meshTriangles = cpuTriangles[i];
            TriangleMeshData *meshData = gpuMeshData.data().get() + i;

            if (gpuMeshData.size() == 0)
                throw std::runtime_error("Mesh data is empty");

            // Populate the mesh triangles with its mesh data
            for (int j = 0; j < meshTriangles.size(); j++)
            {
                meshTriangles[j].setMeshData(meshData);
            }
        }

        for (int i = 0; i < cpuTriangles.size(); i++)
        {
            d_triangles.insert(d_triangles.end(), cpuTriangles[i].begin(), cpuTriangles[i].end());
        }
    }

    unsigned int computeNumLeaves(std::vector<std::vector<Triangle>> &cpuTriangles)
    {
        unsigned int numLeaves = 0;

        for (int i = 0; i < cpuTriangles.size(); i++)
        {
            numLeaves += cpuTriangles[i].size();
        }

        return numLeaves;
    }

    /***************************************************/
    /****************** PUBLIC METHODS *****************/
    /***************************************************/

    public:

    // This BVH parallel construction is an implementation of the algorithm described in
    // "Thinking Parallel, Part III: Tree Construction on the GPU" by Tero Karras
    BVH(std::shared_ptr<std::vector<TriangleMesh>> meshes, 
        std::vector<std::vector<Triangle>> &initTriangles,
        std::vector<TriangleMeshData> &meshData,
        std::vector<Triangle> &cpuTriangles)
    : numLeaves(computeNumLeaves(initTriangles)),
        cpuMeshes(meshes)
    {
        // Copy of the triangles to initialize them for the gpu
        auto cpuInitTriangles(initTriangles);

        d_gpuMeshData = copyMeshData(*meshes);
        initializeTriangles(cpuInitTriangles, d_gpuMeshData);

        // Sort the data by morton code
        thrust::device_vector<unsigned int> morton_positions(numLeaves);
        thrust::device_vector<unsigned int> sorted_element_ids(numLeaves);
        sort_by_morton(d_triangles, morton_positions, sorted_element_ids);

        thrust::host_vector<unsigned int> h_morton_positions(morton_positions);
        thrust::host_vector<unsigned int> h_sorted_element_ids(sorted_element_ids);

        // Generate the hierarchy
        root = generateTree(h_morton_positions, 
                h_sorted_element_ids, 
                d_triangles.size());

        triangles = cpuTriangles;
    }


    unsigned int getRoot() const
    {
        return root;
    }

    unsigned int getNumLeaves() const
    {
        return numLeaves;
    }

    __host__ __device__ 
    static int intersectRay(const Ray& queryRay, 
        const unsigned int root_id,
        const Node *treeNodes,
        const Triangle *triangles,
        bool checkShadows = false)
    {
        int hitId = -1;
        Float hitOffset = INFINITY;

        // Get the root node
        const Node *root = &treeNodes[root_id];

        // Allocate traversal stack from thread-local memory
        // Register memory, with 64 there is space for up to 64 depth
        const Node *stack[64];  
        int stackPtr = -1;

        // Traverse nodes starting from the root

        /*********** Base cases ***********/
        if (root == nullptr) {   // Empty tree
            return hitId;
        }
        else if (root->isLeaf())    // Root is a leaf
        {   
            Triangle leafData = triangles[root->objectId];
            
            if(leafData.intersects(queryRay, hitOffset, checkShadows))
            {
                hitId = root->objectId;
            }

            return hitId;
        }


        /*********** General case ***********/
        stackPtr++;
        stack[stackPtr] = root; // Push root
        

        // While there are nodes to traverse
        while (stackPtr >= 0)
        {
            const Node *node = stack[stackPtr];     // Gets the next node
            stackPtr--;

            // Check each child node for overlap.
            const Node *childL = &treeNodes[node->leftChild];
            const Node *childR = &treeNodes[node->rightChild];

            if (childL != nullptr)
            {
                Float tempHitOffset = hitOffset;

                if (childL->isLeaf())
                {
                    const Triangle &leafData = triangles[childL->objectId];

                    if (leafData.intersects(queryRay, hitOffset, checkShadows))
                    {
                        hitId = childL->objectId;
                    }
                }
                else if (childL->boundingBox.intersects(queryRay, tempHitOffset))
                {
                    stackPtr++;
                    stack[stackPtr] = childL;
                }
            }

            if (childR != nullptr)
            {
                Float tempHitOffset = hitOffset;

                if (childR->isLeaf())
                {
                    const Triangle &leafData = triangles[childR->objectId];

                    if (leafData.intersects(queryRay, hitOffset, checkShadows))
                    {
                        hitId = childR->objectId;
                    }
                }
                else if (childR->boundingBox.intersects(queryRay, tempHitOffset))
                {
                    stackPtr++;
                    stack[stackPtr] = childR;
                }
            }
        }

        return hitId;
    }


    __host__ __device__
    static int intersectBruteForce(const Ray& queryRay, 
        const Triangle *triangles,
        const Node *leafs,
        int numTriangles,
        bool checkShadows = false)
    {
        int hitId = -1;
        Float hitOffset = INFINITY;

        for (int i = 0; i < numTriangles; i++)
        {
            Float tempOffset = INFINITY;

            if (leafs[i].boundingBox.intersects(queryRay, tempOffset))
            {
                int objectId = leafs[i].objectId;
                if (triangles[objectId].intersects(queryRay, hitOffset, checkShadows))
                {
                    hitId = objectId;
                }
            }
        }

        return hitId;
    }

    class IntersectRayFunctor
    {
        public:
        unsigned int root;
        const Node *treeNodes;
        const Triangle *triangles;
        bool checkShadows;

        __host__
        IntersectRayFunctor(unsigned int root, const Node *treeNodes, const Triangle *triangles, bool checkShadows)
        {
            this->root = root;
            this->treeNodes = treeNodes;
            this->triangles = triangles;
        }

        __host__ __device__
        int operator()(const Ray& sample)
        {
            return intersectRay(sample, root, treeNodes, triangles, checkShadows);
        }
    };


    class IntersectBruteForceFunctor
    {
        public:
        const Triangle *triangles;
        const Node *leafs;
        int numTriangles;
        bool checkShadows;

        __host__
        IntersectBruteForceFunctor(const Triangle *triangles, 
                const Node *leafs, 
                int numTriangles,
                bool checkShadows = false)
        {
            this->triangles = triangles;
            this->numTriangles = numTriangles;
            this->leafs = leafs;
            this->checkShadows = checkShadows;
        }

        __host__ __device__
        int operator()(const Ray& sample)
        {
            return intersectBruteForce(sample, triangles, leafs, numTriangles, checkShadows);
        }
    };

    template <typename F>
    void th_apply_functor_intersectCPU(const thrust::host_vector<Ray> &samples, 
            thrust::host_vector<int> &hitIds,
            F functor,
            int iMin, int iMax) const
    {
        for (int i = iMin; i < iMax; i++)
        {
            hitIds[i] = functor(samples[i]);
        }
    }
    
    void intersectRays(const thrust::host_vector<Ray> &samples, 
            thrust::host_vector<int> &hitIds,
            bool checkShadows = false,
            cudaStream_t stream = 0) const
    {
        // Get time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);

        intersectRaysGpu(samples, hitIds, checkShadows, stream);

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        //std::cout << "Time: " << milliseconds << std::endl;
    }

    void intersectRaysGpu(const thrust::host_vector<Ray> &samples, 
            thrust::host_vector<int> &hitIds,
            bool checkShadows = false,
            cudaStream_t stream = 0) const
    {
        thrust::device_vector<Ray> samplesDevice(samples);
        thrust::device_vector<int> hitIdsDevice(samples.size());

        auto functor = IntersectRayFunctor(getRoot(), d_treeNodes.data().get(), d_triangles.data().get(), checkShadows);

        //auto functor = IntersectBruteForceFunctor(d_triangles.data().get(), 
        //        d_treeNodes.data().get(), 
        //        d_triangles.size(),
        //        checkShadows);

        thrust::transform(
                samplesDevice.begin(), 
                samplesDevice.end(), 
                hitIdsDevice.begin(), 
                functor);        

        hitIds = hitIdsDevice;
    }

    void intersectRaysCpu(const thrust::host_vector<Ray> &samples, 
            thrust::host_vector<int> &hitIds,
            bool checkShadows = false,
            cudaStream_t stream = 0) const
    {
        hitIds = thrust::host_vector<int>(samples.size());

        auto functor = IntersectRayFunctor(getRoot(), treeNodes.data(), triangles.data(), checkShadows);
        // auto functor = IntersectBruteForceFunctor(triangles.data(), 
        //        treeNodes.data(), 
        //        triangles.size(),
        //        checkShadows);

        //th_apply_functor_intersectCPU(samples, functor, 0, samples.size());

        // Split in 128 chunks 
        int numChunks = 128;
        int chunkSize = samples.size() / numChunks;

        // Launch 128 threads to apply functor th_apply_functor_intersectCPU
        std::vector<std::thread> threads;

        for (int i = 0; i < numChunks; i++)
        {
            threads.push_back(std::thread(&BVH::th_apply_functor_intersectCPU<IntersectRayFunctor>, 
                        this, 
                        std::ref(samples), 
                        std::ref(hitIds),
                        functor, 
                        i*chunkSize, 
                        (i+1)*chunkSize));
        }

        if (chunkSize > 0 && samples.size() % chunkSize != 0)
        {
            threads.push_back(std::thread(&BVH::th_apply_functor_intersectCPU<IntersectRayFunctor>, 
                        this, 
                        std::ref(samples), 
                        std::ref(hitIds),
                        functor, 
                        numChunks*chunkSize, 
                        samples.size()));
        }

        // Wait for all threads to finish
        for (int i = 0; i < threads.size(); i++)
        {
            threads[i].join();
        }
    }

    class IntitializeLeafFunctor
    {
        public:
        unsigned int *sortedObjectIDs;
        Node *treeNodes;
        const Triangle *triangles;

        __host__ 
        IntitializeLeafFunctor(unsigned int *sortedObjectIDs, 
                thrust::device_vector<Node> &treeNodes, 
                const thrust::device_vector<Triangle> &triangles)
        {
            this->sortedObjectIDs = sortedObjectIDs;
            this->treeNodes = treeNodes.data().get();
            this->triangles = triangles.data().get();
        }

        __host__ __device__
        void operator()(unsigned int idx)
        {
            initializeLeaf(idx, treeNodes, triangles, sortedObjectIDs);
        }
    };
};


