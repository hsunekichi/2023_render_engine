#pragma once

#include "GPUVector.cu"
#include "GPUStack.cu"
#include "GPUPair.cu"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>

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
        Node *parent = nullptr;
        Node *leftChild = nullptr;
        Node *rightChild = nullptr;
        
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

    Node *root;
    thrust::device_vector<Node> leafNodes;
    thrust::device_vector<Node> internalNodes;

    thrust::device_vector<Triangle> triangles;
    thrust::device_vector<TriangleMeshData> gpuMeshData;
    const std::shared_ptr<std::vector<TriangleMesh>> cpuMeshes;

// Function directly provided by NVIDIA's thinking parallel
    __device__
    static int findSplit( unsigned int* sortedMortonCodes,
            unsigned int n,
            int first,
            int last)
    {
        // Only one object
        if (first == last)
            return -1;

        // If the morton codes are equal, split the range in the middle.
        if (sortedMortonCodes[first] == sortedMortonCodes[last])
            return (first + last)/2;

        // Compute the common prefix of all the objects in the range.
        int commonPrefix = common_prefix(first, last, sortedMortonCodes, n);

        // Compute the next object position.
        //  The next object is the first one that has more than
        //  commonPrefix bits different from the first one.
        int split = first; // initial guess
        int step = last - first;

        do
        {
            step = (step + 1)/2; // exponential decrease
            int newSplit = split + step; // proposed new position

            if (newSplit < last)
            {
                int splitPrefix = common_prefix(first, newSplit, sortedMortonCodes, n);

                if (splitPrefix > commonPrefix)
                    split = newSplit; // accept proposal
            }
        }
        while (step > 1);

        return split;
    }


    // Determine the range of objects with identical Morton codes
    __device__
    static int2 determineRange(unsigned int* sortedMortonCodes, int numObjects, int idx) 
    {
        int begin = idx;

        // Determine direction of the range (+1 or -1)
        int direction = common_prefix(begin,
                            begin+1,
                            sortedMortonCodes,
                            numObjects) 
                        - 
                        common_prefix(begin,
                            begin-1,
                            sortedMortonCodes,
                            numObjects);

        direction = clamp(direction, -1, 1);
         
        // Compute upper bound for the length of the range
        int minPrefix = common_prefix(begin,
                            begin - direction,
                            sortedMortonCodes,
                            numObjects);

        // Searches for an element far enough to have less than minPrefix common bits with begin
        int max_len = 2;
        while (common_prefix(begin,                       
                    begin + max_len*direction,
                    sortedMortonCodes,
                    numObjects) 
                > minPrefix)
        {
            max_len = max_len * 2;
        }

        // Find the other end using binary search
        int len = 0;
        for (int offset = max_len / 2; offset >= 1; offset = offset/2)
        {
            if (common_prefix(begin,
                    begin + (len + offset)*direction,
                    sortedMortonCodes,
                    numObjects) 
                > minPrefix)
            {
                len += offset;
            }
        }
        
        int end = begin + len*direction;

        int resX = min(begin, end);
        int resY = max(begin, end);

        return int2(resX, resY);
    }

    __device__
    static void construct_node(unsigned int idx, 
            unsigned int *sortedMortonCodes, 
            unsigned int *sortedObjectIDs, 
            unsigned int numObjects,
            Node *leafNodes,
            Node *internalNodes)
    {
        if (idx > numObjects - 1)
            return;

        // Find out which range of objects the node corresponds to.
        int2 range = determineRange(sortedMortonCodes, numObjects, idx);
        int first = range.x;
        int last = range.y;

        // Determine where to split the range.
        int split = findSplit(sortedMortonCodes, numObjects, first, last);

        // If the range corresponds to a single object, then the node is a leaf.
        if (split == -1)
            return;

        // Select leftChild.

        Node* leftChild;
        if (split == first) // If the left child is a leaf, record the objectID.
            leftChild = &leafNodes[split];
        else
            leftChild = &internalNodes[split];

        // Select rightChild.

        Node* rightChild;
        if (split + 1 == last)
            rightChild = &leafNodes[split + 1];
        else
            rightChild = &internalNodes[split + 1];

        // Record parent-child relationships.

        internalNodes[idx].leftChild = leftChild;
        internalNodes[idx].rightChild = rightChild;
        leftChild->parent = &internalNodes[idx];
        rightChild->parent = &internalNodes[idx];

        //printf("Node %p, lchild: %p, rchild: %p\n", &internalNodes[idx], leftChild, rightChild);
    }

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


    // Builds the tree from the sorted data.
    Node* generateHierarchy(unsigned int* sortedMortonCodes,
                        unsigned int*          sortedObjectIDs,
                        unsigned int           numObjects)
    {
        // Initialize leaf nodes
        IntitializeLeafFunctor initializeLeaf(sortedObjectIDs, leafNodes, triangles);

        // Construct all leaf nodes
        thrust::for_each(thrust::counting_iterator<unsigned int>(0), 
                            thrust::counting_iterator<unsigned int>(numObjects), 
                            initializeLeaf);

        ConstructNodeFunctor binded_construct_node(sortedMortonCodes, 
                                                    sortedObjectIDs, 
                                                    numObjects, 
                                                    leafNodes,
                                                    internalNodes);

        // Construct all internal nodes
        thrust::for_each(thrust::counting_iterator<unsigned int>(0), 
                            thrust::counting_iterator<unsigned int>(numObjects), 
                            binded_construct_node);

        // Node 0 is the root.
        Node *root;

        if (internalNodes.size() > 0)
            root = internalNodes.data().get();
        else
            root = leafNodes.data().get();

        //printf("Root: %p\n", root);

        return root;
    }

    __device__
    static void boundingBoxNode (Node *node)
    {
        Node *leftChild = node->leftChild;
        Node *rightChild = node->rightChild;

        Bound3f &leftBox = leftChild->boundingBox;
        Bound3f &rightBox = rightChild->boundingBox;

        node->boundingBox = leftBox + rightBox;
    }

    __device__
    static void generateBoundingBoxes(int idx, Node* leaves, int nLeaves)
    {
        if (idx >= nLeaves)
            return;

        // Start with the parent
        Node *node = leaves[idx].parent;

        // Until we reach the root or arrive first to the node
        while(node != nullptr)
        {
            // Check if we are the first to arrive
            int old = atomicInc(&node->readyToInit, 1);

            if (old > 0)
            {
                // The second thread bounds the node and continues
                boundingBoxNode(node); // Generate a bounding box for the father

                // Get the parent
                node = node->parent;           
            }
            else {
                node = nullptr;
            }
        }
    }

    class GenerateBoundingBoxesFunctor
    {
        Node *leaves;
        int nLeaves;

        public:
        __host__
        GenerateBoundingBoxesFunctor(Node *leaves, int nLeaves)
        {
            this->leaves = leaves;
            this->nLeaves = nLeaves;
        }

        __device__
        void operator()(int idx)
        {
            generateBoundingBoxes(idx, leaves, nLeaves);
        }
    };

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

            // Populate the mesh triangles with its mesh data
            for (int j = 0; j < meshTriangles.size(); j++)
            {
                meshTriangles[j].setMeshData(meshData);
            }
        }

        for (int i = 0; i < cpuTriangles.size(); i++)
        {
            triangles.insert(triangles.end(), cpuTriangles[i].begin(), cpuTriangles[i].end());
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
    BVH(std::shared_ptr<std::vector<TriangleMesh>> meshes, std::vector<std::vector<Triangle>> &initTriangles)
    : numLeaves(computeNumLeaves(initTriangles)),
        leafNodes(numLeaves), 
        internalNodes(numLeaves - 1),
        cpuMeshes(meshes)
    {
        // Copy of the triangles to initialize them for the gpu
        auto cpuTriangles(initTriangles);

        gpuMeshData = copyMeshData(*meshes);
        initializeTriangles(cpuTriangles, gpuMeshData);

        // Sort the data by morton code
        thrust::device_vector<unsigned int> morton_positions(numLeaves);
        thrust::device_vector<unsigned int> sorted_element_ids(numLeaves);
        sort_by_morton(triangles, morton_positions, sorted_element_ids);

        // Generate the hierarchy
        root = generateHierarchy(morton_positions.data().get(), sorted_element_ids.data().get(), triangles.size());

        // Generate bounding boxes
        GenerateBoundingBoxesFunctor functor(leafNodes.data().get(), numLeaves);
        
        thrust::for_each(thrust::counting_iterator<unsigned int>(0), 
                            thrust::counting_iterator<unsigned int>(numLeaves), 
                            functor);
    }


    Node* getRoot() const
    {
        return root;
    }

    unsigned int getNumLeaves() const
    {
        return numLeaves;
    }

    __device__ 
    static int intersectRay(Ray& queryRay, 
        const Node *root,
        const Triangle *triangles,
        bool checkShadows = false)
    {
        int hitId = -1;
        Float hitOffset = INFINITY;


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
            Node *childL = node->leftChild;
            Node *childR = node->rightChild;

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


    __device__
    static int intersectBruteForce(Ray& queryRay, 
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
        const Node *root;
        const Triangle *triangles;
        bool checkShadows;

        __host__
        IntersectRayFunctor(const Node *root, const Triangle *triangles, bool checkShadows)
        {
            this->root = root;
            this->triangles = triangles;
        }

        __device__
        int operator()(Ray& sample)
        {
            return intersectRay(sample, root, triangles, checkShadows);
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

        __device__
        int operator()(Ray& sample)
        {
            return intersectBruteForce(sample, triangles, leafs, numTriangles, checkShadows);
        }
    };
    
    void intersectRays(const thrust::host_vector<Ray> &samples, 
            thrust::host_vector<int> &hitIds,
            bool checkShadows = false,
            cudaStream_t stream = 0) const
    {
        thrust::device_vector<Ray> samplesDevice(samples);
        thrust::device_vector<int> hitIdsDevice(samples.size());

        //auto functor = IntersectRayFunctor(getRoot(), triangles.data().get(), checkShadows);
        auto functor = IntersectBruteForceFunctor(triangles.data().get(), 
                leafNodes.data().get(), 
                triangles.size(),
                checkShadows);

        thrust::transform(thrust::cuda::par.on(stream), 
                samplesDevice.begin(), 
                samplesDevice.end(), 
                hitIdsDevice.begin(), 
                functor);
        
        //for (int i = 0; i < samplesDevice.size(); i++)
        //{   
        //    thrust::transform(samplesDevice.begin() + i, samplesDevice.begin() + i + 1, hitIdsDevice.begin()+i, functor);
        //}

        hitIds = hitIdsDevice;
    }

    class IntitializeLeafFunctor
    {
        public:
        unsigned int *sortedObjectIDs;
        Node *leafNodes;
        const Triangle *triangles;

        __host__ 
        IntitializeLeafFunctor(unsigned int *sortedObjectIDs, 
                thrust::device_vector<Node> &leafNodes, 
                const thrust::device_vector<Triangle> &triangles)
        {
            this->sortedObjectIDs = sortedObjectIDs;
            this->leafNodes = leafNodes.data().get();
            this->triangles = triangles.data().get();
        }

        __host__ __device__
        void operator()(unsigned int idx)
        {
            initializeLeaf(idx, leafNodes, triangles, sortedObjectIDs);
        }
    };


    class ConstructNodeFunctor
    {
        unsigned int *sortedMortonCodes;
        unsigned int *sortedObjectIDs;
        unsigned int numObjects;
        Node *leafNodes;
        Node *internalNodes;

        public:
        __host__
        ConstructNodeFunctor(unsigned int *sortedMortonCodes, 
            unsigned int *sortedObjectIDs, 
            unsigned int numObjects, 
            thrust::device_vector<Node> &leafNodes,
            thrust::device_vector<Node> &internalNodes)
        {
            this->sortedMortonCodes = sortedMortonCodes;
            this->sortedObjectIDs = sortedObjectIDs;
            this->numObjects = numObjects;
            this->leafNodes = leafNodes.data().get();
            this->internalNodes = internalNodes.data().get();
        }

        __device__
        void operator()(unsigned int idx)
        {
            construct_node(idx, sortedMortonCodes, sortedObjectIDs, 
                                numObjects, leafNodes, internalNodes);
        }
    };
};


