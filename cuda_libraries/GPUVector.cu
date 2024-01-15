#pragma once

#include <assert.h>
#include <vector>
#include <iostream>
#include <mutex>
#include <functional>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


template <typename T>
class GPUVectorIterator
{
private:
    T* ptr;

public:
    __host__ __device__
    GPUVectorIterator(T* ptr) 
    : ptr(ptr) {}

    __host__ __device__
    T& operator*() const { 
        return *ptr; 
    }

    __host__ __device__
    T* operator->() const { 
        return ptr;
    }

    __host__ __device__
    GPUVectorIterator& operator++() { 
        ptr++; 
        return *this; 
    }

    __host__ __device__
    GPUVectorIterator operator++(int) { 
        GPUVectorIterator temp = *this; ptr++; return temp; 
    }

    __host__ __device__
    GPUVectorIterator& operator--() { 
        ptr--; 
        return *this; 
    }

    __host__ __device__
    GPUVectorIterator operator--(int) { 
        GPUVectorIterator temp = *this; 
        ptr--; 
        return temp; 
    }

    __host__ __device__
    bool operator==(const GPUVectorIterator& other) const { 
        return ptr == other.ptr; 
    }

    __host__ __device__
    bool operator!=(const GPUVectorIterator& other) const { 
        return ptr != other.ptr; 
    }
};


// Parallel for kernel
template <typename T, typename F>
__global__
void parallelForKernel(T* data, size_t start, size_t end, F f)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < end - start)
    {
        f(data[index + start]);
    }
}

template <typename T>
class GPUVector {
private:
    T* data;
    size_t max_capacity;
    std::mutex mutex;
    int valid_elements;

public:
    // Constructor
    __host__    
    GPUVector(size_t size = 0) 
    {
        max_capacity = size;
        valid_elements = 0;
        data = nullptr;

        if (max_capacity > 0)
        {
            auto err = cudaMallocManaged(&data, max_capacity * sizeof(T));

            if (err != cudaSuccess)
            {
                printf("ERROR: GPUVector allocation failed\n");
                printf("ERROR: %s\n", cudaGetErrorString(err));
                assert(false);
            }
        }
    }
    
    __host__
    GPUVector(size_t size, T value, bool beginFull=true) 
    {
        max_capacity = size;

        if (beginFull)
            valid_elements = size;
        else
            valid_elements = 0;
        
        data = nullptr;

        if (max_capacity > 0)
            cudaMallocManaged(&data, max_capacity * sizeof(T));

        for (size_t i = 0; i < max_capacity; i++) {
            data[i] = value;
        }
    }

    // Destructor
    __host__ 
    ~GPUVector() 
    {
        cudaFree(data);
    }

    // Copy constructor
    __host__ 
    GPUVector(const GPUVector& other) 
    : max_capacity(other.max_capacity), 
        valid_elements(other.valid_elements) 
    {
        cudaMallocManaged(&data, max_capacity * sizeof(T));

        for (size_t i = 0; i < valid_elements; i++) {
            data[i] = other.data[i];
        }
    }

    // Move constructor
    __host__ __device__
    GPUVector(GPUVector&& other) 
    : data(other.data), 
        max_capacity(other.max_capacity), 
        valid_elements(other.valid_elements) 
    {
        other.data = nullptr;
        other.max_capacity = 0;
        other.valid_elements = 0;
    }

    // Initializer list constructor
    __host__ __device__
    GPUVector(std::initializer_list<T> list) 
    : max_capacity(list.size()), 
        valid_elements(list.size()) 
    {
        cudaMallocManaged(&data, max_capacity * sizeof(T));

        size_t i = 0;
        for (auto& element : list) {
            data[i] = element;
            i++;
        }
    }

    // Constructor from a std::vector
    __host__
    GPUVector(std::vector<T> &other)
    {
        max_capacity = other.size();
        valid_elements = other.size();
        cudaMallocManaged(&data, max_capacity * sizeof(T));

        if (valid_elements > 0)
        {
            T *host_data = other.data();
            cudaMemcpy(data, host_data, valid_elements * sizeof(T), cudaMemcpyHostToDevice);
        }
    }

    // Constructor from a thrust::host_vector
    __host__
    GPUVector(thrust::host_vector<T> &other)
    {
        max_capacity = other.size();
        valid_elements = other.size();
        cudaMallocManaged(&data, max_capacity * sizeof(T));

        if (valid_elements > 0)
        {
            T *host_data = other.data();
            cudaMemcpy(data, host_data, valid_elements * sizeof(T), cudaMemcpyHostToDevice);
        }
    }

    // Constructor from a thrust::device_vector
    __host__
    GPUVector(thrust::device_vector<T> &other)
    {
        max_capacity = other.size();
        valid_elements = other.size();
        cudaMallocManaged(&data, max_capacity * sizeof(T));

        if (valid_elements > 0)
        {
            T *host_data = other.data();
            cudaMemcpy(data, host_data, valid_elements * sizeof(T), cudaMemcpyHostToDevice);
        }
    }

    // Operator new
    __host__ __device__
    void* operator new(size_t size) 
    {
        void* ptr;
        cudaMallocManaged(&ptr, size);
        return ptr;
    }

    __host__ __device__
    void* operator new[](size_t size) 
    {
        void* ptr;
        cudaMallocManaged(&ptr, size);
        return ptr;
    }

    // Operator delete
    __host__ __device__
    void operator delete(void* ptr) 
    {
        cudaFree(ptr);
    }

    __host__ __device__
    void operator delete[](void* ptr) 
    {
        cudaFree(ptr);
    }

    __host__ __device__
    inline int increment_valid_elements()
    {
        # ifdef __CUDA_ARCH__
            return atomicAdd(&valid_elements, 1);
        # else
            std::lock_guard<std::mutex> lock(mutex);
            return valid_elements++;
        # endif
    }

    __host__ __device__
    inline int decrement_valid_elements()
    {
        # ifdef __CUDA_ARCH__
            return atomicSub(&valid_elements, 1);
        # else
            std::lock_guard<std::mutex> lock(mutex);
            return valid_elements--;
        # endif
    }

    // Construct an std::vector with the elements of the vector
    __host__
    std::vector<T> to_std_vector()
    {
        std::vector<T> result(valid_elements);

        if (valid_elements > 0)
        {
            T *host_data = result.data();
            cudaMemcpy(host_data, data, valid_elements * sizeof(T), cudaMemcpyDeviceToHost);
        }

        return result;
    }

    __host__
    thrust::host_vector<T> to_thrust_host_vector()
    {
        thrust::host_vector<T> result(valid_elements);

        if (valid_elements > 0)
        {
            T *host_data = result.data();
            cudaMemcpy(host_data, data, valid_elements * sizeof(T), cudaMemcpyDeviceToHost);
        }

        return result;
    }

    __host__
    thrust::device_vector<T> to_thrust_device_vector()
    {
        thrust::device_vector<T> result(valid_elements);

        if (valid_elements > 0)
        {
            T *host_data = result.data();
            cudaMemcpy(host_data, data, valid_elements * sizeof(T), cudaMemcpyDeviceToHost);
        }

        return result;
    }


    // Copy assignment operator
    __host__ __device__
    GPUVector& operator=(const GPUVector& other) 
    {
        if (this != &other) {
            assign(other);
        }

        return *this;
    }

    // Move assignment operator
    __host__ __device__
    GPUVector& operator=(GPUVector&& other) 
    {
        if (this != &other) 
        {
            clear_and_free();

            data = other.data;
            max_capacity = other.max_capacity;
            valid_elements = other.valid_elements;

            other.data = nullptr;
            other.max_capacity = 0;
            other.valid_elements = 0;
        }

        return *this;
    }

    // Initializer list operator
    __host__ __device__
    GPUVector& operator=(std::initializer_list<T> list) 
    {
        clear_and_free();

        max_capacity = list.size();
        valid_elements = list.size();
        cudaMallocManaged(&data, max_capacity * sizeof(T));

        size_t i = 0;
        for (auto& element : list) {
            data[i] = element;
            i++;
        }

        return *this;
    }



    // Get the size of the vector
    __host__ __device__
    size_t size() const 
    {
        return valid_elements;
    }

    __host__ __device__
    size_t capacity() const 
    {
        return max_capacity;
    }

    __host__ __device__
    bool empty() const 
    {
        return valid_elements == 0;
    }

    __host__
    void reserve(size_t new_size)
    {
        if (new_size > max_capacity) {
            resize(new_size);
        }
    }

    __host__
    void resize(size_t new_size) 
    {
        // If not all elements fit, truncates them
        if (new_size < valid_elements) {      
            valid_elements = new_size;
        }   
        // If not, only the valid elements will be copied

        max_capacity = new_size;    // New maximum capacity

        T* new_data;
        cudaMallocManaged(&data, max_capacity * sizeof(T));

        for (size_t i = 0; i < valid_elements; i++) { // Copies the elements from the old vector to the new one
            new_data[i] = data[i];
        }

        cudaFree(data);    // Frees the old memory
        data = new_data;

        valid_elements = new_size;
    }

    template <typename F>
    __host__ __device__
    void sequentialForEach(size_t start, size_t end, F f)
    {
        for (size_t i = start; i < end; i++)
        {
            f(data[i]);
        }
    } 

    template <typename F>
    __host__ __device__
    void sequentialForEach(F f)
    {
        for (size_t i = 0; i < valid_elements; i++)
        {
            f(data[i]);
        }
    }


    template <typename F>
    __host__
    void GPUparallelForEach(size_t start, size_t end, F f)
    {
        size_t num_threads = end - start;
        size_t num_blocks = (num_threads + 1023) / 1024;

        parallelForKernel<<<num_blocks, 1024>>>(data, start, end, f);
    }

    template <typename F>
    __host__
    void GPUparallelForEach(F f)
    {
        size_t num_threads = valid_elements;
        size_t num_blocks = (num_threads + 1023) / 1024;

        parallelForKernel<<<num_blocks, 1024>>>(data, 0, valid_elements, f);
    }

    __host__
    void shrink_to_fit() 
    {
        resize(valid_elements);
    }


    // Access elements by index
    __host__ __device__
    T& operator[](size_t index) 
    {
        return data[index];
    }

    __host__ __device__
    const T& operator[](size_t index) const 
    {            
        return data[index];
    }

    __host__ __device__
    const T& at(size_t index) const 
    {
        assert(index < valid_elements);

        return data[index];
    }

    __host__ __device__
    T& at(size_t index) 
    {
        assert(index < valid_elements);

        return data[index];
    }

    // Get the first element
    __host__ __device__
    T& front() 
    {
        assert(valid_elements > 0);

        return data[0];
    }

    __host__ __device__
    const T& front() const 
    {
        assert(valid_elements > 0);

        return data[0];
    }

    // Get the last element
    __host__ __device__
    T& back() 
    {
        assert(valid_elements > 0);

        return data[valid_elements - 1];
    }

    __host__ __device__
    const T& back() const 
    {
        assert(valid_elements > 0);

        return data[valid_elements - 1];
    }

    // Push an element to the back of the vector
    // This is an atomic operation
    __host__ __device__
    void push_back(T element) 
    {
        /*
        // Check if there is enough capacity
        if (valid_elements == max_capacity) 
        {
            size_t new_capacity = max_capacity;

            if (new_capacity == 0) 
                new_capacity = 1;
            else 
                new_capacity *= 2;
            
            resize(new_capacity);
        }
        */
        
        if (valid_elements == max_capacity)
        {
            printf("ERROR: GPUVector overflow\n");
            assert(false);
        }

        int index = increment_valid_elements();
        data[index] = element;
    }

    // This is an atomic operation
    __host__ __device__
    T pop_back()
    {
        assert(valid_elements > 0);

        int index = decrement_valid_elements();
        return data[index];
    }

    __host__ __device__
    void assign (GPUVector &other)
    {
        clear_and_free();
        max_capacity = other.max_capacity;
        valid_elements = other.valid_elements;
        cudaMallocManaged(&data, max_capacity * sizeof(T));

        for (size_t i = 0; i < valid_elements; i++) {
            data[i] = other.data[i];
        }
    }

    __host__ __device__
    void clear()
    {
        valid_elements = 0;
    }

    __host__ __device__
    void swap(GPUVector &other)
    {
        T* temp_data = data;
        size_t temp_max_capacity = max_capacity;
        size_t temp_valid_elements = valid_elements;

        data = other.data;
        max_capacity = other.max_capacity;
        valid_elements = other.valid_elements;

        other.data = temp_data;
        other.max_capacity = temp_max_capacity;
        other.valid_elements = temp_valid_elements;
    }

    __host__ __device__
    void clear_and_free()
    {
        cudaFree(data);
        max_capacity = 0;
        valid_elements = 0;
    }


    __host__ __device__
    GPUVectorIterator<T> begin() 
    {
        return GPUVectorIterator<T>(data);
    }

    __host__ __device__
    GPUVectorIterator<T> end() 
    {
        return GPUVectorIterator<T>(data + valid_elements);
    }

    __host__ __device__
    GPUVectorIterator<const T> begin() const 
    {
        return GPUVectorIterator<const T>(data);
    }

    __host__ __device__
    GPUVectorIterator<const T> end() const 
    {
        return GPUVectorIterator<const T>(data + valid_elements);
    }

    __host__ __device__
    GPUVectorIterator<const T> cbegin() const 
    {
        return GPUVectorIterator<const T>(data);
    }

    __host__ __device__
    GPUVectorIterator<const T> cend() const 
    {
        return GPUVectorIterator<const T>(data + valid_elements);
    }

    __host__ __device__
    GPUVectorIterator<const T> rbegin() const 
    {
        return GPUVectorIterator<const T>(data + valid_elements - 1);
    }

    __host__ __device__
    GPUVectorIterator<const T> rend() const 
    {
        return GPUVectorIterator<const T>(data - 1);
    }

    __host__ __device__
    GPUVectorIterator<const T> crbegin() const 
    {
        return GPUVectorIterator<const T>(data + valid_elements - 1);
    }

    __host__ __device__
    GPUVectorIterator<const T> crend() const 
    {
        return GPUVectorIterator<const T>(data - 1);
    }

    // cout
    __host__
    friend std::ostream& operator<<(std::ostream& os, const GPUVector& vector) 
    {
        os << "[";
        for (size_t i = 0; i < vector.valid_elements; i++) {
            os << vector.data[i];
            if (i != vector.valid_elements - 1) {
                os << ", ";
            }
        }
        os << "]";

        return os;
    }
};