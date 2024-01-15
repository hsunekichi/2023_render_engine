#pragma once

#include "GPUVector.cu"

template <typename T>
class GPUStack
{
    private:
    GPUVector<T> stack;
    int top;

    public:
    __host__ __device__
    GPUStack (int size=0)
    : stack(size)
    {
        top = -1;
    }

    __host__ __device__
    void push(const T &item)
    {
        stack.push_back(item);
        top++;
    }

    __host__ __device__
    T pop()
    {
        T item = stack[top];
        stack.pop_back();
        top--;
        return item;
    }

    __host__ __device__
    T peek()
    {
        return stack[top];
    }

    __host__ __device__
    bool isEmpty()
    {
        return top == -1;
    }

    __host__ __device__
    int size()
    {
        return top + 1;
    }

    __host__ __device__
    void clear()
    {
        stack.clear();
        top = -1;
    }
};