#pragma once

class MemoryArena
{
public:
    MemoryArena(size_t blockSize, size_t blockCount);
    ~MemoryArena();

    void* Allocate(size_t size);
    void Free(void* ptr);
};