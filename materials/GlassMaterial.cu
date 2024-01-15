#pragma once


#include "DielectricMaterial.cu"

#include "../transformations/SurfaceInteraction.cu"

class GlassMaterial : public DielectricMaterial
{
public:
    
    GlassMaterial(const Spectrum& color) 
    : DielectricMaterial(color, 1.5f)
    {}
};