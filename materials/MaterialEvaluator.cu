#include "Material.cu"
#include "BeckmannMaterial.cu"
#include "DielectricMaterial.cu"
#include "FresnelBlendMaterial.cu"
#include "LambertianMaterial.cu"
#include "MirrorMaterial.cu"
#include "LightMaterial.cu"
#include "PhongMaterial.cu"
#include "PlainMaterial.cu"


class MaterialEvaluator
{
    public:
    Material *material = nullptr;

    __host__ __device__
    MaterialEvaluator(Material *_material) : material(_material) {}

    __host__ __device__
    MaterialEvaluator(const MaterialEvaluator &other) : material(other.material) {}

    __host__ __device__
    MaterialEvaluator() {}

    __host__ __device__
    ~MaterialEvaluator() {}

    __host__ __device__
    Spectrum getEmittedRadiance(const SurfaceInteraction &interaction) const
    {
        // Cast material type and call specific function
        switch (material->type)
        {
            case MaterialType::BECKMANN:
                return ((BeckmannMaterial*)this)->vgetEmittedRadiance(interaction);
            case MaterialType::DIELECTRIC:
                return ((DielectricMaterial*)this)->vgetEmittedRadiance(interaction);
            case MaterialType::FRESNEL_BLEND:
                return ((FresnelBlendMaterial*)this)->vgetEmittedRadiance(interaction);
            case MaterialType::LAMBERTIAN:
                return ((LambertianMaterial*)this)->vgetEmittedRadiance(interaction);
            case MaterialType::MIRROR:
                return ((MirrorMaterial*)this)->vgetEmittedRadiance(interaction);
            case MaterialType::LIGHT:
                return ((LightMaterial*)this)->vgetEmittedRadiance(interaction);
            case MaterialType::PHONG:
                return ((PhongMaterial*)this)->vgetEmittedRadiance(interaction);
            case MaterialType::PLAIN:
                return ((PlainMaterial*)this)->vgetEmittedRadiance(interaction);
            default:
                printf("Error: Material type not recognized\n");
                exit(1);
        }
    }

    __host__ __device__
    Spectrum lightReflected(const Vector3f &cameraDirection, 
            const Vector3f &sampleDirection,
            Point2f localSamplePoint) const
    {
        // Cast material type and call specific function
        switch (material->type)
        {
            case MaterialType::BECKMANN:
                return ((BeckmannMaterial*)this)->vlightReflected(cameraDirection, sampleDirection, localSamplePoint);
            case MaterialType::DIELECTRIC:
                return ((DielectricMaterial*)this)->vlightReflected(cameraDirection, sampleDirection, localSamplePoint);
            case MaterialType::FRESNEL_BLEND:
                return ((FresnelBlendMaterial*)this)->vlightReflected(cameraDirection, sampleDirection, localSamplePoint);
            case MaterialType::LAMBERTIAN:
                return ((LambertianMaterial*)this)->vlightReflected(cameraDirection, sampleDirection, localSamplePoint);
            case MaterialType::MIRROR:
                return ((MirrorMaterial*)this)->vlightReflected(cameraDirection, sampleDirection, localSamplePoint);
            case MaterialType::LIGHT:
                return ((LightMaterial*)this)->vlightReflected(cameraDirection, sampleDirection, localSamplePoint);
            case MaterialType::PHONG:
                return ((PhongMaterial*)this)->vlightReflected(cameraDirection, sampleDirection, localSamplePoint);
            case MaterialType::PLAIN:
                return ((PlainMaterial*)this)->vlightReflected(cameraDirection, sampleDirection, localSamplePoint);
            default:
                printf("Error: Material type not recognized\n");
                exit(1);
        }
    }

    __host__ __device__
    Float pdf(const Vector3f &cameraDirection, const Vector3f &sampleDirection) const
    {
        // Cast material type and call specific function
        switch (material->type)
        {
            case MaterialType::BECKMANN:
                return ((BeckmannMaterial*)this)->vpdf(cameraDirection, sampleDirection);
            case MaterialType::DIELECTRIC:
                return ((DielectricMaterial*)this)->vpdf(cameraDirection, sampleDirection);
            case MaterialType::FRESNEL_BLEND:
                return ((FresnelBlendMaterial*)this)->vpdf(cameraDirection, sampleDirection);
            case MaterialType::LAMBERTIAN:
                return ((LambertianMaterial*)this)->vpdf(cameraDirection, sampleDirection);
            case MaterialType::MIRROR:
                return ((MirrorMaterial*)this)->vpdf(cameraDirection, sampleDirection);
            case MaterialType::LIGHT:
                return ((LightMaterial*)this)->vpdf(cameraDirection, sampleDirection);
            case MaterialType::PHONG:
                return ((PhongMaterial*)this)->vpdf(cameraDirection, sampleDirection);
            case MaterialType::PLAIN:
                return ((PlainMaterial*)this)->vpdf(cameraDirection, sampleDirection);
            default:
                printf("Error: Material type not recognized\n");
                exit(1);
        }
    }

    __host__ __device__
    Spectrum sampleDirection(
                const Vector3f &cameraDirection, 
                Sampler &sampler,
                Vector3f &sampleDirection,
                Point2f localSamplePoint) const
    {
        // Cast material type and call specific function
        switch (material->type)
        {
            case MaterialType::BECKMANN:
                return ((BeckmannMaterial*)this)->vsampleDirection(cameraDirection, sampler, sampleDirection, localSamplePoint);
            case MaterialType::DIELECTRIC:
                return ((DielectricMaterial*)this)->vsampleDirection(cameraDirection, sampler, sampleDirection, localSamplePoint);
            case MaterialType::FRESNEL_BLEND:
                return ((FresnelBlendMaterial*)this)->vsampleDirection(cameraDirection, sampler, sampleDirection, localSamplePoint);
            case MaterialType::LAMBERTIAN:
                return ((LambertianMaterial*)this)->vsampleDirection(cameraDirection, sampler, sampleDirection, localSamplePoint);
            case MaterialType::MIRROR:
                return ((MirrorMaterial*)this)->vsampleDirection(cameraDirection, sampler, sampleDirection, localSamplePoint);
            case MaterialType::LIGHT:
                return ((LightMaterial*)this)->vsampleDirection(cameraDirection, sampler, sampleDirection, localSamplePoint);
            case MaterialType::PHONG:
                return ((PhongMaterial*)this)->vsampleDirection(cameraDirection, sampler, sampleDirection, localSamplePoint);
            case MaterialType::PLAIN:
                return ((PlainMaterial*)this)->vsampleDirection(cameraDirection, sampler, sampleDirection, localSamplePoint);
            default:
                printf("Error: Material type not recognized\n");
                exit(1);
        }
    }

    __host__ __device__
    Spectrum getAmbientScattering() const
    {
        // Cast material type and call specific function
        switch (material->type)
        {
            case MaterialType::BECKMANN:
                return ((BeckmannMaterial*)this)->vgetAmbientScattering();
            case MaterialType::DIELECTRIC:
                return ((DielectricMaterial*)this)->vgetAmbientScattering();
            case MaterialType::FRESNEL_BLEND:
                return ((FresnelBlendMaterial*)this)->vgetAmbientScattering();
            case MaterialType::LAMBERTIAN:
                return ((LambertianMaterial*)this)->vgetAmbientScattering();
            case MaterialType::MIRROR:
                return ((MirrorMaterial*)this)->vgetAmbientScattering();
            case MaterialType::LIGHT:
                return ((LightMaterial*)this)->vgetAmbientScattering();
            case MaterialType::PHONG:
                return ((PhongMaterial*)this)->vgetAmbientScattering();
            case MaterialType::PLAIN:
                return ((PlainMaterial*)this)->vgetAmbientScattering();
            default:
                printf("Error: Material type not recognized\n");
                exit(1);
        }
    }

    __host__ __device__
    bool isScattering() const
    {
        // Cast material type and call specific function
        switch (material->type)
        {
            case MaterialType::BECKMANN:
                return ((BeckmannMaterial*)this)->visScattering();
            case MaterialType::DIELECTRIC:
                return ((DielectricMaterial*)this)->visScattering();
            case MaterialType::FRESNEL_BLEND:
                return ((FresnelBlendMaterial*)this)->visScattering();
            case MaterialType::LAMBERTIAN:
                return ((LambertianMaterial*)this)->visScattering();
            case MaterialType::MIRROR:
                return ((MirrorMaterial*)this)->visScattering();
            case MaterialType::LIGHT:
                return ((LightMaterial*)this)->visScattering();
            case MaterialType::PHONG:
                return ((PhongMaterial*)this)->visScattering();
            case MaterialType::PLAIN:
                return ((PlainMaterial*)this)->visScattering();
            default:
                printf("Error: Material type not recognized\n");
                exit(1);
        }
    }

    __host__ __device__
    bool isEmissive() const
    {
        // Cast material type and call specific function
        switch (material->type)
        {
            case MaterialType::BECKMANN:
                return ((BeckmannMaterial*)this)->visEmissive();
            case MaterialType::DIELECTRIC:
                return ((DielectricMaterial*)this)->visEmissive();
            case MaterialType::FRESNEL_BLEND:
                return ((FresnelBlendMaterial*)this)->visEmissive();
            case MaterialType::LAMBERTIAN:
                return ((LambertianMaterial*)this)->visEmissive();
            case MaterialType::MIRROR:
                return ((MirrorMaterial*)this)->visEmissive();
            case MaterialType::LIGHT:
                return ((LightMaterial*)this)->visEmissive();
            case MaterialType::PHONG:
                return ((PhongMaterial*)this)->visEmissive();
            case MaterialType::PLAIN:
                return ((PlainMaterial*)this)->visEmissive();
            default:
                printf("Error: Material type not recognized\n");
                exit(1);
        }
    }

    __host__ __device__
    bool isSpecular() const
    {
        // Cast material type and call specific function
        switch (material->type)
        {
            case MaterialType::BECKMANN:
                return ((BeckmannMaterial*)this)->visSpecular();
            case MaterialType::DIELECTRIC:
                return ((DielectricMaterial*)this)->visSpecular();
            case MaterialType::FRESNEL_BLEND:
                return ((FresnelBlendMaterial*)this)->visSpecular();
            case MaterialType::LAMBERTIAN:
                return ((LambertianMaterial*)this)->visSpecular();
            case MaterialType::MIRROR:
                return ((MirrorMaterial*)this)->visSpecular();
            case MaterialType::LIGHT:
                return ((LightMaterial*)this)->visSpecular();
            case MaterialType::PHONG:
                return ((PhongMaterial*)this)->visSpecular();
            case MaterialType::PLAIN:
                return ((PlainMaterial*)this)->visSpecular();
            default:
                printf("Error: Material type not recognized\n");
                exit(1);
        }
    }

    __host__ __device__
    bool isTransparent() const
    {
        // Cast material type and call specific function
        switch (material->type)
        {
            case MaterialType::BECKMANN:
                return ((BeckmannMaterial*)this)->visTransparent();
            case MaterialType::DIELECTRIC:
                return ((DielectricMaterial*)this)->visTransparent();
            case MaterialType::FRESNEL_BLEND:
                return ((FresnelBlendMaterial*)this)->visTransparent();
            case MaterialType::LAMBERTIAN:
                return ((LambertianMaterial*)this)->visTransparent();
            case MaterialType::MIRROR:
                return ((MirrorMaterial*)this)->visTransparent();
            case MaterialType::LIGHT:
                return ((LightMaterial*)this)->visTransparent();
            case MaterialType::PHONG:
                return ((PhongMaterial*)this)->visTransparent();
            case MaterialType::PLAIN:
                return ((PlainMaterial*)this)->visTransparent();
            default:
                printf("Error: Material type not recognized\n");
                exit(1);
        }
    }

    __host__ __device__
    bool isSubsurfaceScattering() const
    {
        // Cast material type and call specific function
        switch (material->type)
        {
            case MaterialType::BECKMANN:
                return ((BeckmannMaterial*)this)->visSubsurfaceScattering();
            case MaterialType::DIELECTRIC:
                return ((DielectricMaterial*)this)->visSubsurfaceScattering();
            case MaterialType::FRESNEL_BLEND:
                return ((FresnelBlendMaterial*)this)->visSubsurfaceScattering();
            case MaterialType::LAMBERTIAN:
                return ((LambertianMaterial*)this)->visSubsurfaceScattering();
            case MaterialType::MIRROR:
                return ((MirrorMaterial*)this)->visSubsurfaceScattering();
            case MaterialType::LIGHT:
                return ((LightMaterial*)this)->visSubsurfaceScattering();
            case MaterialType::PHONG:
                return ((PhongMaterial*)this)->visSubsurfaceScattering();
            case MaterialType::PLAIN:
                return ((PlainMaterial*)this)->visSubsurfaceScattering();
            default:
                printf("Error: Material type not recognized\n");
                exit(1);
        }
    }

    __host__ __device__
    MaterialType getType() const
    {
        return material->type;
    }
};