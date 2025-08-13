#pragma once
#include <vector_types.h>
#include "utils/Color.h"

#include "LambertianBRDF.h"
#include "MirrorBRDF.h"
#include "LightBRDF.h"



enum MaterialType {
    lambertian = 0,
    mirror = 1,
    light = 2,
};

__host__ __device__ struct MaterialDescription {
    MaterialType type;

    union {
        LambertianBRDFConfig lambertian_brdf_config;
        MirrorBRDFConfig mirror_brdf_config;
        LightBRDFConfig mirror_brdf_light;
    };

    bool is_emissive = false;
    float3 emission;  // TODO not float3, spectrum

};


__forceinline__ __device__ void material_sample_wi(const MaterialDescription &md, const float u, const float2 u2, const float3 &wo, float3 &wi, HitRecord& hit, float &f, float &pdf, unsigned int lambda) {
    switch (md.type) {
    case lambertian:
        {
            const LambertianBRDF brdf({.reflectance = getByIndex(md.lambertian_brdf_config.color.getColor(hit.texture_coords), lambda)});
            brdf.sample_wi(u, u2, wo, wi, f, pdf, lambda);
            return;
        }
    case mirror:
        {
            const MirrorBRDF brdf({});
            brdf.sample_wi(u, u2, wo, wi, f, pdf, lambda);
            return;
        }
    case light:
        {
            const LightBRDF brdf({});
            brdf.sample_wi(u, u2, wo, wi, f, pdf, lambda);
            return;
        }
    }
}


__forceinline__ __device__ float material_f(const MaterialDescription &md, const float3 &wo, const float3 &wi, HitRecord& hit, unsigned int lambda) {
    switch (md.type) {
    case lambertian:
        {
            const LambertianBRDF brdf({.reflectance = getByIndex(md.lambertian_brdf_config.color.getColor(hit.texture_coords), lambda)});
            return brdf.f(wo, wi, lambda);
        }
    case mirror:
        {
            const MirrorBRDF brdf({});
            return brdf.f(wo, wi, lambda);
        }
    case light:
        {
            const LightBRDF brdf({});
            return brdf.f(wo, wi, lambda);
        }
    }
}

__forceinline__ __device__ float material_pdf(const MaterialDescription &md, const float3 &wo, const float3 &wi, HitRecord& hit, unsigned int lambda) {
    switch (md.type) {
    case lambertian:
        {
            const LambertianBRDF brdf({.reflectance = getByIndex(md.lambertian_brdf_config.color.getColor(hit.texture_coords), lambda)});
            return brdf.pdf(wo, wi, lambda);
        }
    case mirror:
        {
            const MirrorBRDF brdf({});
            return brdf.pdf(wo, wi, lambda);
        }
    case light:
        {
            const MirrorBRDF brdf({});
            return brdf.pdf(wo, wi, lambda);
        }
    }
}
