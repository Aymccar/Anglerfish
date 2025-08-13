#pragma once
#include "utils/vec_math.h"
#include "utils/shading_utils.h"
#include "utils/Color.h"
#include "HitRecord.h"

__host__ __device__ struct LightBRDFConfig {
    Color color; 
};


__host__ __device__ struct LightBRDFParams {
};


__host__ __device__ class LightBRDF {
    public:
        __host__ __device__ LightBRDF(const LightBRDFParams &params)
            : params(params) {}

        __forceinline__ __host__ __device__ float f(const float3 &wo, const float3 &wi, unsigned int lambda) const {
            // TODO if not same_hemisphere(wo, wi) -> 0;
            return M_1_PIf;
        }

        __forceinline__ __host__ __device__ float pdf(const float3 &wo, const float3 &wi, unsigned int lambda) const {
            // TODO if not same_hemisphere(wo, wi) -> 0;
            return fabsf(wi.z) * M_1_PIf;
        }

        __forceinline__ __host__ __device__ void sample_wi(const float u, const float2 u2, const float3 &wo, float3 &wi, float &f, float &pdf, unsigned int lambda) const {
            const float2 point_in_disk = sample_disk(u2);
            wi.x = point_in_disk.x;
            wi.y = point_in_disk.y;
            wi.z = copysignf(sqrtf(fmaxf(0.f, 1.f - wi.x*wi.x - wi.y*wi.y)), wo.z);

            f = M_1_PIf;
            pdf = fabsf(wi.z) * M_1_PIf;
        }

        const LightBRDFParams &params;
};
