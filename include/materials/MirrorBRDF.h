#pragma once
#include "utils/vec_math.h"
#include "utils/shading_utils.h"

__host__ __device__ struct MirrorBRDFConfig { };

__host__ __device__ struct MirrorBRDFParams { };


__host__ __device__ class MirrorBRDF {
    public:
        __host__ __device__ MirrorBRDF(const MirrorBRDFParams &params) { }

        __forceinline__ __host__ __device__ float f(const float3 &wo, const float3 &wi, unsigned int lambda) const {
            return 0.f;
        }

        __forceinline__ __host__ __device__ float pdf(const float3 &wo, const float3 &wi, unsigned int lambda) const {
            return 0.f;
        }

        __forceinline__ __host__ __device__ void sample_wi(const float u, const float2 u2, const float3 &wo, float3 &wi, float &f, float &pdf, unsigned int lambda) const {
            wi = make_float3(-wo.x, -wo.y, wo.z);
            pdf = 1.f;
            f = 1.f / wo.z;
        }
};
