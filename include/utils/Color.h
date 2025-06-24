#pragma once 
#include "utils/Texture.h"

#include <vector_types.h>
#include <cuda_runtime.h>


__host__ __device__ struct Color {
    
    __forceinline__ __device__ float3 getColor(float2 uv) const;
    __forceinline__ __device__ float3 getColor() const;

    __forceinline__ __host__ void setTexture(Texture& texture_);
    __forceinline__ __host__ void setColor(float3 rgb_);

    private:

    bool hasTexture;
    cudaTextureObject_t texture;
    float3 rgb;
};

__forceinline__ __host__ void Color::setTexture(Texture& texture_){
    hasTexture = true;
    texture = texture_.getTextureObject();
}

__forceinline__ __host__ void Color::setColor(float3 rgb_){
    hasTexture = false;
    rgb = rgb_;
}

#if defined(__CUDACC__)
__forceinline__ __device__ float3 Color::getColor(float2 uv) const {
    if(hasTexture) {
        float4 r = tex2D<float4>(texture, uv.x, uv.y);
        //return XYZColor::fromRGB(make_float3(r.x, r.y, r.z)).tofloat3();
        return {r.x, r.y, r.z};
    }
    else return rgb;
}

__forceinline__ __device__ float3 Color::getColor() const {
    return rgb;
}
#endif
