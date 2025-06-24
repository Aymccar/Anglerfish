#pragma once
#include "vec_math.h"
#include "mat_math.h"

__forceinline__ __device__ float toSRGB(const float c) {
    const float invGamma = 1.0f / 2.4f;
    const float powed = powf( c, invGamma );
    return c < 0.0031308f ? 12.92f * c : 1.055f * powed - 0.055f;
}

__forceinline__ __device__ float3 toSRGB( const float3& c ) {
    const float  invGamma = 1.0f / 2.4f;
    const float3 powed    = make_float3( powf( c.x, invGamma ), powf( c.y, invGamma ), powf( c.z, invGamma ) );
    return make_float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f );
}


__forceinline__ __device__ unsigned char quantizeUnsigned8Bits( float x ) {
    x = clamp( x, 0.0f, 1.0f );
    enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
    return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}


__forceinline__ __device__ unsigned char make_color(const float c) {
    // first apply gamma, then convert to unsigned char
    const float srgb = toSRGB(clamp(c, 0.f, 1.f));
    return quantizeUnsigned8Bits(srgb);
}


__forceinline__ __device__ uchar4 make_color( const float3& c ) {
    // first apply gamma, then convert to unsigned char
    float3 srgb = toSRGB( clamp( c, 0.0f, 1.0f ) );
    return make_uchar4( quantizeUnsigned8Bits( srgb.x ), quantizeUnsigned8Bits( srgb.y ), quantizeUnsigned8Bits( srgb.z ), 255u );
}


static __forceinline__ __device__ Transform4 make_onb(const float3 &n) {
    Transform4 to_ret = identity();

    const float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    const float3 b1 = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    const float3 b2 = make_float3(b, sign + n.y * n.y * a, -n.y);

    to_ret.m[0].z = n.x;
    to_ret.m[1].z = n.y;
    to_ret.m[2].z = n.z;

    to_ret.m[0].x = b1.x;
    to_ret.m[1].x = b1.y;
    to_ret.m[2].x = b1.z;

    to_ret.m[0].y = b2.x;
    to_ret.m[1].y = b2.y;
    to_ret.m[2].y = b2.z;

    return to_ret;
}


static __forceinline__ __device__ float2 sample_disk(const float2 xi) {
    const float r = sqrtf(xi.x);
    const float phi = 2 * M_PIf * xi.y;
    return make_float2(r * cosf(phi), r * sinf(phi));
}
