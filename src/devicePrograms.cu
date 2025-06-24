#include <optix_device.h>
#include "utils/vec_math.h"
#include "utils/mat_math.h"
#include "utils/random.h"
#include "utils/shading_utils.h"
#include "HitRecord.h"
#include "materials/MaterialDescription.h"

#include "LaunchParams.h"

extern "C" __constant__ LaunchParams optixLaunchParams;


static __forceinline__ __device__ void pack_pointer(const void *ptr, unsigned int &hi, unsigned int &lo) {
    lo = (unsigned long long)ptr & 0xffffffffll;
    hi = (unsigned long long)ptr >> 32;
}


static __forceinline__ __device__ void *unpack_pointer(const unsigned int hi, const unsigned int lo) {
    return (void *) ((unsigned long long)hi << 32 | lo);
}


template <typename T>
static __forceinline__ __device__ T* get_ray_data_pointer() {
    unsigned int hi = optixGetPayload_0();
    unsigned int lo = optixGetPayload_1();

    return (T *)unpack_pointer(hi, lo);
}


// from https://link.springer.com/content/pdf/10.1007/978-1-4842-4427-2_6.pdf
// Normal points outward for rays exiting the surface, else is flipped.
static __forceinline__ __device__ float3 offset_ray(const float3 p, const float3 n) {
    constexpr float origin = 1.0f / 32.0f;
    constexpr float float_scale = 1.0f / 65536.0f;
    constexpr float int_scale = 256.0f;

    const int3 of_i = make_int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    const float3 p_i = make_float3(
            __int_as_float(__float_as_int(p.x)+((p.x < 0) ? -of_i.x : of_i.x)),
            __int_as_float(__float_as_int(p.y)+((p.y < 0) ? -of_i.y : of_i.y)),
            __int_as_float(__float_as_int(p.z)+((p.z < 0) ? -of_i.z : of_i.z)));

    return make_float3(
            fabsf(p.x) < origin ? p.x+ float_scale*n.x : p_i.x,
            fabsf(p.y) < origin ? p.y+ float_scale*n.y : p_i.y,
            fabsf(p.z) < origin ? p.z+ float_scale*n.z : p_i.z);
}


extern "C" __global__ void __closesthit__radiance() {
    const unsigned int instance_id = optixGetInstanceId();
    const unsigned int primitive_id = optixGetPrimitiveIndex();

    const float2 bary = optixGetTriangleBarycentrics();

    uint3 attribute_indices = optixLaunchParams.scene_geometry.indices_buffer[optixLaunchParams.scene_geometry.indices_offsets[instance_id] + primitive_id];
    attribute_indices += optixLaunchParams.scene_geometry.attribute_offsets[instance_id];

    // compute hit world position
    const float3 vertex_0 = optixLaunchParams.scene_geometry.vertices_buffer[attribute_indices.x];
    const float3 vertex_1 = optixLaunchParams.scene_geometry.vertices_buffer[attribute_indices.y];
    const float3 vertex_2 = optixLaunchParams.scene_geometry.vertices_buffer[attribute_indices.z];
    float3 hitPoint = vertex_1 * bary.x + vertex_2 * bary.y + vertex_0 * (1.f - bary.x - bary.y);
    hitPoint = optixTransformPointFromObjectToWorldSpace(hitPoint);

    // compute normals
    const float3 normal_0 = optixLaunchParams.scene_geometry.normals_buffer[attribute_indices.x];
    const float3 normal_1 = optixLaunchParams.scene_geometry.normals_buffer[attribute_indices.y];
    const float3 normal_2 = optixLaunchParams.scene_geometry.normals_buffer[attribute_indices.z];

    // no need to normalize the object space normal, we do it after transforming it to world space
    float3 shading_normal = normal_1 * bary.x + normal_2 * bary.y + normal_0 * (1.f - bary.x - bary.y);
    shading_normal = normalize(optixTransformNormalFromObjectToWorldSpace(shading_normal));

    const float3 edge1 = vertex_1 - vertex_0;
    const float3 edge2 = vertex_2 - vertex_0;
    const float3 geometric_normal = normalize(optixTransformNormalFromObjectToWorldSpace(cross(edge1, edge2)));

    // Keller et al. [2017] ( https://arxiv.org/abs/1705.01263 Appendix A.3 ) workaround for artifacts
    const float3 ray_direction = -optixGetWorldRayDirection();
    const float3 r = reflect(ray_direction, shading_normal);
    if (dot(r, geometric_normal) < 0.f) {
        //shading_normal = normalize(-ray_direction + normalize(r - geometric_normal * dot(geometric_normal, r)));
        shading_normal = geometric_normal;
    }


    // compute texture coordinates
    const float2 uv_0 = optixLaunchParams.scene_geometry.uvs_buffer[attribute_indices.x];
    const float2 uv_1 = optixLaunchParams.scene_geometry.uvs_buffer[attribute_indices.y];
    const float2 uv_2 = optixLaunchParams.scene_geometry.uvs_buffer[attribute_indices.z];
    float2 uv = uv_1 * bary.x + uv_2 * bary.y + uv_0 * (1.f - bary.x - bary.y);

    HitRecord *hit = get_ray_data_pointer<HitRecord>();
    hit->instance_id = instance_id;
    hit->primitive_id = primitive_id;
    hit->shading_normal = shading_normal;
    hit->geometric_normal = geometric_normal;
    hit->intersection_point = hitPoint;
    hit->texture_coords = uv;
}

extern "C" __global__ void __anyhit__radiance() {}

extern "C" __global__ void __miss__radiance() {
    HitRecord *hit = get_ray_data_pointer<HitRecord>();
    hit->instance_id = NO_HIT;
}


static __forceinline__ __device__ void create_pinhole_ray(unsigned int &seed, float3 &ray_origin, float3 &ray_direction) {
    ray_origin = optixLaunchParams.camera.origin;

    const float2 subpixel_offset = make_float2(rnd(seed), rnd(seed));

    const uint3 idx = optixGetLaunchIndex();
    const uint2 resolution = optixLaunchParams.fbSize;

    const float4 att = optixLaunchParams.camera.att;
    const float qr = att.w;
    const float qi = att.x;
    const float qj = att.y;
    const float qk = att.z;

    const float r00 = 1 - 2 *(qj * qj + qk * qk);
    const float r01 = 2 * (qi * qj - qk * qr);
    const float r02 = 2 * (qi * qk + qj * qr);
     
    const float r10 = 2 * (qi * qj + qk * qr);
    const float r11 = 1 - 2 * (qi * qi + qk * qk);
    const float r12 = 2 * (qj * qk - qi * qr);
     
    const float r20 = 2 * (qi * qk - qj * qr);
    const float r21 = 2 * (qj * qk + qi * qr);
    const float r22 = 1 - 2 * (qi * qi + qj * qj);

    const float3 cam_dir =   {r00, r10, r20};
    const float3 cam_right = {r01, r11, r21};
    const float3 cam_up =    {r02, r12, r22};

    ray_direction = normalize(
        (((float)idx.x + subpixel_offset.x) / resolution.x - 0.5f) * optixLaunchParams.camera.film_size.x * cam_right
        - (((float)idx.y + subpixel_offset.y) / resolution.y - 0.5f) * optixLaunchParams.camera.film_size.y * cam_up
        + optixLaunchParams.camera.focal * cam_dir
    );
}

// a few uchar4 functions
SUTIL_INLINE SUTIL_HOSTDEVICE void setByIndex(uchar4& v, const unsigned int i, unsigned char x)
{
  ((unsigned char*)(&v))[i] = x;
}

extern "C" __global__ void __raygen__renderFrame() {
    const unsigned int subframe_idx = optixLaunchParams.subframe_idx;
    const uint3 idx = optixGetLaunchIndex();
    const uint2 resolution = optixLaunchParams.fbSize;

    const unsigned int lambda = idx.z;

    unsigned int seed = tea<4>(idx.y * resolution.x + idx.x, subframe_idx);

    float3 ray_origin, ray_direction;
    create_pinhole_ray(seed, ray_origin, ray_direction);

    HitRecord hit;

    uint32_t p0, p1;
    pack_pointer(&hit, p0, p1);

    const unsigned int MAX_BOUNCE = 50;
    float L = 0.f;
    float beta = 1.f;

    float2 xi;  // 2 uniform random numbers;

    // main rendering loop
    unsigned int num_bounces;
    for (num_bounces = 0; num_bounces < MAX_BOUNCE; num_bounces++) {
        float q = 0.1f;
        if (num_bounces < 10)
            q = 0.f;

        const float russian_roulette_u = rnd(seed);
        if (russian_roulette_u < q)
            break;

        // get next intersection
        optixTrace(
                optixLaunchParams.traversable,
                ray_origin,
                ray_direction,
                0.f,                            // tmin
                1e16f,                          // tmax
                0,                              // rayTime
                OptixVisibilityMask(255),       // visibilityMask
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // rayFlags
                0,                              // SBToffset
                1,                              // SBTstride
                0,                              // missSBTIndex
                p0, p1                          // payload
                );

        if (hit.instance_id == NO_HIT) {
            const float3 sky_bottom_color = make_float3(136.f / 255, 171.f / 255, 227.f / 255);
            const float3 sky_top_color = make_float3(1.f);
            const float3 sky_light = lerp(sky_bottom_color, sky_top_color, -max(0.f, ray_direction.z));
            L += getByIndex(beta * sky_light, lambda);
            break;
        }

        const MaterialDescription &mat = optixLaunchParams.material_descriptions[hit.instance_id];

        if (mat.is_emissive) {
            L += getByIndex(beta * mat.emission, lambda);
            break; // TODO later on, don't!
        }

        const Transform4 onb = make_onb(hit.shading_normal);
        const float3 wo = make_float3(transpose(onb) * make_float4(-ray_direction, 0));
        float3 wi;  // the next ray direction (in shading space)
        float f;   // the brdf value for the current channel
        float pdf;  // pdf

        float u = rnd(seed);
        xi.x = rnd(seed);
        xi.y = rnd(seed);

        material_sample_wi(mat, u, xi, wo, wi, hit, f, pdf, lambda);

        const float cosTheta = fabsf(wi.z);
        beta *= f * cosTheta / pdf / (1.f - q);
        ray_origin = offset_ray(hit.intersection_point, copysignf(1.f, wi.z) * hit.geometric_normal);
        ray_direction = normalize(make_float3(onb * make_float4(wi, 0)));
    }

    const uint32_t fbIndex = idx.x + (resolution.y - idx.y) * optixLaunchParams.fbSize.x;
    if (subframe_idx == 0) {
        if (lambda == 0)
            optixLaunchParams.accumBuffer[fbIndex].w = 0.f;

        setByIndex(optixLaunchParams.accumBuffer[fbIndex], lambda, L);
    } else {
        const float previous = getByIndex(optixLaunchParams.accumBuffer[fbIndex], lambda);
        setByIndex(optixLaunchParams.accumBuffer[fbIndex], lambda, previous + L);
    }

    // and write to the frame buffer
    if (lambda == 0)
        optixLaunchParams.colorBuffer[fbIndex].w = 255;

    ((unsigned char*)(&optixLaunchParams.colorBuffer[fbIndex]))[lambda] = make_color(getByIndex(optixLaunchParams.accumBuffer[fbIndex], lambda) / (subframe_idx + 1));
}
