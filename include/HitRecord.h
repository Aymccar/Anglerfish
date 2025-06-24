#pragma once
#include <vector_types.h>

constexpr unsigned int NO_HIT = ~0u;

struct HitRecord {
    unsigned int instance_id;
    unsigned int primitive_id;

    float3 shading_normal;
    float3 geometric_normal;
    float3 intersection_point;
    float2 texture_coords;

    // TODO tangents & bitangents
};
