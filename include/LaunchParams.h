#pragma once

#include <cstdint>
#include <vector_types.h>
#include "materials/MaterialDescription.h"

struct LaunchParams
{
    unsigned int frameID { 0 };
    unsigned int subframe_idx { 0 };
    uchar4 *colorBuffer;
    float4 *accumBuffer;
    uint2 fbSize;

    struct {
        float focal;
        float lens_diameter;

        float2 film_size; // width, height
        float3 origin;
        float4 att;
    } camera;

    OptixTraversableHandle traversable;

    struct {
        unsigned int *indices_offsets;
        unsigned int *attribute_offsets;

        uint3 *indices_buffer;
        float3 *vertices_buffer;
        float3 *normals_buffer;
        float2 *uvs_buffer;
        // TODO tangents, bitangents etc.
    } scene_geometry;

    MaterialDescription *material_descriptions;

    // example of how to pass a texture
    cudaTextureObject_t horse_texture;
};
