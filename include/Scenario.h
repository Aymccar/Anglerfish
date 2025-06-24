#pragma once
#include <string>
#include <vector>
#include <vector_types.h>

struct Mesh {
    std::string file_name;

    float scale;
    std::string material;
    std::string look;

    float3 origin; 
    float3 world_rotation;
    float3 world_translation;
};

struct Look {
    std::string name;
    float3 rgb;
    float roughness;
    float metalness; 
    std::string texture;
};

struct Material {
    std::string name;
    std::string type;
};


struct Scenario {
    std::vector<Look> Looks;
    std::vector<Material> Materials;
    std::vector<Mesh> Static_meshes;
    std::vector<Mesh> Moving_meshes;
};
