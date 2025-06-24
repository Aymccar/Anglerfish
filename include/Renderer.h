// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <vector_types.h>
#include <vector_functions.h>
#include <string>

// our own classes, partly shared between host and device
#include "utils/CUDABuffer.h"
#include "utils/vec_math.h"
#include "utils/mat_math.h"
#include "utils/Texture.h"
#include "LaunchParams.h"

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>

class Renderer {
    public:
        // performs all setup, including initializing optix, creates module, pipeline, programs, SBT etc.
        Renderer();

        void loadScene(std::string filename);

        // render one sample/
        void render();

        // resize frame buffer to given resolution
        void resize(const uint2 &newSize);

        // download the rendered color buffer
        void downloadPixels(uint32_t h_pixels[]);

        // updates the transforms and/or materials of the scene
        void updateScene();

        // publish on ros ropic
        void publish(sensor_msgs::msg::Image msg);

    private:
        // helper function that initializes optix and checks for errors
        void initOptix();

        // creates and configures a optix device context
        void createContext();

        // creates the module that contains all the programs we are going to use
        void createModule();

        // does all setup for the raygen program(s) we are going to use
        void createRaygenPrograms();

        // does all setup for the miss program(s) we are going to use
        void createMissPrograms();

        // does all setup for the hitgroup program(s) we are going to use
        void createHitgroupPrograms();

        // assembles the full pipeline of all programs
        void createPipeline();

        // builds the acceleration structures
        void buildAccel();

        // packs and uploads the vertex, index and attributes buffers to the
        // GPU and sets them up in the LaunchParams
        void populateSceneBuffers();

        // uploads the material descriptions to the GPU
        void uploadMaterials();

        // constructs the handles for the matrix motion transforms
        void buildMotionTransforms();

        // builds/updates the instance acceleration structure over the motion transforms
        void buildInstanceAccelerationStructure(bool shouldUpdate);

        // updates the ASs for the new frame
        void updateAccel();

        // constructs the shader binding table
        void buildSBT();

    private:
        bool is_accel_dirty = false;
        bool is_scene_initialized = false;

        // scene data
        std::vector<Transform4> transforms;
        Texture horse_texture;

        // s/all/geometry/g  (geometry_indices, geometry_vertices etc.)
        std::vector<std::vector<float3>> all_vertices;
        std::vector<std::vector<uint3>> all_indices;
        std::vector<std::vector<float3>> all_normals;
        std::vector<std::vector<float2>> all_uvs;
        std::vector<unsigned int> indices_offsets;
        std::vector<unsigned int> attribute_offsets;

        struct {
            CUDABuffer indices_offsets;
            CUDABuffer attribute_offsets;

            CUDABuffer vertices_buffer;
            CUDABuffer indices_buffer;
            CUDABuffer normals_buffer;
            CUDABuffer uvs_buffer;
        } scene_geometry;

        std::vector<MaterialDescription> material_descriptions;
        CUDABuffer material_descriptions_buf;

        // acceleration structure
        std::vector<OptixTraversableHandle> gases;
        std::vector<CUDABuffer> gasBuffers;

        std::vector<OptixTraversableHandle> gas_motion_transform_handles;
        std::vector<CUDABuffer> gas_motion_transform_buffers;

        OptixTraversableHandle ias;
        CUDABuffer ias_buffer;

        /*! @{ CUDA device context and stream that optix pipeline will run
          on, as well as device properties for this device */
        CUcontext          cudaContext;
        CUstream           stream;
        cudaDeviceProp     deviceProps;
        /*! @} */

        //the optix context that our pipeline will run in.
        OptixDeviceContext optixContext;

        /*! @{ the pipeline we're building */
        OptixPipeline               pipeline;
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        OptixPipelineLinkOptions    pipelineLinkOptions    = {};
        /*! @} */

        /*! @{ the module that contains our device programs */
        OptixModule                 module;
        OptixModuleCompileOptions   moduleCompileOptions = {};
        /* @} */

        /*! vector of all our program(group)s, and the SBT built around
          them */
        std::vector<OptixProgramGroup> raygenPGs;
        CUDABuffer raygenRecordsBuffer;
        std::vector<OptixProgramGroup> missPGs;
        CUDABuffer missRecordsBuffer;
        std::vector<OptixProgramGroup> hitgroupPGs;
        CUDABuffer hitgroupRecordsBuffer;
        OptixShaderBindingTable sbt = {};

        /*! @{ our launch parameters, on the host, and the buffer to store
          them on the device */
        LaunchParams launchParams;
        CUDABuffer   launchParamsBuffer;
        /*! @} */

        CUDABuffer colorBuffer;
        CUDABuffer accumBuffer;

        rclcpp::Node::SharedPtr node;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub;

        double3 robot_pos;
        double4 robot_att;

};
