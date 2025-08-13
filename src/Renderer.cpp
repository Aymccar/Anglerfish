#include "Renderer.h"
#include "Parser.h"

#include <iostream>
#include <cstring>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <fstream>

#include "utils/vec_math.h"
#include "utils/math.h"
#include "utils/Texture.h"

#include "materials/MaterialDescription.h"

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>


using namespace tinyxml2;

extern "C" unsigned char embedded_ptx_code[];
extern "C" unsigned int embedded_ptx_code_len;

/* SBT record for a raygen program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

/* SBT record for a miss program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

/* SBT record for a hitgroup program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

/* constructor - performs all setup, including initializing
  optix, creates module, pipeline, programs, SBT, etc. */
Renderer::Renderer(){
    node = std::make_shared<rclcpp::Node>("Robot", "anglerfish");
    sub = node->create_subscription<nav_msgs::msg::Odometry>("/bluerov/navigator/odometry", 1, [this](nav_msgs::msg::Odometry::SharedPtr msg){

            this->robot_pos = double3({msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z});
            //this->robot_att = quat2euler(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
            this->robot_att = {msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w};
            });

    pub = node->create_publisher<sensor_msgs::msg::Image>("image", 1);

    initOptix();

    std::cout << "creating optix context" << std::endl;
    createContext();

    std::cout << "setting up module" << std::endl;
    createModule();

    std::cout << "creating raygen programs" << std::endl;
    createRaygenPrograms();
    std::cout << "creating miss programs" << std::endl;
    createMissPrograms();
    std::cout << "creating hitgroup programs" << std::endl;
    createHitgroupPrograms();

    std::cout << "setting up optix pipeline" << std::endl;
    createPipeline();


    std::cout << "building SBT" << std::endl;
    buildSBT();

    launchParamsBuffer.alloc(sizeof(launchParams));
    std::cout << "context, module, pipeline, etc., all set up" << std::endl;
}


MaterialDescription load_scene_material_description(const Material& material, const Look& look) {
    MaterialDescription md;
    const std::string type_str = material.type;
    Color color;

    if (type_str == "lambertian") {
        md.type = lambertian;
        if (look.texture != ""){
            //Load image:

            int width, height, number_of_components;
            unsigned char *texture_data = stbi_load(look.texture.c_str(), &width, &height, &number_of_components, 4);

            Texture texture = Texture::Create<uchar4>(
                width,
                height,
                texture_data,
                cudaAddressModeWrap,
                cudaAddressModeWrap,
                cudaFilterModeLinear,
                cudaReadModeNormalizedFloat,
                true
            );

            color.setTexture(texture);
        }
        else {
            color.setColor(look.rgb);
        }
        md.lambertian_brdf_config.color = color;

    } else if (type_str == "mirror") {
        md.type = mirror;
    } else if (type_str == "light") {
        md.type = light;
        md.is_emissive = true;
        md.emission = look.rgb * 15;
    }

    return md;
}


Transform4 load_scene_transform(const Mesh& mesh) {

    Transform4 rotX = rotate(mesh.world_rotation.x, float3({1, 0, 0}));
    Transform4 rotY = rotate(mesh.world_rotation.y, float3({0, 1, 0}));
    Transform4 rotZ = rotate(mesh.world_rotation.z, float3({0, 0, 1}));
    Transform4 R = transpose(rotX * rotY * rotZ); //To world transform

    Transform4 O = translate(mesh.origin.x, mesh.origin.y, mesh.origin.z);
    Transform4 scaleMat = scale(mesh.scale, mesh.scale, mesh.scale);
    Transform4 T = translate(mesh.world_translation.x, 
                             mesh.world_translation.y, 
                             mesh.world_translation.z);

    Transform4 t = T * R * scaleMat * O;
    return t;
}


void Renderer::loadScene(std::string scene_filename, std::string data_path) {

    // TODO tangents, bitangents
    all_vertices.clear();
    all_indices.clear();
    all_normals.clear();
    all_uvs.clear();

    XMLParser parser(scene_filename, data_path);
    parser.parse_file();

    Scenario scenario;
    parser.get_scenario(scenario);

    for (auto mesh : scenario.Static_meshes){
        std::cout<< mesh.file_name<<std::endl;
    }

    for (const auto& object: scenario.Static_meshes) { // load the models
        // loading the mesh
        std::vector<float3> vertices;
        std::vector<uint3> indices;
        std::vector<float3> normals;
        std::vector<float2> uvs;

        Assimp::Importer importer;
        const aiScene *scene = importer.ReadFile(object.file_name, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals);
        if (scene == nullptr) {
            std::cerr << "ERROR: " << importer.GetErrorString() << std::endl;
            exit(3);
        }

        // XXX only works with single object files
        const aiMesh *mesh = scene->mMeshes[0];

        // load vertices
        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            vertices.emplace_back(make_float3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z));
        }

        // load faces
        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            const aiFace &face = mesh->mFaces[i];
            if (face.mNumIndices != 3) {
                std::cerr << "model " << object.file_name << " contains non-triangular faces" << std::endl;
                exit(2);
            }

            indices.emplace_back(make_uint3(face.mIndices[0], face.mIndices[1], face.mIndices[2]));
        }

        // load normals
        if (!mesh->HasNormals()) {
            std::cerr << "model " << object.file_name << " doesn't have normals" << std::endl;
            exit(2);
        }

        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            normals.emplace_back(make_float3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z));
        }

        // load UVs if they exist, if not, set them to 0
        // only handles the 0th UV channel
        uvs.resize(vertices.size());
        if (mesh->HasTextureCoords(0)) {
            for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
                uvs[i] = make_float2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
            }
        } else {
            std::fill(uvs.begin(), uvs.end(), make_float2(0.f));
        }

        all_vertices.push_back(vertices);
        all_indices.push_back(indices);
        all_normals.push_back(normals);
        all_uvs.push_back(uvs);
        

        // getting associated material and look
        Material material;
        for (const auto& mat : scenario.Materials){
            if (mat.name == object.material){material = mat;}
        }

        Look look;
        for (const auto& lk : scenario.Looks){
            if (lk.name == object.look){look = lk;}
        }
        material_descriptions.push_back(load_scene_material_description(material, look));

        // loading the transform or initializing a default one
        transforms.push_back(load_scene_transform(object));
    }


    populateSceneBuffers();

    buildAccel();

    uploadMaterials();

    is_scene_initialized = true;
}


void Renderer::uploadMaterials() {
    // upload the material descriptions to the GPU
    material_descriptions_buf.alloc_and_upload(material_descriptions);

    // configure the material descriptions in the LaunchParams
    launchParams.material_descriptions = (MaterialDescription *)material_descriptions_buf.d_pointer();
}


void Renderer::populateSceneBuffers() {
    std::vector<uint3> concatenated_indices;
    std::vector<float3> concatenated_vertices;
    std::vector<float3> concatenated_normals;
    std::vector<float2> concatenated_uvs;

    unsigned int index_offset = 0, attribute_offset = 0;

    for (unsigned int i = 0; i < all_vertices.size(); i++) {
        indices_offsets.push_back(index_offset);
        attribute_offsets.push_back(attribute_offset);

        concatenated_indices.insert(concatenated_indices.end(), all_indices[i].begin(), all_indices[i].end());
        concatenated_vertices.insert(concatenated_vertices.end(), all_vertices[i].begin(), all_vertices[i].end());
        concatenated_normals.insert(concatenated_normals.end(), all_normals[i].begin(), all_normals[i].end());
        concatenated_uvs.insert(concatenated_uvs.end(), all_uvs[i].begin(), all_uvs[i].end());

        index_offset += all_indices[i].size();
        attribute_offset += all_vertices[i].size();
    }

    // upload the concatenated buffers to the GPU
    scene_geometry.indices_buffer.alloc_and_upload(concatenated_indices);
    scene_geometry.vertices_buffer.alloc_and_upload(concatenated_vertices);
    scene_geometry.normals_buffer.alloc_and_upload(concatenated_normals);
    scene_geometry.uvs_buffer.alloc_and_upload(concatenated_uvs);

    // upload the offset buffers on the GPU
    scene_geometry.indices_offsets.alloc_and_upload(indices_offsets);
    scene_geometry.attribute_offsets.alloc_and_upload(attribute_offsets);

    // configure the concatenated buffers in LaunchParams
    launchParams.scene_geometry.indices_buffer = (uint3 *)scene_geometry.indices_buffer.d_pointer();
    launchParams.scene_geometry.vertices_buffer = (float3 *)scene_geometry.vertices_buffer.d_pointer();
    launchParams.scene_geometry.normals_buffer = (float3 *)scene_geometry.normals_buffer.d_pointer();
    launchParams.scene_geometry.uvs_buffer = (float2 *)scene_geometry.uvs_buffer.d_pointer();

    // configure the offset buffers in LaunchParams
    launchParams.scene_geometry.indices_offsets = (unsigned int *)scene_geometry.indices_offsets.d_pointer();
    launchParams.scene_geometry.attribute_offsets = (unsigned int *)scene_geometry.attribute_offsets.d_pointer();
}

/* helper function that initializes optix and checks for errors */
void Renderer::initOptix() {
    std::cout << "initializing optix..." << std::endl;

    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("no CUDA capable devices found!");
    std::cout << "found " << numDevices << " CUDA devices" << std::endl;

    OPTIX_CHECK( optixInit() );
    std::cout << "successfully initialized optix... yay!" << std::endl;
}

static void context_log_cb(unsigned int level,
        const char *tag,
        const char *message,
        void *)
{
    fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
}

/* creates and configures a optix device context (in this simple
  example, only for the primary GPU device) */
void Renderer::createContext() {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "running on device: " << deviceProps.name << std::endl;

    CUresult cuRes = cuCtxGetCurrent(&cudaContext);
    if( cuRes != CUDA_SUCCESS )
        fprintf( stderr, "Error querying current context: error code %d\n", cuRes );

    OptixDeviceContextOptions options;
    options.logCallbackFunction = context_log_cb;
    options.logCallbackData = nullptr;
    options.logCallbackLevel = 4;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    //options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &optixContext));
    //OPTIX_CHECK(optixDeviceContextCreate(cudaContext, nullptr, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
            (optixContext,context_log_cb,nullptr,4));
}



/* creates the module that contains all the programs we are going
  to use. in this simple example, we use a single module from a
  single .cu file, using a single embedded ptx string */
void Renderer::createModule() {
    moduleCompileOptions.maxRegisterCount  = 150;  // XXX
    moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;  // XXX
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;  // XXX
    //moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;  // XXX
    pipelineCompileOptions.usesMotionBlur     = true;  // XXX
    pipelineCompileOptions.numPayloadValues   = 2;  // XXX
    pipelineCompileOptions.numAttributeValues = 2;  // barycentrics
    pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;//OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth          = 3;  // XXX

    char log[2048];
    size_t sizeof_log = sizeof( log );
#if OPTIX_VERSION >= 70700
    OPTIX_CHECK(optixModuleCreate(optixContext,
                &moduleCompileOptions,
                &pipelineCompileOptions,
                (char *)embedded_ptx_code,
                embedded_ptx_code_len,
                log,&sizeof_log,
                &module
                ));
#else
    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                &moduleCompileOptions,
                &pipelineCompileOptions,
                (char *)embedded_ptx_code,
                embedded_ptx_code_len,
                log,      // Log string
                &sizeof_log,// Log string size
                &module
                ));
#endif
    if (sizeof_log > 1)
        std::cout << log << std::endl;
}



/* does all setup for the raygen program(s) we are going to use */
void Renderer::createRaygenPrograms() {
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                &pgDesc,
                1,
                &pgOptions,
                log,&sizeof_log,
                &raygenPGs[0]
                ));

    if (sizeof_log > 1)
        std::cout << log << std::endl;
}

/* does all setup for the miss program(s) we are going to use */
void Renderer::createMissPrograms() {
    missPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module            = module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                &pgDesc,
                1,
                &pgOptions,
                log,&sizeof_log,
                &missPGs[0]
                ));
    if (sizeof_log > 1)
        std::cout << log << std::endl;
}

/* does all setup for the hitgroup program(s) we are going to use */
void Renderer::createHitgroupPrograms() {
    hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH            = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                &pgDesc,
                1,
                &pgOptions,
                log,&sizeof_log,
                &hitgroupPGs[0]
                ));

    if (sizeof_log > 1)
        std::cout << log << std::endl;
}


/* assembles the full pipeline of all programs */
void Renderer::createPipeline() {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
        programGroups.push_back(pg);
    for (auto pg : missPGs)
        programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
        programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixPipelineCreate(optixContext,
                &pipelineCompileOptions,
                &pipelineLinkOptions,
                programGroups.data(),
                (int)programGroups.size(),
                log,&sizeof_log,
                &pipeline
                ));
    if (sizeof_log > 1)
        std::cout << log << std::endl;

    OPTIX_CHECK(optixPipelineSetStackSize
            (/* [in] The pipeline to configure the stack size for */
             pipeline,
             /* [in] The direct stack size requirement for direct
                callables invoked from IS or AH. */
             20*1024,
             /* [in] The direct stack size requirement for direct
                callables invoked from RG, MS, or CH.  */
             20*1024,
             /* [in] The continuation stack requirement. */
             20*1024,
             /* [in] The maximum depth of a traversable graph
                passed to trace. */
             // XXX IAS -> MT -> GAS
             3));

    if (sizeof_log > 1)
        std::cout << log << std::endl;
}


/* constructs the shader binding table */
void Renderer::buildSBT() {
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (unsigned int i=0;i<raygenPGs.size();i++) {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i],&rec));
        raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (unsigned int i=0;i<missPGs.size();i++) {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i],&rec));
        missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase          = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount         = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------

    int numObjects = 1;
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int i=0;i<numObjects;i++) {
        int objectType = 0;
        HitgroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType],&rec));
        hitgroupRecords.push_back(rec);
    }

    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);

    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
}



/* render one frame */
void Renderer::render() {
    // sanity check: make sure we launch only after first resize is
    // already done:
    if (launchParams.fbSize.x == 0)
        return;

    if (is_accel_dirty) {
        updateAccel();
        uploadMaterials();
        launchParams.subframe_idx = 0;
        is_accel_dirty = false;
    } else {
        launchParams.subframe_idx++;
    }

    if (!is_scene_initialized) {
        std::cerr << "Renderer::render(): scene not initialized, did you forget to loadScene()?" << std::endl;
        return;
    }

    launchParams.camera.focal = 35;
    launchParams.camera.lens_diameter = 0.100;
    launchParams.camera.film_size = make_float2(36, 24);

    launchParamsBuffer.upload(&launchParams,1);
    launchParams.frameID++;

    OPTIX_CHECK(optixLaunch(/* pipeline we're launching launch: */
                pipeline,stream,
                /* parameters and SBT */
                launchParamsBuffer.d_pointer(),
                launchParamsBuffer.sizeInBytes,
                &sbt,
                /* dimensions of the launch: */
                launchParams.fbSize.x,
                launchParams.fbSize.y,
                3  // R, G, B
                ));

    // sync - make sure the frame is rendered before we download and display
    CUDA_SYNC_CHECK();
}

void Renderer::buildAccel() {
    gasBuffers.resize(all_indices.size());
    gases.resize(all_indices.size());

    // building GASes
    for (unsigned int i = 0; i < all_indices.size(); i++) {
        OptixBuildInput buildInput = {};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        float3 *vertexBuffer = ((float3 *)scene_geometry.vertices_buffer.d_pointer()) + attribute_offsets[i];
        uint3 *indexBuffer = ((uint3 *)scene_geometry.indices_buffer.d_pointer()) + indices_offsets[i];

        buildInput.triangleArray.vertexBuffers = (CUdeviceptr *)&vertexBuffer;  // this is a pointer to an array of pointers due to motion keys
        buildInput.triangleArray.numVertices = all_vertices[i].size() * sizeof(float3);
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);

        buildInput.triangleArray.indexBuffer = (CUdeviceptr)indexBuffer;
        buildInput.triangleArray.numIndexTriplets = all_indices[i].size();
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);

        uint32_t triangleInputFlags[1] = { 0 };

        // we have one SBT entry, and no per-primitive data
        buildInput.triangleArray.flags               = triangleInputFlags;
        buildInput.triangleArray.numSbtRecords               = 1;
        buildInput.triangleArray.sbtIndexOffsetBuffer        = 0;
        buildInput.triangleArray.sbtIndexOffsetSizeInBytes   = 0;
        buildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;


        // build options
        OptixAccelBuildOptions buildOptions = {};
        buildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
            //| OPTIX_BUILD_FLAG_ALLOW_COMPACTION
            | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        buildOptions.motionOptions.numKeys = 1;  // XXX for morphing geometry

        // compute the buffer size for the geometric acceleration structure
        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixContext,
                 &buildOptions,
                 &buildInput,
                 1,  // num_build_inputs
                 &gasBufferSizes
                ));

        // allocate the memory for the build
        CUDABuffer tempBuffer;
        tempBuffer.alloc(gasBufferSizes.tempSizeInBytes);

        gasBuffers[i].alloc(gasBufferSizes.outputSizeInBytes);

        // do the build
        OPTIX_CHECK(optixAccelBuild(optixContext,
                    0, // stream
                    &buildOptions,
                    &buildInput,
                    1,  // number of build inputs
                    tempBuffer.d_pointer(),
                    tempBuffer.sizeInBytes,

                    gasBuffers[i].d_pointer(),
                    gasBuffers[i].sizeInBytes,

                    &gases[i],  // output traversable handle

                    nullptr, // emittedProperties
                    0  // numEmittedProperties
                    ));

        CUDA_SYNC_CHECK();

        // TODO you could compact the GAS to go even faster

        tempBuffer.free();
    }

    buildMotionTransforms();

    buildInstanceAccelerationStructure(false);
}

void Renderer::buildMotionTransforms() {
    gas_motion_transform_handles.resize(all_indices.size());
    gas_motion_transform_buffers.resize(all_indices.size());

    for (unsigned int i = 0; i < all_indices.size(); i++) {
        OptixMatrixMotionTransform motion_transform;

        motion_transform.child = gases[i];
        motion_transform.motionOptions.numKeys = 2;
        motion_transform.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
        motion_transform.motionOptions.timeBegin = 0;
        motion_transform.motionOptions.timeEnd = 1;

        // XXX for now, we only use one transform for both the start and the end times
        transforms[i].to_float12(motion_transform.transform[0]);
        transforms[i].to_float12(motion_transform.transform[1]);

        gas_motion_transform_buffers[i].resize(sizeof(OptixMatrixMotionTransform));
        gas_motion_transform_buffers[i].upload(&motion_transform, 1);

        OPTIX_CHECK(optixConvertPointerToTraversableHandle(
            optixContext,
            gas_motion_transform_buffers[i].d_pointer(),
            OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM,
            &gas_motion_transform_handles[i]
        ))
    }
}

void Renderer::buildInstanceAccelerationStructure(bool shouldUpdate) {
    // TODO respect shouldUpdate
    float identity3x4[12] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    };

    // populate the instances list
    std::vector<OptixInstance> host_instances;
    host_instances.resize(all_indices.size());

    for (unsigned int i = 0; i < all_indices.size(); i++) {
        memcpy(host_instances[i].transform, identity3x4, sizeof(float) * 12);
        host_instances[i].instanceId = i;
        host_instances[i].sbtOffset = 0;
        host_instances[i].visibilityMask = 1;
        host_instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
        host_instances[i].traversableHandle = gas_motion_transform_handles[i];
    }

    CUDABuffer instances_buffer;
    instances_buffer.alloc_and_upload(host_instances);

    // create the build input
    OptixBuildInput iasBuildInput{};

    iasBuildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    iasBuildInput.instanceArray.numInstances = all_indices.size();
    iasBuildInput.instanceArray.instances = instances_buffer.d_pointer();

    // set up the build options
    OptixAccelBuildOptions iasBuildOptions = {};
    iasBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;  // TODO allow update
    iasBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    iasBuildOptions.motionOptions.numKeys = 1;
    iasBuildOptions.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
    iasBuildOptions.motionOptions.timeBegin = 0;
    iasBuildOptions.motionOptions.timeEnd = 1;

    // compute the buffer size for the ias
    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
            (optixContext,
             &iasBuildOptions,
             &iasBuildInput,
             1,  // num_build_inputs
             &iasBufferSizes
            ));

    // allocate the memory for the build
    CUDABuffer tempBuffer;
    tempBuffer.alloc(iasBufferSizes.tempSizeInBytes);

    ias_buffer.resize(iasBufferSizes.outputSizeInBytes);

    // do the build
    OPTIX_CHECK(optixAccelBuild(optixContext,
                0, // stream
                &iasBuildOptions,
                &iasBuildInput,
                1,  // number of build inputs
                tempBuffer.d_pointer(),
                tempBuffer.sizeInBytes,

                ias_buffer.d_pointer(),
                ias_buffer.sizeInBytes,

                &ias,  // output traversable handle

                nullptr, // emittedProperties
                0  // numEmittedProperties
                ));

    CUDA_SYNC_CHECK();

    tempBuffer.free();
    instances_buffer.free();

    launchParams.traversable = ias;
}

void Renderer::updateScene() {

    // new camera position
    constexpr float cameraAngularSpeed = 2 * M_PIf;
    constexpr float cameraDistanceFromCenter = 20.0f;

    //callback
    rclcpp::spin_some(node);

    launchParams.camera.origin = make_float3(robot_pos.x, robot_pos.y, robot_pos.z); 
    launchParams.camera.att = {(float)robot_att.x, (float)robot_att.y, (float)robot_att.z, (float)robot_att.w};

    //UGLY
    Transform4 transform;

    // Position
    float3 t = make_float3(robot_pos.x, robot_pos.y, robot_pos.z);

    // Quaternion (normalized)
    const float qr = robot_att.w;
    const float qi = robot_att.x;
    const float qj = robot_att.y;
    const float qk = robot_att.z;

    const float r00 = 1 - 2 *(qj * qj + qk * qk);
    const float r01 = 2 * (qi * qj - qk * qr);
    const float r02 = 2 * (qi * qk + qj * qr);
     
    const float r10 = 2 * (qi * qj + qk * qr);
    const float r11 = 1 - 2 * (qi * qi + qk * qk);
    const float r12 = 2 * (qj * qk - qi * qr);
     
    const float r20 = 2 * (qi * qk - qj * qr);
    const float r21 = 2 * (qj * qk + qi * qr);
    const float r22 = 1 - 2 * (qi * qi + qj * qj);

    // Row-major rotation + translation packed in float4s
    // Each row is float4: (Rxx, Rxy, Rxz, Tx) etc.

    // First row
    transform.m[0] = make_float4(
        r00,
        r01,
        r02,
        t.x
    );

    // Second row
    transform.m[1] = make_float4(
        r10,
        r11,
        r12,
        t.y
    );

    // Third row
    transform.m[2] = make_float4(
        r20,
        r21,
        r22,
        t.z
    );

    // Fourth row (unused for affine, but can be identity row)
    transform.m[3] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    transform = transform * rotate(1.57, {0, 0, 1}) * translate(3, 0, 0) * scale(1, 1, 1);
    transforms[0] = transform;
                                                                                    
    is_accel_dirty = true;
}

void Renderer::updateAccel() {
    // rebuild the motion transforms
    buildMotionTransforms();

    // rebuild the IAS
    buildInstanceAccelerationStructure(true);
}

/* resize frame buffer to given resolution */
void Renderer::resize(const uint2 &newSize) {
    // if window minimized
    if (newSize.x == 0 || newSize.y == 0)
        return;

    // resize our cuda frame buffer
    colorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));
    accumBuffer.resize(newSize.x * newSize.y * sizeof(float4));

    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.fbSize      = newSize;
    launchParams.colorBuffer = (uchar4 *)colorBuffer.d_ptr;
    launchParams.accumBuffer = (float4*)accumBuffer.d_ptr;
    launchParams.subframe_idx = 0;
}

/* download the rendered color buffer */
void Renderer::downloadPixels(uint32_t h_pixels[]) {
    colorBuffer.download(h_pixels,
            launchParams.fbSize.x*launchParams.fbSize.y);
}

void Renderer::publish(sensor_msgs::msg::Image msg){
    pub->publish(msg);
}
