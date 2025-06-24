#pragma once
#include <cuda_runtime.h>
#include "optix_util.h"
#include <iostream>
#include <cstring>


class Texture {
    public:
        template <typename TexelType>
        static Texture Create(
            const unsigned int width,
            const unsigned int height,
            const void * data,
            cudaTextureAddressMode addressMode_s = cudaAddressModeWrap,
            cudaTextureAddressMode addressMode_t = cudaAddressModeWrap,
            cudaTextureFilterMode filterMode = cudaFilterModeLinear,
            cudaTextureReadMode readMode = cudaReadModeNormalizedFloat,
            bool sRGB = false
        );

        // TODO delete copy constructor and copy assignment operator

        cudaTextureObject_t getTextureObject();

    private:
        cudaChannelFormatDesc formatDesc{};
        cudaArray_t cuda_array{};
        cudaResourceDesc res_desc{};
        cudaTextureDesc texture_desc{};
        cudaTextureObject_t cuda_tex = 0;
};


template <typename TexelType>
Texture Texture::Create(
    const unsigned int width,
    const unsigned int height,
    const void * data,
    cudaTextureAddressMode addressMode_s,
    cudaTextureAddressMode addressMode_t,
    cudaTextureFilterMode filterMode,
    cudaTextureReadMode readMode,
    bool sRGB
) {
    Texture to_ret;

    // create cuda array
    //to_ret.formatDesc = cudaCreateChannelDesc<TexelType>();
    to_ret.formatDesc = cudaCreateChannelDesc<TexelType>();
    if (to_ret.formatDesc.f == cudaChannelFormatKindNone) {
        std::cerr << "unknown channel format in texture\n";
        exit(2);
    }

    const uint32_t pitch = width * sizeof(TexelType);  // no padding
    to_ret.cuda_array = nullptr;
    CUDA_CHECK(MallocArray(
        &to_ret.cuda_array,
        &to_ret.formatDesc,
        width,
        height
    ));

    // copy data to array
    CUDA_CHECK(Memcpy2DToArray(
        to_ret.cuda_array,
        0, 0, // offsets
        data,
        pitch,
        width * sizeof(TexelType),  // the width is measured in bytes, the height is measured in rows
        height,
        cudaMemcpyHostToDevice
    ));

    // create cuda resource description
    memset(&to_ret.res_desc, 0, sizeof(to_ret.res_desc));
    to_ret.res_desc.resType = cudaResourceTypeArray;
    to_ret.res_desc.res.array.array = to_ret.cuda_array;

    // create cuda texture
    memset(&to_ret.texture_desc, 0, sizeof(to_ret.texture_desc));
    to_ret.texture_desc.addressMode[0] = addressMode_t;
    to_ret.texture_desc.addressMode[1] = addressMode_s;
    to_ret.texture_desc.filterMode = filterMode;
    to_ret.texture_desc.readMode = readMode;
    to_ret.texture_desc.normalizedCoords = 1;
    //to_ret.texture_desc.maxAnisotropy = 1;
    //to_ret.texture_desc.maxMipmapLevelClamp = 0;
    //to_ret.texture_desc.minMipmapLevelClamp = 0;
    to_ret.texture_desc.mipmapFilterMode = cudaFilterModeLinear;
    to_ret.texture_desc.sRGB = sRGB;

    to_ret.cuda_tex = 0;
    CUDA_CHECK(CreateTextureObject(
        &to_ret.cuda_tex,
        &to_ret.res_desc,
        &to_ret.texture_desc,
        nullptr
    ));

    return to_ret;
}
