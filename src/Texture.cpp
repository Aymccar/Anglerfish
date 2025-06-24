#include "utils/Texture.h"
#include "utils/optix_util.h"


cudaTextureObject_t Texture::getTextureObject() {
    return cuda_tex;
}

// TODO destructor
