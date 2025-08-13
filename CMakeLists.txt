cmake_minimum_required(VERSION 3.8)
project(anglerfish)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPTIX_ROOT "/home/aymeric/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64")
enable_language(CUDA)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(assimp REQUIRED)
find_package(ament_index_cpp REQUIRED)

# paths
set(PTX_INPUT ${CMAKE_SOURCE_DIR}/src/devicePrograms.cu)
set(PTX_OUTPUT ${CMAKE_BINARY_DIR}/devicePrograms.ptx)
set(PTX_EMBEDDED_CPP ${CMAKE_BINARY_DIR}/embeddedDevicePrograms.cpp)

# step 1: generate devicePrograms.ptx
add_custom_command(
  OUTPUT ${PTX_OUTPUT}
  COMMAND ${CMAKE_CUDA_COMPILER} 
  -std=c++17 
  -I${CMAKE_SOURCE_DIR}/include 
  -I${OPTIX_ROOT}/include 
  -arch=sm_89
  --ptx ${PTX_INPUT} 
  -O3
  -lineinfo
  -o ${PTX_OUTPUT}
  DEPENDS ${PTX_INPUT}
  COMMENT "Compiling ${PTX_INPUT} to PTX"
)

# step 2: embed PTX using xxd
add_custom_command(
  OUTPUT ${PTX_EMBEDDED_CPP}
  COMMAND xxd -i -n embedded_ptx_code ${PTX_OUTPUT} > ${PTX_EMBEDDED_CPP}
  DEPENDS ${PTX_OUTPUT}
  COMMENT "Embedding PTX with xxd"
)

# step 3: create static library from generated .cpp
add_library(embedded_ptx STATIC ${PTX_EMBEDDED_CPP})
target_include_directories(embedded_ptx PUBLIC ${CMAKE_BINARY_DIR})

# step 4: collect all sources
file(GLOB_RECURSE SOURCES "src/*.cpp")

# step 5: add executable
add_executable(${PROJECT_NAME} 
    ${SOURCES}
    "include/external/tinyxml2.cpp"  
    )

add_dependencies(${PROJECT_NAME} embedded_ptx)

# link libraries
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  nav_msgs
  sensor_msgs
  ament_index_cpp
)

target_link_libraries(${PROJECT_NAME}
    embedded_ptx
    assimp
    CUDA::cudart
    CUDA::cuda_driver
)

# include paths
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_ROOT}/include
    ${CMAKE_BINARY_DIR}  # so embedded .cpp can be found
)

# install targets
install(TARGETS ${PROJECT_NAME}
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION share/include/
)

ament_package()
