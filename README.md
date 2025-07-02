# Anglerfish
**Warning** This repo is a work in progress!

Anglerfish is an [OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) based underwater raytracer that publish its frame in a ROS2 topic. 
The camera movement are comming from [Stonefish_ros2](https://github.com/patrykcieslak/stonefish_ros2) through an odometry sensor topic.

## Dependencies
- OptiX 7.7.0
- Cuda
- assimp (libassymp-dev)
- ROS2 (tested on jazzy)

## Scenario
Scenario files parsed by Anglerfish are based on one's used by Stonefish. Then you can use your actual Stonefish scenario file in Anglerfihs by just adding some tags. 

### Materials
You need to specify what kind of type are your materials. 
For now it can be: 
- lambertian (purely Lambertian)
- mirror (purely reflective)
Example: 
```xml
<material name="Mirror" density="1250" restitution="0.5" type="mirror"/>
```
 
### Look
When a texture is specified then it will also be applied in Anglerfish (If it is compatible with the material type, either it will be ignored).
