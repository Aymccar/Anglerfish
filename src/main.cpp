#include <vector_types.h>
#include <vector_functions.h>

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sstream>

#include "Renderer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        constexpr unsigned int num_samples_per_pixel = 10;
        Renderer renderer;

        renderer.loadScene("/home/aymeric/Desktop/ros2_ws/install/stonefish_ros2/share/stonefish_ros2/other/scenarios/bluerov_heavy_simplified_shipwreck.scn");

        const uint2 fbSize = make_uint2(1920, 1080);
        renderer.resize(fbSize);

        while (1){
            renderer.updateScene();
            for (unsigned int i = 0; i < num_samples_per_pixel; i++) {
                renderer.render();
            }

            //Get pixels
            std::vector<uint32_t> pixels(fbSize.x * fbSize.y);
            renderer.downloadPixels(pixels.data());

            //Create ROS2 msg
            sensor_msgs::msg::Image msg;
            msg.height = fbSize.y;
            msg.width = fbSize.x;
            msg.encoding = "rgb8";  
            msg.is_bigendian = false;
            msg.step = msg.width * 3;

            msg.data.resize(msg.height * msg.step);
            uint8_t* dst = msg.data.data();
            for (size_t i = 0; i < pixels.size(); ++i) {
                uint32_t pixel = pixels[i];
                dst[i * 3 + 0] = (pixel >> 0) & 0xFF;   // R
                dst[i * 3 + 1] = (pixel >> 8) & 0xFF;   // G
                dst[i * 3 + 2] = (pixel >> 16) & 0xFF;  // B
            }

            //Publish
            renderer.publish(msg);
        }
    } catch (std::runtime_error& e) {
        std::cout << "FATAL ERROR: " << e.what() << std::endl;
        exit(1);
    }

    return 0;
}
