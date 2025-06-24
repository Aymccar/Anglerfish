#pragma once
#include<vector>
#include<vector_types.h>
#include<cmath>

#define PI 3.14159265359

double3 quat2euler(double x, double y, double z, double w){
    double phi = std::atan2(2*(w*x+y*z), 1-2*(x*x+y*y)); 
    double theta = -PI/2+2*std::atan2(std::sqrt(1+2*(w*y-x*z)), std::sqrt(1-2*(w*y-x*z)));
    double psi = std::atan2(2*(w*z+x*y), 1-2*(y*y+z*z));

    return {phi, theta, psi};
}

float3 dir_from_euler(float3 att){
}
