#pragma once

#include "vec_math.h"
#include "mat_math.h"
#include <cuda_runtime.h>

#include "../external/Eigen/Dense"



/*
    All the wavelength values are given in nm and not in meters to facilitate human comprehension
*/


struct DenseSpectrum;
struct SampledSpectrum;

static float const lambda_min = 300;
static float const lambda_max = 900;
static int const SAMPLE_SIZE = lambda_max - lambda_min + 1;

typedef Eigen::Matrix<float, 3, SAMPLE_SIZE, Eigen::RowMajor> MatrixSpectrumToXYZ;
typedef Eigen::Matrix<float, SAMPLE_SIZE, 3, Eigen::RowMajor> MatrixXYZToSpectrum;

static constexpr float CIE_Y_integral = 116.92804;
static constexpr float CIE_X_WEIGHT = 0.166937;


__host__ __device__ struct XYZColor {
    float x, y, z;

    __device__ XYZColor() : x(0), y(0), z(0) {}
    __device__ XYZColor(float x, float y, float z) : x(x), y(y), z(z) {}
    __device__ XYZColor(float3 u) : x(u.x), y(u.y), z(u.z) {}

    __device__ float3 tofloat3() {
        return make_float3(x, y, z);
    }


    static __device__ XYZColor fromRGB(float3 rgb) {
        XYZColor xyz;
        xyz.x = 0.4124564 * rgb.x + 0.3575761 * rgb.y + 0.1804375 * rgb.z;
        xyz.y = 0.2126729 * rgb.x + 0.7151522 * rgb.y + 0.0721750 * rgb.z;
        xyz.z = 0.0193339 * rgb.x + 0.1191920 * rgb.y + 0.9503041 * rgb.z;
        return xyz;
    }

    __device__ float3 toRGB() {
        float3 rgb = make_float3(
             3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
            -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
             0.0556434 * x - 0.2040259 * y + 1.0572252 * z
        );
        return rgb;
    }

    float2 toxyY() {
        return make_float2(x / (x + y + z), y / (x + y + z));
    }

    static XYZColor fromxyY(float2 xy, float Y = 1) {
        if (xy.y == 0) return XYZColor (0, 0, 0);

        return XYZColor(xy.x * Y / xy.y, Y, (1 - xy.x - xy.y) * Y / xy.y);
    }

    float3 toLMS() {
        Transform4 XYZToLMS = {
            .m = {
                make_float4( 0.389709217, 0.688980868, -0.0786861881, 0),
                make_float4(-0.229811039,  1.18340759, 0.0464011469,  0),
                make_float4(           0,           0,            1,  0),
                make_float4(           0,           0,            0,  1)}
        };
        return make_float3(XYZToLMS * tofloat3());
    }

    static XYZColor fromLMS(float3 lms) {

        Transform4 LMSToXYZ = {
            .m = {
                make_float4(1.91020, -1.11212, 0.20191, 0),
                make_float4(0.37095,  0.62905,       0, 0),
                make_float4(      0,        0,       1, 0),
                make_float4(      0,        0,       0, 1)}
        };
        float4 xyz = LMSToXYZ * lms;
        return XYZColor(xyz.x, xyz.y, xyz.z);
    }

    float __device__ operator[](unsigned int i) {
        if (i==0) return x;
        if (i==1) return y;
        return z;
    }
    
    DenseSpectrum __host__ __device__ toSpectrum();


    
};



inline __host__ __device__ Transform4 WhiteBalance(XYZColor srcWhite, XYZColor destWhite) {
    float3 srcLMS = srcWhite.toLMS(); float3 destLMS = destWhite.toLMS();
    Transform4 XYZToLMS = {
        .m = {
            make_float4( 0.389709217, 0.688980868, -0.0786861881, 0),
            make_float4(-0.229811039,  1.18340759, 0.0464011469,  0),
            make_float4(           0,           0,            1,  0),
            make_float4(           0,           0,            0,  1)}
    };
    Transform4 LMSToXYZ = {
        .m = {
            make_float4(1.91020, -1.11212, 0.20191, 0),
            make_float4(0.37095,  0.62905,       0, 0),
            make_float4(      0,        0,       1, 0),
            make_float4(      0,        0,       0, 1)}
    };

    Transform4 LMSCorrect = {
        .m = {
            make_float4(destLMS.x / srcLMS.x, 0, 0, 0),
            make_float4(0, destLMS.y / srcLMS.y, 0, 0),
            make_float4(0, 0, destLMS.z / srcLMS.z, 0),
            make_float4(0, 0, 0, 1)
        }
    };

    return XYZToLMS * LMSCorrect * LMSToXYZ;
}


struct DenseSpectrum {
    /*Spectrum is sampled every 1nm from lambda_min to lambda_max*/
    float spectrum[SAMPLE_SIZE];
    float max;

    static __device__ DenseSpectrum Create(float const givenSpectrum[]) {
        int size = lambda_max - lambda_min + 1;
        DenseSpectrum s;
        for(int i=0; i<size; i++) {
            s.spectrum[i] = givenSpectrum[i];
        }
        s.calculateMax();
        return s;
    }

    float __forceinline__ __host__ __device__ sample(float const lambda) const {
        if (lambda > lambda_max || lambda < lambda_min) return 0;
        if (lambda == lambda_max) return spectrum[(int) (lambda_max - lambda_min)];
        int i = lambda - lambda_min;
        int floor_lambda = lambda;

        return spectrum[i] + (lambda - floor_lambda) * (spectrum[i + 1] - spectrum[i]);
    }

    __host__ __device__ void calculateMax() {
        max = spectrum[0];
        for(int i=1; i<SAMPLE_SIZE; i++) {
            if (spectrum[i] > max) max = spectrum[i];
        }
    }

    __host__ __device__ void normalize() {
        for(int i=0; i<SAMPLE_SIZE; i++) {
            spectrum[i] = spectrum[i] / max;
        }
        max = 1.0f;
    }

    float __forceinline__ __host__ __device__ operator()(float const lambda) const {
        return sample(lambda);
    }

    __device__ void print() const {

        for(int i=0; i<SAMPLE_SIZE; i++) {
            printf("wl = %f, s = %f\n", lambda_min + i, spectrum[i]);
        }

    }

    __host__ __device__ __forceinline__ DenseSpectrum operator+ (DenseSpectrum const & other_spectrum) const {
        int size = lambda_max - lambda_min + 1;
        DenseSpectrum s;
        for(int i=0; i<size; i++) {
            s.spectrum[i] = spectrum[i] + other_spectrum.spectrum[i];
        }
        return s; 
    }

    inline float __host__ __device__ getMax() { return max; }

    XYZColor toXYZ() const;

};

struct UniformSpectrum {
    float value;

    static __device__ UniformSpectrum Create(float value) {
        UniformSpectrum s;
        s.value = value;
        return s;
    }

    float __host__ __device__ sample(float const lambda) const {
        if (lambda < lambda_min || lambda > lambda_max) return 0;
        return value;
    }

    __host__ __device__ void normalize() {
        value = 1;
    }

    float __host__ __device__ operator()(float const lambda) const {
        return sample(lambda);
    }

    __host__ __device__ DenseSpectrum toDenseSpectrum() const {
        DenseSpectrum s;
        for(int i=0; i<SAMPLE_SIZE; i++) {
            s.spectrum[i] = sample(i + lambda_min);
        }
        s.calculateMax();
        return s;
    }

    XYZColor toXYZ() const;

};

struct SampledSpectrum {
    float spectrum[SAMPLE_SIZE];
    float lambda_values[SAMPLE_SIZE];
    int numberOfSamples;
    float max;

    static __device__ SampledSpectrum Create(float const givenSpectrum[], float const lambdas[], int size) {
        /*
            givenSpectrum and lambdas must both be of size size and lambdas must be sorted and all its values within the range [lambda_min, lambda_max]
        */
        SampledSpectrum s;
        s.numberOfSamples = size;
        //If lambda_min or lambda_max samples are not given, we set them to extend the border values by default
        if (lambdas[0] != lambda_min) s.numberOfSamples++;
        if (lambdas[size - 1] != lambda_max) s.numberOfSamples++;
        

        if (lambdas[0] != lambda_min) {
            s.lambda_values[0] = lambda_min;
            s.spectrum[0] = givenSpectrum[0];
            for(int i=0; i<min(size, SAMPLE_SIZE-1); i++) {
                s.spectrum[i+1] = givenSpectrum[i];
                s.lambda_values[i+1] = lambdas[i];
            }
        } else {
            for(int i=0; i<size; i++) {
                s.spectrum[i] = givenSpectrum[i];
                s.lambda_values[i] = lambdas[i];
            }
        }
        if (lambdas[size - 1] != lambda_max) {
            s.lambda_values[s.numberOfSamples - 1] = lambda_max;
            s.spectrum[s.numberOfSamples - 1] = givenSpectrum[size - 1];
        }
        s.calculateMax();
        return s;
    }

    __host__ __device__ void calculateMax() {
        max = spectrum[0];
        for(int i=1; i<numberOfSamples; i++) {
            if (spectrum[i] > max) max = spectrum[i];
        }
    }

    __host__ __device__ void normalize() {
        for(int i=0; i<numberOfSamples; i++) {
            spectrum[i] = spectrum[i] / max;
        }
        max = 1.0f;
    }


    __host__ __device__ float sample(float const lambda) const {
        if (lambda > lambda_max || lambda < lambda_min) return 0;
        if (fabsf(lambda - lambda_min) < 1e-3) return spectrum[0];
        
        for(int i=0; i<numberOfSamples; i++) {
            if (lambda_values[i] >= lambda) {
                return spectrum[i-1] + (lambda - lambda_values[i-1]) / (lambda_values[i] - lambda_values[i-1]) * (spectrum[i] - spectrum[i-1]);                
            }
        }  

        return -1;
    }

    float __host__ __device__ operator()(float const lambda) const {
        return sample(lambda);
    }

    inline float __host__ __device__ getMax() { return max; }

    __device__ void print() const {
        printf("Number of vals = %d\n", numberOfSamples);

        for(int i=0; i<numberOfSamples; i++) {
            printf("wl = %f, s = %f\n", lambda_values[i], spectrum[i]);
        }

    }


    __device__ __host__ DenseSpectrum toDenseSpectrum() {
        DenseSpectrum s;
        for(int i=0; i<SAMPLE_SIZE; i++) {
            s.spectrum[i] = sample(i + lambda_min);
        }
        s.calculateMax();
        return s;
    }

    XYZColor toXYZ() const;

};

struct RegularlySampledSpectrum {
    float spectrum[SAMPLE_SIZE];
    int size;
    float step;
    float max;

    static __device__ RegularlySampledSpectrum Create(float const givenSpectrum[], int size) {
        /*
            givenSpectrum and lambdas must both be of size size and lambdas must be sorted and all its values within the range [lambda_min, lambda_max]
        */
        RegularlySampledSpectrum s;
        s.size = size;
        s.step = ((float) SAMPLE_SIZE) / (size - 1);
        

        for(int i=0; i<size; i++) {
            s.spectrum[i] = givenSpectrum[i];
        }
        s.calculateMax();
        return s;
    }

    __host__ __device__ void calculateMax() {
        max = spectrum[0];
        for(int i=1; i<size; i++) {
            if (spectrum[i] > max) max = spectrum[i];
        }
    }

    __host__ __device__ void normalize() {
        for(int i=0; i<size; i++) {
            spectrum[i] = spectrum[i] / max;
        }
        max = 1.0f;
    }


    __host__ __device__ float sample(float const lambda) const {
        if (lambda > lambda_max || lambda < lambda_min) return 0;
        
        float slope = (lambda - lambda_min) / step;
        int i = slope;

        return spectrum[i] + (slope - i) * (spectrum[i+1] - spectrum[i]);
    }

    float __host__ __device__ operator()(float const lambda) const {
        return sample(lambda);
    }


    __host__ __device__ RegularlySampledSpectrum operator+(RegularlySampledSpectrum const& other) const {
        RegularlySampledSpectrum s;

        s.size = size;
        s.step = step;
        for(int i=0; i<size; i++) {
            s.spectrum[i] = spectrum[i] + other.spectrum[i];
        }
        s.calculateMax();
        return s;
    }

    inline float __host__ __device__ getMax() { return max; }

    __device__ void print() const {
        printf("Number of vals = %d\n", size);
        printf("Step = %f", step);

        for(int i=0; i<size; i++) {
            printf("wl = %f, s = %f\n", lambda_min + i*step, spectrum[i]);
        }

    }
};


struct DiscreteSpectrum {
    float spectrum[SAMPLE_SIZE];
    float lambda_values[SAMPLE_SIZE];
    int numberOfSamples;
    float max;

    static __device__ DiscreteSpectrum Create(float givenSpectrum[], float lambdas[], int size) {
        /*
            givenSpectrum and lambdas must both be of size size and lambdas must be sorted and all its values within the range [lambda_min, lambda_max]
        */
        DiscreteSpectrum s;
        s.numberOfSamples = size;
        
        for(int i=0; i<size; i++) {
            s.spectrum[i] = givenSpectrum[i];
            s.lambda_values[i] = lambdas[i];
        }
        s.calculateMax();
        return s;
    }

    __host__ __device__ void calculateMax() {
        max = spectrum[0];
        for(int i=1; i<numberOfSamples; i++) {
            if (spectrum[i] > max) max = spectrum[i];
        }
    }

    __host__ __device__ void normalize() {
        for(int i=0; i<numberOfSamples; i++) {
            spectrum[i] = spectrum[i] / max;
        }
        max = 1.0f;
    }
   

    float __host__ __device__ sample(float const lambda) const {
        for(int i=0; i<numberOfSamples; i++) {
            if (fabsf(lambda_values[i] - lambda) < 1e-3) {
                return spectrum[i];                
            }
        }  
        return 0;
    }

    float __host__ __device__ operator()(float const lambda) const {
        return sample(lambda);
    }

    inline float __host__ __device__ getMax() { return max; }

    XYZColor toXYZ() const;

};

enum SpectrumType {
    uniform,
    dense,
    sampled,
    regularlySampled,
    discrete,
};


__host__ __device__ struct Spectrum {

    unsigned int type;

    union {
        UniformSpectrum uniformSpectrum;
        DenseSpectrum denseSpectrum;
        SampledSpectrum sampledSpectrum;
        RegularlySampledSpectrum regularlySampledSpectrum;
        DiscreteSpectrum discreteSpectrum;
    };   
};



float innerProduct(UniformSpectrum const& f, DenseSpectrum const& g);
float innerProduct(DenseSpectrum const& f, DenseSpectrum const& g);
float innerProduct(SampledSpectrum const& f, DenseSpectrum const& g);
float innerProduct(DiscreteSpectrum const& f, DenseSpectrum const& g);


inline DenseSpectrum operator/(const DenseSpectrum& s, const float a) {
    DenseSpectrum other;
    for(int i=0; i<SAMPLE_SIZE; i++) {
        other.spectrum[i] = s.spectrum[i] / a; 
    }
    other.max = s.max / a;
    return other;
} 

inline SampledSpectrum operator/(const SampledSpectrum& s, const float a) {
    SampledSpectrum other;
    other.numberOfSamples = s.numberOfSamples;
    for(int i=0; i<s.numberOfSamples; i++) {
        other.lambda_values[i] = s.lambda_values[i];
        other.spectrum[i] = s.spectrum[i] / a; 
    }
    other.max = s.max / a;
    return other;
}

inline RegularlySampledSpectrum operator/(const RegularlySampledSpectrum& s, const float a) {
    RegularlySampledSpectrum other;
    other.size = s.size;
    other.step = s.step;
    for(int i=0; i<s.size; i++) {
        other.spectrum[i] = s.spectrum[i] / a; 
    }
    other.max = s.max / a;
    return other;
}



const float x_bar[] = {0.001368, 0.002236, 0.004243, 0.007650, 0.014310, 0.023190, 0.043510, 0.077630, 0.134380, 0.214770, 0.283900, 0.328500, 0.348280, 0.348060, 0.336200, 0.318700, 0.290800, 0.251100, 0.195360, 0.142100, 0.095640, 0.057950, 0.032010, 0.014700, 0.004900, 0.002400, 0.009300, 0.029100, 0.063270, 0.109600, 0.165500, 0.225750, 0.290400, 0.359700, 0.433450, 0.512050, 0.594500, 0.678400, 0.762100, 0.842500, 0.916300, 0.978600, 1.026300, 1.056700, 1.062200, 1.045600, 1.002600, 0.938400, 0.854450, 0.751400, 0.642400, 0.541900, 0.447900, 0.360800, 0.283500, 0.218700, 0.164900, 0.121200, 0.087400, 0.063600, 0.046770, 0.032900, 0.022700, 0.015840, 0.011359, 0.008111, 0.005790, 0.004109, 0.002899, 0.002049, 0.001440, 0.001000, 0.000690, 0.000476, 0.000332, 0.000235, 0.000166, 0.000117, 0.000083, 0.000059, 0.000042};
const float y_bar[] = {0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210, 0.002180, 0.004000, 0.007300, 0.011600, 0.016840, 0.023000, 0.029800, 0.038000, 0.048000, 0.060000, 0.073900, 0.090980, 0.112600, 0.139020, 0.169300, 0.208020, 0.258600, 0.323000, 0.407300, 0.503000, 0.608200, 0.710000, 0.793200, 0.862000, 0.914850, 0.954000, 0.980300, 0.994950, 1.000000, 0.995000, 0.978600, 0.952000, 0.915400, 0.870000, 0.816300, 0.757000, 0.694900, 0.631000, 0.566800, 0.503000, 0.441200, 0.381000, 0.321000, 0.265000, 0.217000, 0.175000, 0.138200, 0.107000, 0.081600, 0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920, 0.008210, 0.005723, 0.004102, 0.002929, 0.002091, 0.001484, 0.001047, 0.000740, 0.000520, 0.000361, 0.000249, 0.000172, 0.000120, 0.000085, 0.000060, 0.000042, 0.000030, 0.000021, 0.000015};
const float z_bar[] = {0.006450, 0.010550, 0.020050, 0.036210, 0.067850, 0.110200, 0.207400, 0.371300, 0.645600, 1.039050, 1.385600, 1.622960, 1.747060, 1.782600, 1.772110, 1.744100, 1.669200, 1.528100, 1.287640, 1.041900, 0.812950, 0.616200, 0.465180, 0.353300, 0.272000, 0.212300, 0.158200, 0.111700, 0.078250, 0.057250, 0.042160, 0.029840, 0.020300, 0.013400, 0.008750, 0.005750, 0.003900, 0.002750, 0.002100, 0.001800, 0.001650, 0.001400, 0.001100, 0.001000, 0.000800, 0.000600, 0.000340, 0.000240, 0.000190, 0.000100, 0.000050, 0.000030, 0.000020, 0.000010, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000};




namespace Spectra {
    inline DenseSpectrum CreateX_First() {
        DenseSpectrum x;
        for(float lambda=lambda_min; lambda <= lambda_max; lambda++) {
            x.spectrum[(int)(lambda - lambda_min)] = 0.366 * expf(-0.5 * powf((lambda - 446.8) / 19.44, 2));
        }
        x.calculateMax();
        return x;
    }

    inline DenseSpectrum CreateX() {
        DenseSpectrum x;
        for(float lambda=lambda_min; lambda <= lambda_max; lambda++) {
            x.spectrum[(int)(lambda - lambda_min)] = 1.065 * expf(-0.5 * powf((lambda - 595.8) / 33.33, 2)) + 0.366 * expf(-0.5 * powf((lambda - 446.8) / 19.44, 2));
        }
        x.calculateMax();
        return x;
    }
    inline DenseSpectrum CreateY() {
        DenseSpectrum y;
        for(float lambda=lambda_min; lambda <= lambda_max; lambda++) {
            y.spectrum[(int)(lambda - lambda_min)] = 1.014 * expf(-0.5 * powf((logf(lambda) - logf(556.3)) / 0.075, 2));
        }
        y.calculateMax();
        return y;
    }
    inline DenseSpectrum CreateZ() {
        DenseSpectrum x;
        for(float lambda=lambda_min; lambda <= lambda_max; lambda++) {
            x.spectrum[(int)(lambda - lambda_min)] = 1.839 * expf(-0.5 * powf((logf(lambda) - logf(449.8) / 0.051), 2));
        }
        x.calculateMax();
        return x;
    }

    inline float* buildWL(int n) {
        if (n > SAMPLE_SIZE) return NULL;

        float* wl = new float[n];
        for(int i=0; i<n; i++) {
            wl[i] = lambda_min + ((float)i / (n-1)) * (lambda_max - lambda_min);
        }
        return wl;
    }



    /*
        See the document: 
        Simple Analytics Approximations to the CIE XYZ Color Matching Functions
        by Chris Wyman, Peter-Pile Sloan, Peter Shirley, page 3.

        These functions are built upon approximations made by the CIE in 1964.
    */
    /*const DenseSpectrum X = Spectra::CreateX();
    const DenseSpectrum Y = Spectra::CreateY();
    const DenseSpectrum Z = Spectra::CreateZ();*/


    const auto X = RegularlySampledSpectrum::Create(x_bar, 81);
    const auto Y = RegularlySampledSpectrum::Create(y_bar, 81);
    const auto Z = RegularlySampledSpectrum::Create(z_bar, 81);
}

inline MatrixSpectrumToXYZ buildM() {
    MatrixSpectrumToXYZ M;
    for(int i=0; i<SAMPLE_SIZE; i++) {
        M(0,i) = Spectra::X(lambda_min + i)/CIE_Y_integral;
        M(1,i) = Spectra::Y(lambda_min + i)/CIE_Y_integral;
        M(2,i) = Spectra::Z(lambda_min + i)/CIE_Y_integral;
    }

    return M;
}




namespace Spectra {
    // XYZ_vector = M*Spectrum_vector | PBRT 4.23
    const MatrixSpectrumToXYZ M = buildM(); 

    // Spectrum_vector = M_inv * XYZ_vector
    const MatrixXYZToSpectrum M_inv = M.completeOrthogonalDecomposition().pseudoInverse();


    inline __device__ __host__ float sampleWavelength(float u, unsigned int lambda) {
        switch(lambda) {
            //case 0 : return sampleLogNormal(u, );

            //case 1 : return CIE_Y_integral * sampleNormal(u, 556.1, 46.14);

            //case 2 : return sampleLogNormal(u, , );

            default : return 0;
        }
    }


}




const float s_r2[]  = {0.328455134, 0.324039100, 0.315349590, 0.292792770, 0.246316933, 0.198108029, 0.130068113, 0.079657502, 0.047536766, 0.030925762, 0.023739245, 0.017858899, 0.014560638, 0.012790919, 0.011391265, 0.010621609, 0.010019665, 0.009843010, 0.010040448, 0.010026949, 0.009896261, 0.010490400, 0.009780279, 0.008394966, 0.007078490, 0.006339279, 0.005491672, 0.004880634, 0.004483955, 0.004185756, 0.004029708, 0.004096559, 0.004260582, 0.004472863, 0.004811227, 0.005409461, 0.006287819, 0.007615900, 0.009731549, 0.013081085, 0.019375748, 0.327707567, 0.538874667, 0.725699391, 0.951408718, 0.962637428, 0.966971579, 0.968007753, 0.967112589, 0.963775324, 0.958418605, 0.952601048, 0.944569777, 0.931722624, 0.913700042, 0.891598969, 0.860505987, 0.824225062, 0.773069484, 0.709647070, 0.642591809, 0.562173723, 0.490358422, 0.445050267, 0.414903338, 0.392213639, 0.375678397, 0.360661444, 0.349760473, 0.347712371, 0.339279547, 0.342893608, 0.338489141, 0.336913338, 0.341734672, 0.334898485, 0.334735839, 0.334317322, 0.333924136, 0.333359358, 0.333012936};
const float s_g2[]  = {0.330648932, 0.329150364, 0.326106934, 0.314306096, 0.285621445, 0.246483644, 0.176356833, 0.115924753, 0.072260822, 0.048849075, 0.038884781, 0.030676675, 0.026470494, 0.024892729, 0.024141373, 0.024877463, 0.026478710, 0.030160991, 0.038137818, 0.051247767, 0.089042464, 0.316275641, 0.447692245, 0.645266804, 0.827179315, 0.921339153, 0.942909702, 0.955503903, 0.963102866, 0.968404133, 0.972079059, 0.973896551, 0.975085761, 0.976143135, 0.976820260, 0.976783210, 0.976380637, 0.975499356, 0.973843427, 0.971305193, 0.966234176, 0.658916933, 0.449088083, 0.263712302, 0.038957672, 0.028189759, 0.023919124, 0.022538392, 0.022706962, 0.024634248, 0.027942452, 0.031543891, 0.036587085, 0.044729441, 0.056138794, 0.070030169, 0.089641578, 0.111148860, 0.141853788, 0.178303772, 0.212975249, 0.253865556, 0.283887105, 0.301548721, 0.311745496, 0.319037251, 0.323070178, 0.326947909, 0.329858486, 0.330452606, 0.330682210, 0.335519233, 0.330196327, 0.333260022, 0.331231570, 0.334947675, 0.331484024, 0.331632713, 0.331732129, 0.331951176, 0.332076988};
const float s_b2[]  = {0.340895932, 0.346810535, 0.358543476, 0.392901134, 0.468061621, 0.555408327, 0.693575054, 0.804417744, 0.880202412, 0.920225163, 0.937375974, 0.951464426, 0.958968868, 0.962316352, 0.964467362, 0.964500928, 0.963501626, 0.959995999, 0.951821734, 0.938725283, 0.901061275, 0.673233959, 0.542527476, 0.346338230, 0.165742195, 0.072321568, 0.051598626, 0.039615464, 0.032413179, 0.027410111, 0.023891233, 0.022006890, 0.020653657, 0.019384003, 0.018368514, 0.017807329, 0.017331544, 0.016884744, 0.016425024, 0.015613722, 0.014390077, 0.013375499, 0.012037251, 0.010588306, 0.009633610, 0.009172813, 0.009109297, 0.009453855, 0.010180450, 0.011590428, 0.013638943, 0.015855062, 0.018843138, 0.023547934, 0.030161164, 0.038370863, 0.049852435, 0.064626077, 0.085076729, 0.112049158, 0.144432942, 0.183960721, 0.225754473, 0.253401011, 0.273351165, 0.288749110, 0.301251423, 0.312390644, 0.320381036, 0.321835018, 0.330038236, 0.321587149, 0.331314518, 0.329826618, 0.327033724, 0.330153783, 0.333780041, 0.334049853, 0.334343601, 0.334689271, 0.334909795};


inline __device__ SampledSpectrum RGBtoSampledSpectrum(float3 rgb) {
    SampledSpectrum s;
    s.numberOfSamples = 81;
    
    for(int i=0; i<s.numberOfSamples; i++) {
        s.lambda_values[i] = lambda_min + ((float)i / (s.numberOfSamples - 1)) * (lambda_max - lambda_min);
        s.spectrum[i] = rgb.x * s_r2[i] + rgb.y * s_g2[i] + rgb.z * s_b2[i];
    }
    s.calculateMax();
    return s;
}
inline __device__ DenseSpectrum RGBtoDenseSpectrum(float3 rgb) {
    return RGBtoSampledSpectrum(rgb).toDenseSpectrum();
}


inline DenseSpectrum __device__ XYZColor::toSpectrum()  {
        //Algorithm converting some color coordinates to a spectrum using the XYZ matrix's pseudo-inverse
        using Spectra::M_inv;
        float3 XYZ = make_float3(x, y, z);

        float spectrum_vector[SAMPLE_SIZE];
        for(int i=0; i<SAMPLE_SIZE; i++) {
            float3 row = make_float3(M_inv(i, 0), M_inv(i, 1), M_inv(i, 2));
            spectrum_vector[i] = fabsf(dot(row, XYZ));
        }
        DenseSpectrum s =  DenseSpectrum::Create(spectrum_vector);
        return s;
}
