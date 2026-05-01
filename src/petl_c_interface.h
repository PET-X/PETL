
#ifdef WIN32
    #pragma once

    #ifdef PROJECTOR_EXPORTS
        #define PROJECTOR_API __declspec(dllexport)
    #else
        #define PROJECTOR_API __declspec(dllimport)
    #endif
#else
    #define PROJECTOR_API
#endif

extern "C" PROJECTOR_API bool set_model(int);
extern "C" PROJECTOR_API int create_new_model();
extern "C" PROJECTOR_API void about();

extern "C" PROJECTOR_API bool copy_parameters(int);

extern "C" PROJECTOR_API void clearAll();

extern "C" PROJECTOR_API void print_parameters();
extern "C" PROJECTOR_API bool volume_defined();
extern "C" PROJECTOR_API bool geometry_defined();

extern "C" PROJECTOR_API int get_numX();
extern "C" PROJECTOR_API bool set_numX(int);
extern "C" PROJECTOR_API int get_numY();
extern "C" PROJECTOR_API bool set_numY(int);
extern "C" PROJECTOR_API int get_numZ();
extern "C" PROJECTOR_API bool set_numZ(int);

extern "C" PROJECTOR_API float get_voxelWidth();
extern "C" PROJECTOR_API bool set_voxelWidth(float);
extern "C" PROJECTOR_API float get_voxelHeight();
extern "C" PROJECTOR_API bool set_voxelHeight(float);

extern "C" PROJECTOR_API float get_offsetX();
extern "C" PROJECTOR_API bool set_offsetX(float);
extern "C" PROJECTOR_API float get_offsetY();
extern "C" PROJECTOR_API bool set_offsetY(float);
extern "C" PROJECTOR_API float get_offsetZ();
extern "C" PROJECTOR_API bool set_offsetZ(float);

extern "C" PROJECTOR_API int get_numPlanograms();
extern "C" PROJECTOR_API bool get_planogramSize(int which, int* shape);

extern "C" PROJECTOR_API bool add_planogram(float psi, float R, float L, float H, float v_m0, float v_m1, float T);
extern "C" PROJECTOR_API bool remove_planogram(int);
extern "C" PROJECTOR_API bool keep_only_planogram(int);

extern "C" PROJECTOR_API bool set_default_volume(float scale);
extern "C" PROJECTOR_API bool set_volume(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

extern "C" PROJECTOR_API bool bin(float** g, char* file_name);

extern "C" PROJECTOR_API bool simulate_scatter(float** g, float* mu);

extern "C" PROJECTOR_API bool project(float** g, float* f);
extern "C" PROJECTOR_API bool backproject(float** g, float* f);
extern "C" PROJECTOR_API bool FBP(float** g, float* f);
extern "C" PROJECTOR_API bool PFDR(float** g, float** g_reb);

extern "C" PROJECTOR_API bool add_object(int type, float* c, float* r, float val, float* A, float* clip);
extern "C" PROJECTOR_API bool add_module(int type, float* c, float* r, float val, float* A, float* clip);
extern "C" PROJECTOR_API bool clear_phantom();
extern "C" PROJECTOR_API bool clear_modules();
extern "C" PROJECTOR_API bool scale_phantom(float, float, float);
extern "C" PROJECTOR_API bool voxelize(float* f, int oversampling);
extern "C" PROJECTOR_API bool ray_trace(float** g, int oversampling);
extern "C" PROJECTOR_API bool stopping_power(float** g, int oversampling);
extern "C" PROJECTOR_API bool set_solid_angle_correction(float** g, bool do_inverse);
extern "C" PROJECTOR_API bool apply_corrections(float** g, float** r, float threshold);

extern "C" PROJECTOR_API bool multiply3D(float* x, float* y, int N_1, int N_2, int N_3);
extern "C" PROJECTOR_API bool multiply4D(float* x, float* y, int N_1, int N_2, int N_3, int N_4);

extern "C" PROJECTOR_API bool divide3D(float* x, float* y, int N_1, int N_2, int N_3);
extern "C" PROJECTOR_API bool divide4D(float* x, float* y, int N_1, int N_2, int N_3, int N_4);

extern "C" PROJECTOR_API bool rdivide3D(float* num, float* denom, int N_1, int N_2, int N_3);
extern "C" PROJECTOR_API bool rdivide4D(float* num, float* denom, int N_1, int N_2, int N_3, int N_4);

extern "C" PROJECTOR_API bool reciprocal3D(float* x, int N_1, int N_2, int N_3, float divide_by_zero_value);
extern "C" PROJECTOR_API bool reciprocal4D(float* x, int N_1, int N_2, int N_3, int N_4, float divide_by_zero_value);

extern "C" PROJECTOR_API float inner_product3D(float* x, float* y, float* w, int N_1, int N_2, int N_3);
extern "C" PROJECTOR_API float inner_product4D(float* x, float* y, float* w, int N_1, int N_2, int N_3, int N_4);

// Anisotropic Total Variation for 3D data
extern "C" PROJECTOR_API float TV_cost(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu);
extern "C" PROJECTOR_API bool TV_gradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu);
extern "C" PROJECTOR_API float TV_quadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu);
extern "C" PROJECTOR_API bool diffuse(float* f, int N_1, int N_2, int N_3, float delta, float p, int numIter, bool data_on_cpu);
extern "C" PROJECTOR_API bool TV_denoise(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, int numIter, bool doMean, bool data_on_cpu);
extern "C" PROJECTOR_API bool relativeDifferences_gradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta);
extern "C" PROJECTOR_API bool gaussian_filter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims);