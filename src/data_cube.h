#ifndef __DATA_CUBE_H
#define __DATA_CUBE_H

#ifdef WIN32
#pragma once
#endif

class dataCube
{
public:
    dataCube();
    dataCube(float*, int, int, int);
    dataCube(float*, int, int, int, int);
    ~dataCube();

    void clearAll();
    bool init(float* data_in, int N_1_in, int N_2_in, int N_3_in, float T_1_in = 1.0, float T_2_in = 1.0, float T_3_in = 1.0, float first_1_in = 0.0, float first_2_in = 0.0, float first_3_in = 0.0);
    bool init(float* data_in, int N_1_in, int N_2_in, int N_3_in, int N_4_in, float T_1_in = 1.0, float T_2_in = 1.0, float T_3_in = 1.0, float T_4_in = 1.0, float first_1_in = 0.0, float first_2_in = 0.0, float first_3_in = 0.0, float first_4_in = 0.0);

    int N_1, N_2, N_3, N_4;
    float T_1, T_2, T_3, T_4;
    float first_1, first_2, first_3, first_4;
    float* data;

    bool dimensionsMatch(dataCube*);

    enum BINARY_OPERATIONS {NONE=0,ADD,SUB,MULT,DIV,RDIV,SADD};
    dataCube* binary_operation(dataCube*, int which, float other = 0.0);
    dataCube* add(dataCube*);
    dataCube* sub(dataCube*);
    dataCube* multiply(dataCube*);
    dataCube* divide(dataCube*);
    dataCube* rdivide(dataCube*);
    dataCube* scalarAdd(float, dataCube*);

    enum UNITARY_OPERATIONS {SCALE=1,EXPONENTIAL,LOGARITHM,RECIPROCAL,CLIP,CONSTANT,EXPNEG,NEGLOG};
    dataCube* unitary_operation(int which, float other = 0.0);
    dataCube* scale(float);
    dataCube* exponential();
    dataCube* logarithm(float negative_value = 12.0);
    dataCube* reciprocal(float divide_by_zero_value = 0.0);
    dataCube* clip(float low = 0.0);
    dataCube* set_constant(float val = 0.0);
    dataCube* expNeg();
    dataCube* negLog(float negative_value = 12.0);
    
    float innerProduct(dataCube* x, dataCube* w = nullptr);
    float sum();
//private:
    int numDims;
};

#endif
