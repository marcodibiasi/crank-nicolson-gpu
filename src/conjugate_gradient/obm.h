#ifndef OBM_H
#define OBM_H

// Offsetted banded matrix representation

typedef struct {
    int rows;
    int* offset;
    float* values;
    int non_zero_values;
} OBMatrix;

#endif // OBM_H