#ifndef PGM_UTILS_H
#define PGM_UTILS_H

int *pgm_loader(const char *filename, int *width, int *height);
float *pgm_normalisation(int* matrix, int n);
unsigned char *pgm_denormalisation(float* matrix, int n);
void pgm_save(const char *filename, unsigned char* vec, int size);
int png_save(const char *filename, unsigned char* vec, int size, int verbose);
unsigned char* get_colormap(unsigned char* vec, int size, float gamma);

#endif
