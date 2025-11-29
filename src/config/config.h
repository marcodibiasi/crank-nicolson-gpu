#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>
#include <stdlib.h>

typedef struct{
    char* file;
    float dx;
    float dt;
    float alpha;
    int iterations;
} Configuration;

Configuration* load_config(const char *path);

#endif // CONFIG_H