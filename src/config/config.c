#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cJSON.h"

Configuration* load_config(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Could not open config file %s\n", path);
        return NULL;
    }

    Configuration *cfg = malloc(sizeof(Configuration));

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);

    char *text = malloc(size + 1);
    fread(text, 1, size, fp);
    text[size] = '\0';
    fclose(fp);

    cJSON *json = cJSON_Parse(text);
    free(text);

    if (!json) {
        fprintf(stderr, "Error: Could not parse JSON in config file %s\n", path);
        return NULL;
    }

    cJSON *sim = cJSON_GetObjectItem(json, "simulation");
    if (!sim) { 
        fprintf(stderr, "Error: Could not find 'simulation' object in config file %s\n", path);
        cJSON_Delete(json); 
        return NULL; 
    }

    const char *tmp = cJSON_GetObjectItem(sim, "file")->valuestring;
    cfg->file = malloc(strlen(tmp) + 1);
    if(cfg->file) strcpy(cfg->file, tmp);
    cfg->dx = cJSON_GetObjectItem(sim, "dx")->valuedouble;
    cfg->dt = cJSON_GetObjectItem(sim, "dt")->valuedouble;
    cfg->alpha = cJSON_GetObjectItem(sim, "alpha")->valuedouble;
    cfg->iterations = cJSON_GetObjectItem(sim, "iterations")->valueint;
    cJSON_Delete(json);

    printf("Configuration loaded from %s\n", path);
    return cfg;
}