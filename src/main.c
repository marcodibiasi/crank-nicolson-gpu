#include <stdio.h>
#include <stdlib.h>
#include "pgm_utils.h"
#include "crank_nicolson.h"
#include "config.h"
#include "flags.h"

// ./cn config.json
int main(int argc, char *argv[]) {
    if(argc < 2) {
        fprintf(stderr, "Usage: config.json [-E]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // HANDLE CONFIGURATION AND FLAGS
    Configuration* cfg = load_config(argv[1]);

    Flags flags = {0};
    for(int i = 2; i < argc; i++) {
        if(strcmp(argv[i], "-e") == 0) {
            flags.show_energy = 1;
        }
        else if(strcmp(argv[i], "-p") == 0) {
            flags.profile = 1;
        }
        else if(strcmp(argv[i], "-v") == 0) {
            flags.verbose = 1;
        }
    }

    // INITIALIZE AND RUN SOLVER
    int width, height, *img;
    if((img = pgm_loader(cfg->file, &width, &height)) == NULL) exit(EXIT_FAILURE);
    float* norm_img = pgm_normalisation(img, width * height);

    if(width != height) {
        fprintf(stderr, "This version of the Crank-Nicolson solver only works with square heatmaps");
        return 1;
    }

    CrankNicolsonSetup *solver = setup(width * width, cfg->dx, cfg->dt, cfg->alpha, norm_img);
    run(solver, cfg->iterations, &flags);

    // CLEAN UP
    free_solver(solver);
    free(img);
    free(cfg->file);
    free(cfg);

    return EXIT_SUCCESS;
}