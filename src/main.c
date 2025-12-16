#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pgm_utils.h"
#include "crank_nicolson.h"
#include "config.h"
#include "flags.h"

// ./cn config.json
int main(int argc, char *argv[]) {
    if(argc < 2) {
        fprintf(stderr, "Usage: config.json [-e show_energy] [-p profile] [-v verbose] [--percentage]\n");
        fflush(stderr);
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
        else if(strcmp(argv[i], "--progress") == 0) {
            flags.progress = 1;
        }
    }

    if(flags.progress == 1){
        if(flags.show_energy == 1 || flags.profile == 1 || flags.verbose == 1) {
            fprintf(stderr, "Flag --progress must be called alone\n");
            return EXIT_FAILURE;
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
    if (solver) free_solver(solver);
    if (img) free(img);
    if (cfg->file) free(cfg->file);
    if (cfg) free(cfg);

    return EXIT_SUCCESS;
}
