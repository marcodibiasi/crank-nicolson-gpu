#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pgm_utils.h"
#include "crank_nicolson.h"
#include "config.h"
#include "flags.h"
#include "profiler.h"

Flags parse_args(int argc, char* argv[]);

int main(int argc, char *argv[]) {
    if(argc < 2) {
        fprintf(stderr, "Usage: config.json [--show-energy]" 
                " [--profile] [--verbose] [--progress] [--delta-save=<n>]\n");
        fflush(stderr);
        return EXIT_FAILURE;
    }

    // HANDLE FLAGS
    Flags flags = parse_args(argc, argv);

    if(flags.progress == 1){
        if(flags.show_energy == 1 || flags.profile == 1 || flags.verbose == 1) {
            fprintf(stderr, "Flag --progress must be called alone\n");
            return EXIT_FAILURE;
        }
    }
    
    // HANDLE CONFIGURATION
    Configuration* cfg = load_config(argv[1]);


    //HANDLE PROFILER 
    Profiler *p = NULL;
    if(flags.profile == 1) {
        p = malloc(sizeof(Profiler));
        profiler_init(p, cfg->iterations);
    }

    // INITIALIZE AND RUN SOLVER
    int width, height, *img;
    if((img = pgm_loader(cfg->file, &width, &height)) == NULL) exit(EXIT_FAILURE);
    float* norm_img = pgm_normalisation(img, width * height);

    if(width != height) {
        fprintf(stderr, "This version of the Crank-Nicolson solver only works with square heatmaps");
        return EXIT_FAILURE;
    }

    CrankNicolsonSetup *solver = setup(width * width, cfg->dx, cfg->dt, cfg->alpha, norm_img);
    run(solver, cfg->iterations, &flags, p);
    if(flags.profile) save_profiler_json(p, "data/output/profiler.json");;

    // CLEAN UP
    if (solver) free_solver(solver);
    if (img) free(img);
    if (cfg->file) free(cfg->file);
    if (cfg) free(cfg);

    return EXIT_SUCCESS;
}

Flags parse_args(int argc, char* argv[]){
    Flags flags = {0};
    flags.delta_save = 1;

    for(int i = 2; i < argc; i++) {
        if(strcmp(argv[i], "--show-energy") == 0) {
            flags.show_energy = 1;
        }
        else if(strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--profile") == 0) {
            flags.profile = 1;
        }
        else if(strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            flags.verbose = 1;
        }
        else if(strcmp(argv[i], "--progress") == 0) {
            flags.progress = 1;
        }
        else if(strncmp(argv[i], "--delta-save=", 13) == 0){
            flags.delta_save = atoi(argv[i] + 13);
        }
    }

    return flags;
}
