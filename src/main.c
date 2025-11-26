#include <stdio.h>
#include <stdlib.h>
#include "pgm_utils.h"
#include "crank_nicolson.h"

// ./cn file.pgm dx dt alpha iterations
int main(int argc, char *argv[]) {
    if(argc != 6){
        fprintf(stderr, "Wrong argument\n");
        exit(EXIT_FAILURE);
    }

    int width, height, *img;
    if((img = pgm_loader(argv[1], &width, &height)) == NULL) exit(EXIT_FAILURE);
    float* norm_img = pgm_normalisation(img, width * height);

    if(width != height) {
        fprintf(stderr, "This version of the Crank-Nicolson solver only works with square heatmaps");
        return 1;
    }

    CrankNicolsonSetup *solver = setup(width * width, atof(argv[2]), atof(argv[3]), atof(argv[4]), norm_img);
    run(solver, atoi(argv[5]));
    free_solver(solver);

    return EXIT_SUCCESS;
}