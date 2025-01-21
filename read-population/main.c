#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ReadPopulation(char *fname) {
    FILE *dump;
    int i, np;
    double *weight, *xcoord, *ycoord;

    dump = fopen(fname, "r");

    if (dump == NULL) {
        fprintf(stderr, "Error read open file %s\n", fname);
        return (1);
    }

    fread(&np, sizeof((int) 1), 1, dump);
    printf("\n%d particles found\n", np);

    weight = malloc(np * sizeof((double) 1.0));
    fread(weight, sizeof((double) 1.0), np, dump);

    xcoord = malloc(np * sizeof((double) 1.0));
    fread(xcoord, sizeof((double) 1.0), np, dump);

    ycoord = malloc(np * sizeof((double) 1.0));
    fread(ycoord, sizeof((double) 1.0), np, dump);

    for (i = 0; i < np; i++) {
        fprintf(stdout, "Particle %6.6d: x, y, weight = %lf, %lf, %lf\n", i, xcoord[i], ycoord[i], weight[i]);
    }

    fclose(dump);
    free(weight);
    free(xcoord);
    free(ycoord);

    return 0;
}

int main(int argc, char *argv[]) {
    char fname[256];

    if (argc < 2) {
        fprintf(stderr, "To be launched as: ReadPopulation <Population????.dmp>\n");
        return (1);
    }

    strcpy(fname, argv[1]);
    printf("Reading file %s ...\n", fname);

    return ReadPopulation(fname);
}
