/*   
!                          Program Particles 2024
!  mimics the behaviour of a system of particles affected by mutual forces.
!
!  Final application of the Course "Parallel Computing using MPI and OpenMP"
!
!  This program is meant to be used by course participants for demonstrating
!  the abilities acquired in optimizing and parallelising programs.
!
!  The techniques learnt at the course should be extensively used in order to
!  improve the program response times as much as possible, while obtaining
!  the same or very closed results, stored in the files '*.dmp'
!
!  The code implemented herewith has been written for course exercise only,
!  therefore source code, algorithms and produced results must not be 
!  trusted nor used for anything different from their original purpose. 
!  
!  Description of the program:
!  a squared grid is hit by a field whose result is the distribution of particles
!  with different properties.
!  After having been created the particles move under the effect of mutual
!  forces.
!
!  If any, would you please send your comments to m.cremonesi@cineca.it 
!
!  Program outline:
!  1 - the program starts reading the initial values (InitGrid)
!  2 - the generating field is computed (GeneratingField)
!  3 - the set of created particles is computed (ParticleGeneration)
!  4 - the evolution of the system of particles is computed (SystemEvolution)
!
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

// ----------------------------------------------------------------------------
// Utilities.
// ----------------------------------------------------------------------------

// Whether the ImageMagick tool is available or not.
// #define IMAGEMAGICK

// Element position in a 2D array.
#define index2D(x, y, width) ((x) + (y) * (width))

#define EXIT(code) MPI_Abort(MPI_COMM_WORLD, code)

// ----------------------------------------------------------------------------
// Data types.
// ----------------------------------------------------------------------------

struct i2dGrid {
    int width, height;                  // Extensions in X and Y directions
    double xBegin, xEnd, yBegin, yEnd;  // Initial and final value for X and Y directions
    int *values;                        // 2D matrix of values
};

struct particle {
    double weight, x, y, vx, vy, fx, fy;
};

struct population {
    int amount;
    double *weight;     // Weigths
    double *x, *y;      // Positions
    double *vx, *vy;    // Velocities
};

// ----------------------------------------------------------------------------
// Global variables.
// ----------------------------------------------------------------------------

// MPI.
int procId, nProcs;

// Configuration.
int MaxIters, MaxSteps;
double timeStep;

// Data.
struct i2dGrid GenFieldGrid, ParticleGrid;
struct population particles;

// ----------------------------------------------------------------------------
// Function prototypes.
// ----------------------------------------------------------------------------

/// Get the length of a row of the configuration file.
int rowLength(char *row);

/// Read a row from the configuration file.
int readRow(char *rg, int nc, FILE *daleg);

void print_i2dGrid(struct i2dGrid g);

void print_particle(struct particle p);

void print_Population(struct population p);

/// Save population values on file.
void DumpPopulation(struct population p, int t);

/// Write a file with statistics on population.
void ParticleStats(struct population p, int t);

/// Load the configuration from a file and initialize the grids.
void readConfiguration(char *inputFile);

/// Distribute a particle population in a grid for visualization purposes.
void ParticleScreen(struct i2dGrid *particleGrid, struct population population, int s);

/// Dump double data with fixed min & max values in a PPM format.
/// If the ImageMagick tool is available, it also converts the .ppm files in .jpg images.
void IntVal2ppm(int width, int height, int *idata, int *vmin, int *vmax, char *name);

/// Get the maximum value from an array of integers.
int MaxIntVal(int s, int *a);

/// Get the minimum value from an array of integers.
int MinIntVal(int s, int *a);

/// Get the maximum value from an array of doubles.
double MaxDoubleVal(int s, double *a);

/// Get the minimum value from an array of doubles.
double MinDoubleVal(int s, double *a);

/// Set the parameters of a given particle.
void configParticle(struct particle *p, double weight, double x, double y, double vx, double vy);

void GeneratingField(struct i2dGrid *grid, int maxIterations);

void ParticleGeneration(struct i2dGrid grid, struct i2dGrid pgrid, struct population *population);

void SystemEvolution(struct i2dGrid *pgrid, struct population *population, int numSteps);

/// Compute the forces acting on p1 by p1-p2 interactions.
/// The force is computed using the inverse-square law of gravitational attraction: F = k * m1 * m2 / d^2.
void computeForces(double *f, struct particle p1, struct particle p2);

/// Compute the effects of forces on particles in a interval time.
/// x(t + dt) = x(t) + v(t)*dt + a(t)*dt^2/2.
/// v(t + dt) = v(t) + a(t)*dt.
void applyForces(struct population *p, int beginParticle, int endParticle, double *forces);

/// Gather the particles data computed by each process.
void gatherParticles(struct population *population, int *particlesPerProcess, int *particlesDisplacements);

// ----------------------------------------------------------------------------
// Function implementations.
// ----------------------------------------------------------------------------

void configParticle(struct particle *p, double weight, double x, double y, double vx, double vy) {
    p->weight = weight;
    p->x = x;
    p->y = y;
    p->vx = vx;
    p->vy = vy;
}

void computeForces(double *forces, struct particle p1, struct particle p2) {
    double k = 0.001;
    double tiny = (double) 1.0 / (double) 1000000.0;

    // Compute the (squared) distance between the two particles.
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double d2 = dx * dx + dy * dy;

    if (d2 < tiny) {
        // Avoid the case in which particles get in touch.
        d2 = tiny;
    }

    double force = (k * p1.weight * p2.weight) / d2;
    forces[0] = force * dx / sqrt(d2);
    forces[1] = force * dy / sqrt(d2);
}

void applyForces(struct population *p, int beginParticle, int endParticle, double *forces) {
    #pragma omp parallel for default(none) shared(timeStep, p, forces, beginParticle, endParticle)
    for (int i = beginParticle; i < endParticle; i++) {
        p->x[i] = p->x[i] + (p->vx[i] * timeStep) + (0.5 * forces[index2D(0, i - beginParticle, 2)] * timeStep * timeStep / p->weight[i]);
        p->vx[i] = p->vx[i] + forces[index2D(0, i - beginParticle, 2)] * timeStep / p->weight[i];

        p->y[i] = p->y[i] + (p->vy[i] * timeStep) + (0.5 * forces[index2D(1, i - beginParticle, 2)] * timeStep * timeStep / p->weight[i]);
        p->vy[i] = p->vy[i] + forces[index2D(1, i - beginParticle, 2)] * timeStep / p->weight[i];
    }
}

void gatherParticles(struct population *population, int *particlesPerProcess, int *particlesDisplacements) {
    MPI_Allgatherv(population->weight + particlesDisplacements[procId], particlesPerProcess[procId], MPI_DOUBLE,
                   population->weight, particlesPerProcess, particlesDisplacements, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    MPI_Allgatherv(population->x + particlesDisplacements[procId], particlesPerProcess[procId], MPI_DOUBLE,
                   population->x, particlesPerProcess, particlesDisplacements, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    MPI_Allgatherv(population->y + particlesDisplacements[procId], particlesPerProcess[procId], MPI_DOUBLE,
                   population->y, particlesPerProcess, particlesDisplacements, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    MPI_Allgatherv(population->vx + particlesDisplacements[procId], particlesPerProcess[procId], MPI_DOUBLE,
                   population->vx, particlesPerProcess, particlesDisplacements, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    MPI_Allgatherv(population->vy + particlesDisplacements[procId], particlesPerProcess[procId], MPI_DOUBLE,
                   population->vy, particlesPerProcess, particlesDisplacements, MPI_DOUBLE,
                   MPI_COMM_WORLD);
}

void readConfiguration(char *InputFile) {
    if (procId == 0) {
        char row[80];

        fprintf(stdout, "Initializing grids ...\n");

        FILE *f = fopen(InputFile, "r");

        if (!f) {
            fprintf(stderr, "Error read access to file %s\n", InputFile);
            EXIT(-1);
        }

        // Now read measured values; they are read in the following order:
        // GenFieldGrid.EX, GenFieldGrid.EY,
        // GenFieldGrid.Xs, GenFieldGrid.Xe, GenFieldGrid.Ys, GenFieldGrid.Ye
        // ParticleGrid.Xs, ParticleGrid.Xe, ParticleGrid.Ys, ParticleGrid.Ye

        int nv = 0;
        int iv = 0;
        double dv = 0.0;

        while (1) {
            if (readRow(row, 80, f) < 1) {
                fprintf(stderr, "Error reading input file\n");
                EXIT(-1);
            }

            if (row[0] == '#') {
                continue;
            }

            if (nv <= 0) {
                if (sscanf(row, "%d", &iv) < 1) {
                    fprintf(stderr, "Error reading EX from string\n");
                    EXIT(-1);
                }

                GenFieldGrid.width = iv;
                nv = 1;
                continue;
            }

            if (nv == 1) {
                if (sscanf(row, "%d", &iv) < 1) {
                    fprintf(stderr, "Error reading EY from string\n");
                    EXIT(-1);
                }

                GenFieldGrid.height = iv;
                nv++;
                continue;
            }

            if (nv == 2) {
                if (sscanf(row, "%lf", &dv) < 1) {
                    fprintf(stderr, "Error reading GenFieldGrid.Xs from string\n");
                    EXIT(-1);
                }

                GenFieldGrid.xBegin = dv;
                nv++;
                continue;
            }

            if (nv == 3) {
                if (sscanf(row, "%lf", &dv) < 1) {
                    fprintf(stderr, "Error reading GenFieldGrid.Xe from string\n");
                    EXIT(-1);
                }

                GenFieldGrid.xEnd = dv;
                nv++;
                continue;
            }

            if (nv == 4) {
                if (sscanf(row, "%lf", &dv) < 1) {
                    fprintf(stderr, "Error reading GenFieldGrid.Ys from string\n");
                    EXIT(-1);
                }

                GenFieldGrid.yBegin = dv;
                nv++;
                continue;
            }

            if (nv == 5) {
                if (sscanf(row, "%lf", &dv) < 1) {
                    fprintf(stderr, "Error reading GenFieldGrid.Ye from string\n");
                    EXIT(-1);
                }

                GenFieldGrid.yEnd = dv;
                nv++;
                continue;
            }

            if (nv <= 6) {
                if (sscanf(row, "%d", &iv) < 1) {
                    fprintf(stderr, "Error reading ParticleGrid.EX from string\n");
                    EXIT(-1);
                }

                ParticleGrid.width = iv;
                nv++;
                continue;
            }

            if (nv == 7) {
                if (sscanf(row, "%d", &iv) < 1) {
                    fprintf(stderr, "Error reading ParticleGrid.EY from string\n");
                    EXIT(-1);
                }

                ParticleGrid.height = iv;
                nv++;
                continue;
            }

            if (nv == 8) {
                if (sscanf(row, "%lf", &dv) < 1) {
                    fprintf(stderr, "Error reading ParticleGrid.Xs from string\n");
                    EXIT(-1);
                }

                ParticleGrid.xBegin = dv;
                nv++;
                continue;
            }

            if (nv == 9) {
                if (sscanf(row, "%lf", &dv) < 1) {
                    fprintf(stderr, "Error reading ParticleGrid.Xe from string\n");
                    EXIT(-1);
                }

                ParticleGrid.xEnd = dv;
                nv++;
                continue;
            }

            if (nv == 10) {
                if (sscanf(row, "%lf", &dv) < 1) {
                    fprintf(stderr, "Error reading ParticleGrid.Ys from string\n");
                    EXIT(-1);
                }

                ParticleGrid.yBegin = dv;
                nv++;
                continue;
            }

            if (nv == 11) {
                if (sscanf(row, "%lf", &dv) < 1) {
                    fprintf(stderr, "Error reading ParticleGrid.Ye from string\n");
                    EXIT(-1);
                }

                ParticleGrid.yEnd = dv;
                break;
            }
        }

        // Read MaxIters.
        MaxIters = 0;

        while (1) {
            if (readRow(row, 80, f) < 1) {
                fprintf(stderr, "Error reading MaxIters from input file\n");
                EXIT(-1);
            }

            if (row[0] == '#' || rowLength(row) < 1) {
                continue;
            }

            if (sscanf(row, "%d", &MaxIters) < 1) {
                fprintf(stderr, "Error reading MaxIters from string\n");
                EXIT(-1);
            }

            printf("MaxIters = %d\n", MaxIters);
            break;
        }

        // Read MaxSteps.
        MaxSteps = 0;

        while (1) {
            if (readRow(row, 80, f) < 1) {
                fprintf(stderr, "Error reading MaxSteps from input file\n");
                EXIT(-1);
            }

            if (row[0] == '#' || rowLength(row) < 1) {
                continue;
            }

            if (sscanf(row, "%d", &MaxSteps) < 1) {
                fprintf(stderr, "Error reading MaxSteps from string\n");
                EXIT(-1);
            }

            printf("MaxSteps = %d\n", MaxSteps);
            break;
        }

        // Read the time step.
        timeStep = 0;

        while (1) {
            if (readRow(row, 80, f) < 1) {
                fprintf(stderr, "Error reading the time step from input file\n");
                EXIT(-1);
            }

            if (row[0] == '#' || rowLength(row) < 1) {
                continue;
            }

            if (sscanf(row, "%lf", &timeStep) < 1) {
                fprintf(stderr, "Error reading the time step from string\n");
                EXIT(-1);
            }

            printf("TimeBit = %lf\n", timeStep);
            break;
        }

        fclose(f);
    }

    // Broadcast the configuration.
    MPI_Bcast(&MaxIters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MaxSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&timeStep, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&GenFieldGrid.width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&GenFieldGrid.height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&GenFieldGrid.xBegin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&GenFieldGrid.xEnd, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&GenFieldGrid.yBegin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&GenFieldGrid.yEnd, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&ParticleGrid.width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ParticleGrid.height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ParticleGrid.xBegin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ParticleGrid.xEnd, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ParticleGrid.yBegin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ParticleGrid.yEnd, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Grid allocations.
    GenFieldGrid.values = (int *) malloc(GenFieldGrid.width * GenFieldGrid.height * sizeof(int));

    if (!GenFieldGrid.values) {
        fprintf(stderr, "Process %d: Error allocating GenFieldGrid.Values\n", procId);
        EXIT(-1);
    }

    ParticleGrid.values = (int *) malloc(ParticleGrid.width * ParticleGrid.height * sizeof(int));

    if (!ParticleGrid.values) {
        fprintf(stderr, "Process %d: Error allocating ParticleGrid.Values\n", procId);
        EXIT(-1);
    }

    if (procId == 0) {
        fprintf(stdout, "GenFieldGrid ");
        print_i2dGrid(GenFieldGrid);

        fprintf(stdout, "ParticleGrid ");
        print_i2dGrid(ParticleGrid);
    }
}

void GeneratingField(struct i2dGrid *grid, int maxIterations) {
    if (procId == 0) {
        fprintf(stdout, "Computing generating field ...\n");
    }

    int width = grid->width;
    int height = grid->height;
    double Sr = grid->xEnd - grid->xBegin;
    double Si = grid->yEnd - grid->yBegin;
    double Ir = grid->xBegin;
    double Ii = grid->yBegin;

    double Xinc = Sr / (double) width;
    double Yinc = Si / (double) height;

    // Determine the range of rows handled by each process.
    int rowsPerProcess = height / nProcs;
    int processRowCount = 0;
    int *procElementsCounts = (int *) malloc(nProcs * sizeof(int));
    int *dataDisplacements = (int *) malloc(nProcs * sizeof(int));

    for (int i = 0; i < nProcs; i++) {
        int startRow = i * rowsPerProcess;
        int endRow = startRow + rowsPerProcess + (i == nProcs - 1 ? height % nProcs : 0);
        int numOfProcRows = endRow - startRow;

        if (i == procId) {
            processRowCount = numOfProcRows;
        }

        procElementsCounts[i] = numOfProcRows * width;

        if (i == 0) {
            dataDisplacements[i] = 0;
        } else {
            dataDisplacements[i] = dataDisplacements[i - 1] + procElementsCounts[i - 1];
        }
    }

    // Perform computations for the assigned rows.
    int beginRow = rowsPerProcess * procId;
    int endRow = rowsPerProcess * procId + processRowCount;

    for (int iy = beginRow; iy < endRow; iy++) {
        #pragma omp parallel for default(none) shared(iy, width, Xinc, Yinc, Ir, Ii, maxIterations, grid)
        for (int ix = 0; ix < width; ix++) {
            double ca = Xinc * ix + Ir;
            double cb = Yinc * iy + Ii;
            double rad = sqrt(ca * ca * ((double) 1.0 + (cb / ca) * (cb / ca)));

            double za, zb;
            double zaNext = 0, zbNext = 0;

            int iz;

            for (iz = 1; iz <= maxIterations; iz++) {
                if (rad > 2) {
                    break;
                }

                za = zaNext;
                zb = zbNext;
                zaNext = ca + (za - zb) * (za + zb);
                zbNext = 2.0 * (za * zb + cb / 2.0);
                rad = sqrt(zaNext * zaNext * ((double) 1.0 + (zbNext / zaNext) * (zbNext / zaNext)));
            }

            if (iz >= maxIterations) {
                iz = 0;
            }

            grid->values[index2D(ix, iy, width)] = iz;
        }
    }

    MPI_Allgatherv(grid->values + dataDisplacements[procId], procElementsCounts[procId], MPI_INT,
                   grid->values, procElementsCounts, dataDisplacements, MPI_INT,
                   MPI_COMM_WORLD);

    free(procElementsCounts);
    free(dataDisplacements);
}

void ParticleGeneration(struct i2dGrid grid, struct i2dGrid pgrid, struct population *population) {
    // A system of particles is generated according to the value distribution of grid.Values.
    int vmax = MaxIntVal(grid.width * grid.height, grid.values);
    int vmin = MinIntVal(grid.width * grid.height, grid.values);

    // Determine the range of rows handled by each process.
    int rowsPerProcess = grid.height / nProcs;
    int beginRow = procId * rowsPerProcess;
    int endRow = beginRow + rowsPerProcess + (procId == nProcs - 1 ? grid.height % nProcs : 0);

    // Count number of particles to be generated.
    vmin = (double) (1 * vmax + 29 * vmin) / 30.0;
    int np = 0;

    for (int iy = beginRow; iy < endRow; iy++) {
        #pragma omp parallel for default(none) shared(iy, grid, vmin, vmax) reduction(+:np)
        for (int ix = 0; ix < grid.width; ix++) {
            int v = grid.values[index2D(ix, iy, grid.width)];

            if (v >= vmin && v <= vmax) {
                np++;
            }
        }
    }

    // Collect the number of particles identified by each process.
    int *particlesPerProcess = (int *) malloc(nProcs * sizeof(int));
    int *particlesDisplacements = (int *) malloc(nProcs * sizeof(int));

    MPI_Allgather(&np, 1, MPI_INT, particlesPerProcess, 1, MPI_INT, MPI_COMM_WORLD);
    population->amount = 0;

    for (int i = 0; i < nProcs; i++) {
        particlesDisplacements[i] = population->amount;
        population->amount += particlesPerProcess[i];
    }

    // Allocate memory space for particles.
    population->weight = (double *) malloc(population->amount * sizeof(double));
    population->x = (double *) malloc(population->amount * sizeof(double));
    population->y = (double *) malloc(population->amount * sizeof(double));
    population->vx = (double *) malloc(population->amount * sizeof(double));
    population->vy = (double *) malloc(population->amount * sizeof(double));

    // Initialize the particles.
    int initialized = 0;

    for (int iy = beginRow; iy < endRow; iy++) {
        #pragma omp parallel for default(none) shared(iy, grid, pgrid, population, vmin, vmax, initialized, particlesPerProcess, particlesDisplacements, procId)
        for (int ix = 0; ix < grid.width; ix++) {
            int v = grid.values[index2D(ix, iy, grid.width)];

            if (v >= vmin && v <= vmax) {
                int finished;
                int relativePos = 0;

                #pragma omp critical
                {
                    finished = initialized >= particlesPerProcess[procId];

                    if (!finished) {
                        relativePos = initialized;
                        initialized++;
                    }
                }

                if (!finished) {
                    double p;
                    int absolutePos = particlesDisplacements[procId] + relativePos;
                    population->weight[absolutePos] = v * 10.0;

                    p = (pgrid.xEnd - pgrid.xBegin) * ix / (grid.width * 2.0);
                    population->x[absolutePos] = pgrid.xBegin + ((pgrid.xEnd - pgrid.xBegin) / 4.0) + p;

                    p = (pgrid.yEnd - pgrid.yBegin) * iy / (grid.height * 2.0);
                    population->y[absolutePos] = pgrid.yBegin + ((pgrid.yEnd - pgrid.yBegin) / 4.0) + p;

                    // At start the particles are still.
                    population->vx[absolutePos] = 0.0;
                    population->vy[absolutePos] = 0.0;
                }
            }
        }
    }

    // Gather data from processes.
    gatherParticles(population, particlesPerProcess, particlesDisplacements);

    if (procId == 0) {
        print_Population(*population);
    }

    free(particlesPerProcess);
    free(particlesDisplacements);
}

void SystemEvolution(struct i2dGrid *pgrid, struct population *population, int numSteps) {
    // Determine the number of particles handled by the process.
    int *particlesPerProcess = (int *) malloc(nProcs * sizeof(int));
    int *particlesDisplacements = (int *) malloc(nProcs * sizeof(int));

    for (int i = 0; i < nProcs; i++) {
        int processParticles = population->amount / nProcs;
        int beginParticle = i * processParticles;
        int endParticle = beginParticle + processParticles + (i == nProcs - 1 ? population->amount % nProcs : 0);
        particlesPerProcess[i] = endParticle - beginParticle;
        particlesDisplacements[i] = beginParticle;
    }

    // Temporary array of forces.
    double *forces = (double *) malloc(2 * particlesPerProcess[procId] * sizeof(double));

    if (!forces) {
        fprintf(stderr, "Process %d: Error mem alloc of forces!\n", procId);
        EXIT(-1);
    }

    int beginParticle = particlesDisplacements[procId];
    int endParticle = particlesDisplacements[procId] + particlesPerProcess[procId];

    // Compute forces acting on each particle step by step.
    for (int step = 0; step < numSteps; step++) {
        if (procId == 0) {
            fprintf(stdout, "Step %d of %d\n", step, numSteps);

            ParticleScreen(pgrid, *population, step);

            // DumpPopulation call frequency may be changed.
            if (step % 4 == 0 || step == numSteps - 1) {
                DumpPopulation(*population, step);
            }

            ParticleStats(*population, step);
        }

        // Set forces to zero.
        memset(forces, 0, 2 * particlesPerProcess[procId] * sizeof(double));

        // Compute the forces applied to the particles.
        #pragma omp parallel for default(none) shared(beginParticle, endParticle, population, forces)
        for (int i = beginParticle; i < endParticle; i++) {
            struct particle p1, p2;
            configParticle(&p1, population->weight[i], population->x[i], population->y[i], population->vx[i], population->vy[i]);

            for (int j = 0; j < population->amount; j++) {
                if (j != i) {
                    configParticle(&p2, population->weight[j], population->x[j], population->y[j], population->vx[j], population->vy[j]);
                    double f[2];
                    computeForces(f, p1, p2);

                    forces[index2D(0, i - beginParticle, 2)] += f[0];
                    forces[index2D(1, i - beginParticle, 2)] += f[1];
                }
            }
        }

        // Apply the forces to the particles.
        applyForces(population, beginParticle, endParticle, forces);

        // Gather data from processes.
        gatherParticles(population, particlesPerProcess, particlesDisplacements);
    }

    free(particlesPerProcess);
    free(particlesDisplacements);
    free(forces);
}

void ParticleScreen(struct i2dGrid *particleGrid, struct population population, int step) {
    int static vmin, vmax;

    int Xdots = particleGrid->width;
    int Ydots = particleGrid->height;

    #pragma omp parallel for collapse(2) default(none) shared(particleGrid, Xdots, Ydots)
    for (int ix = 0; ix < Xdots; ix++) {
        for (int iy = 0; iy < Ydots; iy++) {
            particleGrid->values[index2D(ix, iy, Xdots)] = 0;
        }
    }

    double rmin = MinDoubleVal(population.amount, population.weight);
    double rmax = MaxDoubleVal(population.amount, population.weight);
    double wint = rmax - rmin;
    double Dx = particleGrid->xEnd - particleGrid->xBegin;
    double Dy = particleGrid->yEnd - particleGrid->yBegin;

    #pragma omp parallel for default(none) shared(particleGrid, population, step, Xdots, Ydots, rmin, rmax, wint, Dx, Dy)
    for (int n = 0; n < population.amount; n++) {
        // Keep a tiny border free anyway.
        int ix = Xdots * population.x[n] / Dx;

        if (ix >= Xdots - 1 || ix <= 0) {
            continue;
        }

        int iy = Ydots * population.y[n] / Dy;

        if (iy >= Ydots - 1 || iy <= 0) {
            continue;
        }

        double wv = population.weight[n] - rmin;
        int wp = 10.0 * wv / wint;

        particleGrid->values[index2D(ix, iy, Xdots)] = wp;
        particleGrid->values[index2D(ix - 1, iy, Xdots)] = wp;
        particleGrid->values[index2D(ix + 1, iy, Xdots)] = wp;
        particleGrid->values[index2D(ix, iy - 1, Xdots)] = wp;
        particleGrid->values[index2D(ix, iy + 1, Xdots)] = wp;
    }

    char name[40];
    sprintf(name, "stage%3.3d\0", step);

    if (step <= 0) {
        vmin = vmax = 0;
    }

    if (procId == 0) {
        IntVal2ppm(particleGrid->width, particleGrid->height, particleGrid->values, &vmin, &vmax, name);
    }
}

int MinIntVal(int s, int *a) {
    int v = a[0];

    #pragma omp parallel for default(none) shared(s, a) reduction(min:v)
    for (int i = 0; i < s; i++) {
        if (a[i] < v) {
            v = a[i];
        }
    }

    return v;
}

int MaxIntVal(int s, int *a) {
    int v = a[0];

    #pragma omp parallel for default(none) shared(s, a) reduction(max:v)
    for (int i = 0; i < s; i++) {
        if (a[i] > v) {
            v = a[i];
        }
    }

    return v;
}

double MinDoubleVal(int s, double *a) {
    double v = a[0];

    #pragma omp parallel for default(none) shared(s, a) reduction(min:v)
    for (int i = 0; i < s; i++) {
        if (a[i] < v) {
            v = a[i];
        }
    }

    return v;
}

double MaxDoubleVal(int s, double *a) {
    double v = a[0];

    #pragma omp parallel for default(none) shared(s, a) reduction(max:v)
    for (int i = 0; i < s; i++) {
        if (a[i] > v) {
            v = a[i];
        }
    }

    return v;
}

int rowLength(char *row) {
    int lungh;
    char c;

    lungh = strlen(row);

    while (lungh > 0) {
        lungh--;
        c = *(row + lungh);
        if (c == '\0') continue;
        if (c == '\40') continue; // Space
        if (c == '\b') continue;
        if (c == '\f') continue;
        if (c == '\r') continue;
        if (c == '\v') continue;
        if (c == '\n') continue;
        if (c == '\t') continue;
        return lungh + 1;
    }

    return 0;
}

int readRow(char *rg, int nc, FILE *daleg) {
    if (!fgets(rg, nc, daleg)) {
        return -1;
    }

    int lrg = rowLength(rg);

    if (lrg < nc) {
        rg[lrg] = '\0';
        lrg++;
    }

    return lrg;
}

void print_i2dGrid(struct i2dGrid g) {
    printf("i2dGrid: EX, EY = %d, %d\n", g.width, g.height);
    printf("         Xs, Xe = %lf, %lf; Ys, Ye = %lf, %lf\n", g.xBegin, g.xEnd, g.yBegin, g.yEnd);
}

void print_particle(struct particle p) {
    printf("particle: weight=%lf, x,y=(%lf,%lf), vx,vy=(%lf,%lf), fx,fy=(%lf,%lf)\n",
           p.weight, p.x, p.y, p.vx, p.vy, p.fx, p.fy);
}

void print_Population(struct population p) {
    printf("Population: np = %d\n", p.amount);
}

void DumpPopulation(struct population p, int t) {
    char fname[80];
    FILE *dump;

    sprintf(fname, "Population%4.4d.dmp", t);
    dump = fopen(fname, "w");

    if (!dump) {
        fprintf(stderr, "Error write open file %s\n", fname);
        EXIT(-1);
    }

    fwrite(&p.amount, sizeof(int), 1, dump);
    fwrite(p.weight, sizeof(double), p.amount, dump);
    fwrite(p.x, sizeof(double), p.amount, dump);
    fwrite(p.y, sizeof(double), p.amount, dump);
    fclose(dump);
}

void ParticleStats(struct population p, int t) {
    FILE *stats;
    double w, xg, yg, wmin, wmax;

    if (t <= 0) {
        stats = fopen("Population.sta", "w");
    } else {
        stats = fopen("Population.sta", "a");
    }

    if (!stats) {
        fprintf(stderr, "Error append/open file Population.sta\n");
        EXIT(-1);
    }

    // Initialize reduction variables.
    w = xg = yg = 0.0;
    wmin = wmax = p.weight[0];

    // Compute statistics.
    #pragma omp parallel for default(none) shared(p) reduction(min:wmin) reduction(max:wmax) reduction(+:w, xg, yg)
    for (int i = 0; i < p.amount; i++) {
        if (wmin > p.weight[i]) {
            wmin = p.weight[i];
        }

        if (wmax < p.weight[i]) {
            wmax = p.weight[i];
        }

        w += p.weight[i];
        xg += p.weight[i] * p.x[i];
        yg += p.weight[i] * p.y[i];
    }

    xg = xg / w;
    yg = yg / w;

    fprintf(stats, "At iteration %d particles: %d; wmin, wmax = %lf, %lf;\n", t, p.amount, wmin, wmax);
    fprintf(stats, "   total weight = %lf; CM = (%10.4lf,%10.4lf)\n", w, xg, yg);
    fclose(stats);
}

void IntVal2ppm(int width, int height, int *idata, int *vmin, int *vmax, char *name) {
    // RGB Color Map.
    int cm[3][256];

    FILE *ouni, *ColMap;
    int vp;
    int rmin, rmax, value;
    char fname[80];

    #ifdef IMAGEMAGICK
    char jname[80], command[80];
    #endif

    unsigned char *row = (unsigned char *) malloc(width * sizeof(unsigned char) * 3);

    if (!row) {
        fprintf(stderr, "Errore allocazione row[%d][3]\n", width);
        EXIT(-1);
    }

    // Define color map: 256 colors
    ColMap = fopen("ColorMap.txt", "r");

    if (!ColMap) {
        fprintf(stderr, "Error read opening file ColorMap.txt\n");
        EXIT(-1);
    }

    for (int i = 0; i < 256; i++) {
        if (fscanf(ColMap, " %3d %3d %3d",
                   &cm[0][i], &cm[1][i], &cm[2][i]) < 3) {
            fprintf(stderr, "Error reading colour map at line %d: r, g, b =", (i + 1));
            fprintf(stderr, " %3.3d %3.3d %3.3d\n", cm[0][i], cm[1][i], cm[2][i]);
            EXIT(-1);
        }
    }

    // Write in PPM format.
    strcpy(fname, name);
    strcat(fname, ".ppm\0");
    ouni = fopen(fname, "wb");

    if (!ouni) {
        fprintf(stderr, "!!!! Error write access to file %s\n", fname);
    }

    // Magic code.
    fprintf(ouni, "P6\n");

    // Dimensions.
    fprintf(ouni, "%d %d\n", width, height);

    // Maximum value.
    fprintf(ouni, "255\n");

    // Values from 0 to 255.
    rmin = MinIntVal(width * height, idata);
    rmax = MaxIntVal(width * height, idata);

    if ((*vmin == *vmax) && (*vmin == 0)) {
        *vmin = rmin;
        *vmax = rmax;
    } else {
        rmin = *vmin;
        rmax = *vmax;
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            value = idata[i * width + j];

            if (value < rmin) {
                value = rmin;
            }

            if (value > rmax) {
                value = rmax;
            }

            vp = (int) ((double) (value - rmin) * (double) 255.0 / (double) (rmax - rmin));
            row[j * 3] = (unsigned char) cm[0][vp];
            row[j * 3 + 1] = (unsigned char) cm[1][vp];
            row[j * 3 + 2] = (unsigned char) cm[2][vp];
        }

        fwrite(row, (width * sizeof(unsigned char) * 3), 1, ouni);
    }

    fclose(ouni);

    #ifdef IMAGEMAGICK
    strcpy(jname, name);
    strcat(jname, ".jpg\0");
    sprintf(command, "convert %s %s\0", fname, jname);
    system(command);
    #endif

    free(row);
}

int main(int argc, char *argv[]) {
    // Initialize the MPI environment.
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procId);

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <configuration_file>\n", argv[0]);
        EXIT(-1);
    }

    // Run the main program.
    struct timespec t0;
    time_t t0s;

    if (procId == 0) {
        time(&t0s);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        fprintf(stdout, "Starting at: %s", asctime(localtime(&t0s)));
    }

    readConfiguration(argv[1]);

    // Initialize the generating field.
    if (procId == 0) {
        printf("GeneratingField...\n");
    }

    GeneratingField(&GenFieldGrid, MaxIters);

    // Create the particles.
    if (procId == 0) {
        printf("ParticleGeneration...\n");
    }

    ParticleGeneration(GenFieldGrid, ParticleGrid, &particles);

    if (procId == 0) {
        // Compute evolution of the particle population
        printf("SystemEvolution...\n");
    }

    SystemEvolution(&ParticleGrid, &particles, MaxSteps);

    if (procId == 0) {
        struct timespec t1;
        time_t t1s;

        clock_gettime(CLOCK_MONOTONIC, &t1);
        time(&t1s);

        fprintf(stdout, "Ending   at: %s", asctime(localtime(&t1s)));
        fprintf(stdout, "Computations ended in %lf seconds\n", (double) (t1.tv_nsec - t0.tv_nsec) / 1000000000.0 + (double) (t1.tv_sec - t0.tv_sec));

        fprintf(stdout, "End of program!\n");
    }

    free(GenFieldGrid.values);
    free(ParticleGrid.values);

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}
