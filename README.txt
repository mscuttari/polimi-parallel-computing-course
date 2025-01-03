In the 07-Particles2D folder there is the sequential source code 'particles2D_2024.c' to be optimized 
and scripts for compiling and running on Leonardo (with few changes on any other Linux platform).
Tests have been carried on with GNU compilers, but other C and Fortran compilers should do as well.

The parallelisation benchmarks should be run with the original Particles.inp file, but during 
the optimization process you may lower the 'iterations for particle dynamics' and/or increase
the 'bit of time for particle dynamics'.

In order to check parallelization correctness the files 'Population0000.dmp' and 'Population0096.dmp'
(or the latest one) should match with the ones of the original code. You can compare the numbers by
generating plain text files with the utility program ReadPopulation.c.

The original program produces a lot of .ppm image files also, with which a movie can be visualized using
the utility program Movie.py. You may choose to avoid generating of these files by commenting out the call 
to the function 'ParticleScreen()' , but please report this decision when you show benchmark timings.

Would you please read the comments in the source code for further instructions.

Would you please let me know about code issues by writing to m.cremonesi@cineca.it 
