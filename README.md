# HPC4M Global Minima of Benchmark Optimisation Problems

Aim: Implement algorithms to find the global minima of the Eggholder function and Shekel function exploiting parallel architectures in an orderly C++ way.

## Getting Started

These implementations use [Eigen 3.3.9](https://eigen.tuxfamily.org/index.php?title=Main_Page) for its linear algebra. To compile the file `gd.cpp` with Eigen using g++,

```
g++ -I ..\eigen-3.3.9 gd.cpp -o gd.exe
```



## Test Objective Functions

Included in `testfun.hpp`. Contains:

BaseObjective: Takes dimension d, implements gradient in 2d using finite difference. Two functions: `evaluate` which takes a vector reference and returns a double, and `gradient` which takes a vector reference and returns a vector.

Quadratic: (n-dimensional, x^Tx) overload gradient with exact (2x)







## C++ Software Structure 
- [ ]  Use of Eigen or Armadillo?
- [ ] Intel MKL for distributed linear algebra  

## Basic Algorithms
Algorithms:
- [ ] Gradient Descent
- [ ] Stochastic Gradient Descent
- [ ] BFGS 
- [ ] Nelder Mead

Opportunities to parallelise?
Crossover with sampling algorithms?

## Algorithms Designed for Parallel
- [ ] Particle Swarm -- better to hunt as a swarm or packs (i.e. local or global interaction)?
- [ ] Ant/Bee Colony
- [ ] Genetic/Evolutionary Algorithms
- [ ] Work of Lelievre? -- parallel replica algorithm (possibly for metastable particle systems rather than optimisation)

