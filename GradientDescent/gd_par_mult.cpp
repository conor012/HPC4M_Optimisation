#include "../testfun.hpp"
#include <iostream>
#include <Eigen/Dense>
# include "mpi.h"
# include <cfloat>

// Use template to allow any type to be passed to gd
template<typename ObjFunc> // Paramter 'max_bound' to define edge of a hypercube - optional
Eigen::VectorXd gd(ObjFunc f, const int max_iter, int num_particles, int dim, double rel_sol_change_tol,
                double grad_tol, double gam, const double max_bound = DBL_MAX){
    // Learning rate is fixed value at the moment. Could change this?
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialise an array of vectors for value at step x_i and x_{i-1}, df at each point and step_lowest to hold the minimum value out of all ..
    // .. n particles to be returned.
    Eigen::VectorXd step [num_particles], step_prev [num_particles], df[num_particles], step_lowest(dim);

    // Randomly set initial point for each particle
    for (int n = 0; n < num_particles; ++n){
    step_prev[n] = max_bound * Eigen::VectorXd::Random(dim);
    }


    // Loop over all n particles to find the local minimiser for each (hopefully one of these will be the global minimiser)
    for(int n = 0; n < num_particles; ++n){

    // These to be something high enough for while loop to run through first step
    double rel_sol_change = 100;
    double grad_change = 100;

    int iter = 0;

    // Whilst the max number of iterations has not been reached, the relative change in the solutions and ...
    // ... the change in the gradient are both less than the prescribed tol we look...
    // ... hopefully getting closer and closer to the true minimiser.
    while(iter < max_iter && rel_sol_change > rel_sol_change_tol && grad_change > grad_tol){
        df[n] = f.gradient(step_prev[n]);                                                     // Calculate \nabla F(x_{i-1})...
        step[n] = step_prev[n] - gam*df[n];

         // Project iteration back into hypercube if needed
        for (int i=0; i<dim; ++i){
            if (step[n](i) < -max_bound){step[n](i) = -max_bound;}
            else if (step[n](i)>max_bound){step[n](i) = max_bound;}
        }

        rel_sol_change = abs(f.evaluate(step_prev[n]) - f.evaluate(step[n]));    // The relative change in solution x_i and x_{i-1}
        // L2 norm of change in gradient at step x_i
        grad_change =  (df[n]).norm();
        // Then update the steps (x_i's) and gradients (df's) for the next iteration.
        step_prev[n] = step[n];
        iter++;
    }}

    // After all conditions are met for all initial point we return smallest minimiser
    step_lowest = step[0];
    for(int n = 1; n < num_particles; ++n){
        if(f.evaluate(step_lowest) > f.evaluate(step_prev[n])){
            step_lowest = step_prev[n];
        }
    }
    return step_lowest;
}

int main()
{
    int rank;  
    int size;
    const int root = 0;
    MPI_Status status;

    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Comm_size(comm, &size);
    if(size < 2){
        std::cout << "Error: Need at least 2 processes!" << std::endl;
        MPI_Finalize();
        return 0;
    }
    MPI_Comm_rank(comm, &rank);

    const int d = {2};
    double gam = 0.001; // Set the learning rate
    int n_iter = 100000; // Set the maximum number of iterations
    double grad_tol = pow(10,-10); // Set the tolerance for norm gradient
    double rel_sol_change_tol = pow(10,-10); // Set the tolerance for change of solution
    double max_bound = 512; // domain boundaries

    int  num_particles = 200; // number of particles in each process

    // Initialise vectors to hold the minimum value to be found and the intial values (which need to be inputed)
    Eigen::VectorXd min_val(d);
    Eggholder f(d);

    // Set intial values. At the moment this selects them randomly but that could be changed.
    // Seed the time differently for each process
    unsigned seed = time(0) + rank;
    // Seed the random number generator.
    srand(seed);

    // Use the gradient descent algorithm to calculate the minimum (comment max_bound if not needed).
    min_val = gd(f,n_iter, num_particles, d, grad_tol, rel_sol_change_tol, gam, max_bound);

    // All processors except the root send the min_val they have found to the root
    if(rank!=root){
        MPI_Send(&min_val[0], d, MPI_DOUBLE, 0, 1, comm);
    }
    if(rank == root)
    {
    Eigen::VectorXd buffer(d); // buffer to hold incoming min_vals
    // Root process receives min_vals in any order (this should make the code faster)...
    for(int i=1; i < size; i++){
    MPI_Recv(&buffer[0], d, MPI_DOUBLE, MPI_ANY_SOURCE, 1, comm, MPI_STATUS_IGNORE);
    // ... it then evalutes the min_val recieved from each other process...
    if(f.evaluate(buffer) < f.evaluate(min_val)){
        min_val = buffer;                         // .. and keeps it only if it evaluates to a lower value than the current min_val.
    }
    }
    std::cout << "\nAfter " << n_iter <<
     " iterations and with a change in gradient tolerance of " <<
     grad_tol << "\nand a change in relative solution tolerance of " <<
     rel_sol_change_tol << "\nthe minimiser is:\n" << min_val;

    std::cout << "\nThe objective value at this point is "
                        << f.evaluate(min_val) << std::endl;
    }

    MPI_Finalize();

    return 0;

}
