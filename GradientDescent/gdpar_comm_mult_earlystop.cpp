#include "../testfun.hpp"
#include <iostream>
#include <Eigen/Dense>
#include "mpi.h"
#include <cfloat>
#include <fstream>

// Use template to allow any type to be passed to gd
template<typename ObjFunc> // Paramters 'max_bound' and 'min_bound' to define edge of a hypercube - optional
Eigen::VectorXd gd(ObjFunc f, const int max_iter, int num_particles, int dim, double rel_sol_change_tol,
                double grad_tol, double gam, int tau, const Eigen::VectorXd& true_min, 
                const double min_bound = -DBL_MAX, const double max_bound = DBL_MAX){
    // Intialise the step, which will be x_i and step_prev which will be x_{i-1} at...
    //.. each step of the gd algorithm
    int rank; int finish = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Eigen::VectorXd step [num_particles], step_prev [num_particles], df[num_particles];

    // Randomise initial points
    for (int n = 0; n < num_particles; ++n){
        step_prev[n] = max_bound*Eigen::VectorXd::Random(dim);
        for (int i=0; i<dim; ++i){
            if (step_prev[n](i) < min_bound){ step_prev[n](i) = -step_prev[n](i);}
        }
    }

    Eigen::VectorXd x_centre(dim), step_lowest(dim);

    int iter = 0;

    // Whilst the max number of iterations has not been reached and no other process has found the global minimum
    while(iter < max_iter && finish == 0){
        if(iter % tau == 0)   // When tau divides iter we communicate with the root
        {
            // First check what the minimum is at this process
            step_lowest = step_prev[0];
            for(int n = 0; n < num_particles; ++n){
            if(f.evaluate(step_lowest) > f.evaluate(step_prev[n])){
            step_lowest = step_prev[n];           
            }}
            // Check if this process is within 0.0001 of the true global minimum...
            if((step_lowest - true_min).squaredNorm() < 0.0001){
                // .. If it is send message to root process to stop all other processes.
                finish = 1;
                MPI_Send(&finish, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }else{ // Otherwise if process is rank 1 send this to keep the loop going in root process
                if(rank == 1){
            MPI_Send(&finish, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }}

            // Root process broadcasts if the true minimum has been found or not
            // if it has each process breaks after this loop.
            MPI_Bcast(&finish, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }


        // Perform gd algorithm for each of the n particles
        for(int n = 0; n < num_particles; ++n){
        df[n] = f.gradient(step_prev[n]);
        step[n] = step_prev[n] - gam*df[n];

        // Project iteration back into hypercube if needed
        for (int i=0; i<dim; ++i){
            //if (step(i) < 0){step(i) = 0;}
            if (step[n](i) < min_bound){step[n](i) = min_bound;}
            else if (step[n](i)>max_bound){step[n](i) = max_bound;}
        }
        //Update the step for each particle
        step_prev[n] = step[n];
        }
        iter++;
    }

    // Select the minimum from all runs of gd on this node
    step_lowest = step[0];
    for(int n = 1; n < num_particles; ++n){
        if(f.evaluate(step_lowest) > f.evaluate(step_prev[n])){
            step_lowest = step_prev[n];
        }
    }
    return step_lowest; // When one processor has found the root we return the minimum
}

void centre_func(){
    int finish = 0;
    int buffer;
    while(finish == 0){
    MPI_Recv(&finish, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Bcast(&finish, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

int main()
{
    int rank, size;  
    const int root = 0;
    MPI_Status status;

    MPI_Group group_world;
    MPI_Group new_group;

    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm comm_no_root;

    MPI_Comm_size(comm, &size);
    if(size < 2){
        std::cout << "Error: Need at least 2 processes!" << std::endl;
        MPI_Finalize();
        return 0;
    }
    MPI_Comm_rank(comm, &rank);

    // Create communicator with all process except root
    int process_ranks[size];
    for(int i = 0; i < size-1; i++){
        process_ranks[i] = i+1;
    }

    MPI_Comm_group( comm, &group_world);
    MPI_Group_incl(group_world, size-1, process_ranks, &new_group);
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &comm_no_root);

    const int d = {4};
    double gam = 0.0001; // Set the learning rate
    int n_iter = pow(10,5); // Set the maximum number of iterations
    double grad_tol = pow(10,-10); // Set the tolerance for norm gradient
    double rel_sol_change_tol = pow(10,-10); // Set the tolerance for change of solution
    double max_bound = 10; // domain boundaries
    double min_bound = 0;
    int tau = 1000; // communication period
    int total_particles = 100;
    int num_particles = total_particles/size; // Divide by size and comment/uncomment the line below according to strong/weak scaling
    if (rank >= size - total_particles%(size)){num_particles++;}  // Designate leftover work (strong scaling)
    // Initialise vectors to hold the minimum value to be found and the intial values (which need to be inputed)
    Eigen::VectorXd min_val(d), true_min(d);

    // True minimum
    true_min.fill(4);
    //true_min(0)= 512; true_min(1) = 404.2319;
    Shekel f(d);

    srand(100 * rank * time(0));
    // Seed the random number generator.
    MPI_Barrier(comm);
    // Start timing the code
    double t_start = MPI_Wtime();

    // Use the gradient descent algorithm to calculate the minimum (comment max_bound if not needed).
    if(root != rank){
    min_val = gd(f,n_iter,  num_particles, d, grad_tol, rel_sol_change_tol, gam, tau, true_min, min_bound , max_bound);
    MPI_Barrier(comm_no_root);
    int finish = -1;
    MPI_Send(&finish, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    // The root process is the communicator process so does not run gd
    if(root == rank){
        centre_func();
    }
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
    // Finish timing the code
    double t_end = MPI_Wtime();
    double time = t_end - t_start;
    std::cout << "\nAfter " << n_iter <<
     " iterations and with a change in gradient tolerance of " <<
     grad_tol << "\nand a change in relative solution tolerance of " <<
     rel_sol_change_tol << "\nthe minimiser is:\n" << min_val;

    std::cout << "\nThe objective value at this point is "
                        << f.evaluate(min_val) << std::endl;

    std::cout << "Run time for number is "  << time <<  " seconds."<<std::endl;
    std::ofstream file;
    file.open("runtimes-loc.csv", std::ios_base::app);
    file << size << " " << time << "\n";
    file.close();
  }

    MPI_Finalize();

}
