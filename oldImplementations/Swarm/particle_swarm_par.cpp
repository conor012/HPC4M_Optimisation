#include <iostream>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>
#include <iomanip>
#include <stdlib.h>
#include <random>
#include <mpi.h>

#include "../testfun.hpp"

using namespace std;

template<typename ObjFunc> // Use the parameter 'max_bound' to define edge of hypercube to optimse in
Eigen::VectorXd particle_swarm(ObjFunc f, const int total_particles, const int iter_skip,
  const double gamma, const int dim, const double min_bound, const double max_bound,
  const int exit_bound){
     // Initialise MPI
     int rank, size;
     double buffer [dim];  // Used to send/recieve data
     MPI_Init(NULL, NULL);
     MPI_Comm comm = MPI_COMM_WORLD;
     MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
     int root = 0;  // Designate root process

     if (rank == root){ // The root process performs job of updates and sharing memory
       Eigen::VectorXd global_best(dim), temp(dim); // Create vector to hold global best
       double f_opt = pow(10,16);
       int done = 0;
       MPI_Status status;
       double t_init = MPI_Wtime(); // Start time
       ofstream file;
       file.open("runtimes.csv");
       while (done < size -1){
         MPI_Recv(buffer, dim, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, &status); // Recieve from any source
         for (int i=0; i<dim; ++i){temp(i) = buffer[i];} // Check if optimal
         if (buffer[0] >= 1.5*max_bound){done += 1;continue;} // If exit status recieved, update no.completed processes
         else if (f.evaluate(temp)<f_opt){global_best = temp; f_opt = f.evaluate(temp);}
         for (int i=0; i<dim; ++i){buffer[i] = global_best(i);}
         MPI_Send(buffer, dim, MPI_DOUBLE, status.MPI_SOURCE, 1, comm); // And return to source
       }
       double t_end = MPI_Wtime() - t_init;
       file << size << " " << t_end<<"\n";
       cout << "Runtime: " << t_end <<"\n";
       cout << "\nThe minimum function value is f = " << setprecision(10)<< f.evaluate(global_best) <<
         " at:\n" << global_best <<"\n";
       file.close();
       return global_best;
     }

     else{
       // Number of particles for each process - check whether the required number is divisible
       // by the number of workers. If not, designate leftover work to final process
       int num_particles;
       if (rank < size - 1){num_particles = total_particles/(size - 1);}
       else {num_particles = total_particles/(size - 1) + total_particles%(size - 1);}


       // Initialise position, personal best and velocity vectors
       Eigen::VectorXd position [num_particles], p_best [num_particles], velocity [num_particles], glob_best;
       double f_pbest [num_particles], f_opt = pow(10, 16); // Store f val at personal and global best


       // Initialise random position and velocities
       for (int n=0; n<num_particles; ++n){
         position[n] = min_bound*Eigen::MatrixXd::Random(dim, 1);
         velocity[n] = gamma*Eigen::MatrixXd::Random(dim, 1);

         // Compute initial pbest and fbest
         p_best[n] = position [n];
         f_pbest[n] = f.evaluate(position[n]);
         if (f_pbest[n] < f_opt){f_opt = f_pbest[n]; glob_best = position[n];}
       }
       default_random_engine generator;
       uniform_real_distribution<double> distribution (0.0, 1.0);
       // Loop over all iterations
       for (int m=0; m<exit_bound; ++m){

         for (int n=0; n<num_particles; ++n){
           position[n] += velocity[n]; // Update position of particle n

           for (int i=0; i<dim; ++i){ // Project back into domain
             if (position[n](i) < min_bound){position[n](i) = min_bound;}
             else if (position[n](i) > max_bound){position[n](i) = max_bound;}
           }

           // Compare to local best
           if (f.evaluate(position[n]) < f_pbest[n]){
             f_pbest[n] = f.evaluate(position[n]);
             p_best[n] = position[n];
             // Compare to global best - reset counter if optimal
             if (f_pbest[n] < f_opt){f_opt = f_pbest[n]; glob_best = position[n]; m = 0;}
           }

           // Update velocity
           velocity[n] += gamma*distribution(generator)*(glob_best - position[n])
             + gamma*distribution(generator)*(p_best[n] - position[n]);
         }

         if (m%iter_skip == 0){  // Share data with root after iter_skip steps
           // Create array to store data
           for (int i=0; i<dim;++i){buffer[i] = glob_best(i);}
           MPI_Send(buffer, dim, MPI_DOUBLE, root, 0, comm); // Send optimal to root
           MPI_Recv(buffer, dim, MPI_DOUBLE, root, 1, comm, MPI_STATUS_IGNORE); // Recieve new optimal
           for (int i=0; i < dim; ++i){glob_best(i) = buffer[i];}
         }
       }
       // Send exit signal to root
       for (int i=0; i<dim;++i){buffer[i]=2*max_bound;}
       MPI_Send(buffer, dim, MPI_DOUBLE, root, 0, comm); // Send exit status to root
     }

     MPI_Finalize();
  }


int main(){
  const int dim = 2, exit_bound = 10000, total_particles = 1000, iter_skip = 100; // Integer params
  const double gamma = 2, min_bound = -512, max_bound = 512; // Double params

  Eggholder f(dim);  // Objective function

  Eigen::VectorXd glob_best;
  glob_best = particle_swarm(f, total_particles, iter_skip, gamma, dim, min_bound, max_bound, exit_bound);

  return 0;
}
