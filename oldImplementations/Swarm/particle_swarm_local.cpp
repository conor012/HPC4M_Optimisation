#include <iostream>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>
#include <iomanip>
#include <stdlib.h>
#include <random>
#include <mpi.h>
#include <string>

#include "../testfun.hpp"

using namespace std;




int main(){

  const int dim = 2, exit_bound = 1000, total_particles = 100, iter_skip = 100; // Integer params
  const double gamma = 1, min_bound = -512, max_bound = 512; // Double params
  const double N = exit_bound;

  Eggholder f(dim);  // Objective function

  // Initialise MPI
  int rank, size;
  double buffer [dim];  // Used to send/recieve data
  MPI_Init(NULL, NULL);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
  int root = 0;  // Designate root proces
  double mean_time = 0;
  int seeds = 1;
  double w, c1, c2;
  for (int seed = 0; seed < seeds; ++seed){
  srand(rank+100*seed + 100*size);
  double t_init = MPI_Wtime();

  // Open file to store data
  ofstream fileOut;
  fileOut.open("points" + to_string(rank) + ".csv");


  // Number of particles for each process - check whether the required number is divisible
  // by the number of workers. If not, designate leftover work to required no. processors
  int num_particles;
  num_particles = total_particles/size; // Divide by size and comment/uncomment the line below according to strong/weak scaling
  if (rank >= size - total_particles%(size)){num_particles++;}  // Designate leftover work (strong scaling)


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
  for (int m=0; m<=exit_bound; ++m){
    w = 2*(0.4*(N-m)/pow(N, 1) + 0.0);
    c1 = 2*(-3*m/N + 3.5);
    c2 = 2*(3*m/N + 0.5);

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
        if (f_pbest[n] < f_opt){f_opt = f_pbest[n]; glob_best = position[n];}
      }

      // Update velocity
      velocity[n] = w*velocity[n] + c2*distribution(generator)*(glob_best - position[n])
        + c1*distribution(generator)*(p_best[n] - position[n]);
    }
    for (int i=0; i<dim; ++i){
      for (int j=0; j<num_particles; ++j){
        fileOut << position[j](i) << " ";
      }
      fileOut << "\n";
    }

  }

  // Gather global bests at root
  if (rank == root){
    MPI_Status status;
    Eigen::VectorXd temp(dim);
    for (int i = 0; i < size - 1; ++i){
      MPI_Recv(buffer, dim, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, &status);
      for (int j=0; j<dim; ++j){temp(j) = buffer[j];}
      if (f.evaluate(temp) < f_opt){f_opt = f.evaluate(temp); glob_best = temp;}
    }
    double t_end = MPI_Wtime();
    mean_time += t_end - t_init;
    cout << "Runtime: " << t_end - t_init<<"\n";
    cout << "\nThe minimum function value is f = " << setprecision(10)<< f_opt <<
      " at:\n" << glob_best <<"\n";
  }

  // Send global bests to root
  else {
    for (int i = 0; i < dim; ++i){buffer[i] = glob_best(i);}
    MPI_Send(buffer, dim, MPI_DOUBLE, 0, 0, comm);
  }
  fileOut.close();
  MPI_Barrier(comm);
}
  if (rank == root){
    mean_time /= seeds;
    ofstream file;
    file.open("runtimes-loc.csv", ios_base::app);
    file << size << " " << mean_time << "\n";
    file.close();
  }

  MPI_Finalize();
  return 0;
}
