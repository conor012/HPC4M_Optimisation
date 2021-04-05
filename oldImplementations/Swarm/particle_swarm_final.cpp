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


template<typename ObjFunc> // Use the parameter 'max_bound' to define edge of hypercube to optimse in
Eigen::VectorXd bfgs(ObjFunc f, const int max_iter, const Eigen::VectorXd& int_val,
  const double rel_sol_change_tol, const double grad_nrm_tol, const double gam_max,
   const double max_bound, const double min_bound, const int rank){
    // Initialise vectors to hold xk, xk+1 and print initial value.
    Eigen::VectorXd step_prev = int_val, step;

    // Initialise dimension and gradient vectors of xk, xk+1 respectively
    const int dim = int_val.size();
    Eigen::VectorXd df_prev(dim), df(dim), s(dim), q(dim), w(dim);

    // Useful constants for updates
    double st_q, qt_D_q;
    double gam;
    // Initialise relative changes to large number
    double rel_sol_change = 2*rel_sol_change_tol;
    double grad_nrm = 2*grad_nrm_tol;

    // Initialise matrix D as the identity matrix
    Eigen::MatrixXd D = Eigen::MatrixXd::Identity(dim, dim);

    int iter = 0;  // Initial no. iterations

    // Compute initial gradient
    df_prev = f.gradient(step_prev);
    grad_nrm = df_prev.norm();
    // While max iteraitons has not been reached and all change tolerances have
    // not been met, perform bfgs update
    Eigen::VectorXd p(dim);
    const double c =1e-4, tau = 0.95;
    while (iter < max_iter && rel_sol_change > rel_sol_change_tol && grad_nrm > grad_nrm_tol){
      if (rank == 0){cout << "\n iter: " << iter << "\n";}
      // Update step
      p= -D*df_prev;
      gam = gam_max;
      while(f.evaluate(step_prev + gam*p) > (f.evaluate(step_prev) + gam*c*p.transpose()*df_prev)){
        gam *= tau;
      }
      step = step_prev + gam*p;
      // Project iteration back into hypercube if needed
      for (int i=0; i<dim; ++i){
        if (step(i) < min_bound){step(i) = min_bound;}
        else if (step(i)>max_bound){step(i)=max_bound;}
      }

      // Compute new gradient and relative changes
      df = f.gradient(step);
      rel_sol_change = abs((f.evaluate(step_prev) - f.evaluate(step))/f.evaluate(step));
      grad_nrm = df.norm();

      // Compute difference between points and gradients at k and k+1.
      s = step - step_prev;
      q = df - df_prev;

      // Update the matrix d
      st_q = s.transpose()*q;
      qt_D_q = abs(q.transpose()*D*q); // Small values may cause erratic negative values
      w = sqrt(qt_D_q)*(s/st_q - D*q/qt_D_q);
      D += s*s.transpose()/st_q - D*q*q.transpose()*D/qt_D_q + w*w.transpose();
      // Update iterate values
      step_prev = step;
      df_prev = df;
      ++iter;
    }
    // cout << "gamma is" << gamma << endl;
    return step;
  }


int main(){

  const int dim = 2, exit_bound = 1000, total_particles = 100, iter_skip = 100; // Integer params
  const double gamma = 1, min_bound = -512, max_bound = 512; // Double params
  const double N = exit_bound;

  Eigen::VectorXd true_sol (dim);
  true_sol(0) = 512; true_sol(1) = 404.2319;

  // Final BFGS params
  const int max_iter = pow(10, 5);
  const double rel_sol_change_tol = pow(10, -7), gam_max = 2, grad_nrm_tol = pow(10, -7);


  Eggholder f(dim);  // Objective function

  // Initialise MPI
  int rank, size;
  double buffer [dim];  // Used to send/recieve data
  MPI_Init(NULL, NULL);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
  int root = 0;  // Designate root proces
  double mean_time = 0, bfgs_time = 0, av_comm_time = 0;
  double mean_error = 0;
  int seeds = 10;  // Number diferent seeds to average over
  double w, c1, c2;


  for (int seed = 0; seed < seeds; ++seed){
  srand(rank+100*seed + 100*size);
  double t_init = MPI_Wtime();


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


  }



  // Gather global bests at root
  if (rank == root){
    double comm_time = MPI_Wtime();
    MPI_Status status;
    Eigen::VectorXd temp(dim);
    for (int i = 0; i < size - 1; ++i){
      MPI_Recv(buffer, dim, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, &status);
      for (int j=0; j<dim; ++j){temp(j) = buffer[j];}
      if (f.evaluate(temp) < f_opt){f_opt = f.evaluate(temp); glob_best = temp;}
    }
    comm_time = MPI_Wtime() - comm_time;
    av_comm_time += comm_time;
    // Final BFGS
    double bfgs_time_0 = MPI_Wtime();
    glob_best = bfgs(f, max_iter, glob_best, rel_sol_change_tol, grad_nrm_tol,
      gam_max, max_bound, min_bound, rank);
    bfgs_time = MPI_Wtime() - bfgs_time_0;
    double t_end = MPI_Wtime();
    mean_time += t_end - t_init;
    cout << "Runtime: " << t_end - t_init<<"\n";
    cout << "\nThe minimum function value is f = " << setprecision(10)<< f_opt <<
      " at:\n" << glob_best <<"\n";
    mean_error += (true_sol - glob_best).norm();
  }

  // Send global bests to root
  else {
    for (int i = 0; i < dim; ++i){buffer[i] = glob_best(i);}
    MPI_Send(buffer, dim, MPI_DOUBLE, 0, 0, comm);
  }
  MPI_Barrier(comm);
}
  if (rank == root){
    cout << "\nProcessors: " << size <<"\nCommunication Time: " << av_comm_time/seeds
      << "\nRuntime: " << mean_time/seeds <<"\n";
    ofstream file;
    file.open("data/runtimes-egg-small.csv", ios_base::app);
    file << size << " " << mean_time/seeds << "\n";
    file.close();
    file.open("data/errors-egg-small.csv", ios_base::app);
    file << size << " " << mean_error/seeds << "\n";
    file.close();
    file.open("data/bfgs_runtimes-small.csv", ios_base::app);
    file << size << " " << bfgs_time/seeds << "\n";
    file.close();
  }

  MPI_Finalize();
  return 0;
}
