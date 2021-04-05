/* TESTPARPSO.CPP
Example usage of a parallel particle swarm algorithm to explore the space, before using BFGS to
 accurately determine the optimum.
For implementation of PSO see PSO.hpp
See also:
  - BFGS.hpp for a implementation of BFGS.
*/
#include"HPC_Opt.hpp"

int main(){
  PSOSettings settings;
  settings.gamma = 1;
  settings.min_bound = -512;
  settings.max_bound = 512;
  settings.dim = 2;
  settings.exit_bound = 1000;
  const double N = settings.exit_bound;
  const int total_particles = 100;
  const int iter_skip = 100; // Integer params
  Eigen::VectorXd true_sol (settings.dim);
  true_sol(0) = 512; true_sol(1) = 404.2319;
  double f_opt = pow(10,16);
  Eggholder f(settings.dim);  // Objective function

  // Initialise MPI
  int rank, size;
  double buffer [settings.dim];  // Used to send/recieve data
  MPI_Init(NULL, NULL);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
  int root = 0;  // Designate root proces
  double mean_time = 0, bfgs_time = 0, av_comm_time = 0;
  double mean_error = 0;
  int seeds = 5;  // Number diferent seeds to average over


  for (int seed = 0; seed < seeds; ++seed){
  srand(rank+100*seed + 100*size);
  double t_init = MPI_Wtime();

  // Number of particles for each process - check whether the required number is divisible
  // by the number of workers. If not, designate leftover work to required no. processors
  int num_particles;
  num_particles = total_particles/size; // Divide by size and comment/uncomment the line below according to strong/weak scaling
  if (rank >= size - total_particles%(size)){num_particles++;}  // Designate leftover work (strong scaling)
  settings.particles = num_particles;
  ParticleSwarm pso;
  Result res = pso.minimise(f, settings);
  Eigen::VectorXd glob_best = res.minimiser;
  // Gather global bests at root
  if (rank == root){

    double comm_time = MPI_Wtime();
    MPI_Status status;
    Eigen::VectorXd temp(settings.dim);
    for (int i = 0; i < size - 1; ++i){
      MPI_Recv(buffer, settings.dim, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, &status);
      for (int j=0; j<settings.dim; ++j){temp(j) = buffer[j];}
      if (f.evaluate(temp) < f_opt){f_opt = f.evaluate(temp); glob_best = temp;}
    }
    comm_time = MPI_Wtime() - comm_time;
    av_comm_time += comm_time;
    // Final BFGS
    double bfgs_time_0 = MPI_Wtime();
    // Final BFGS params
    BFGSSettings bfgs_settings;
    bfgs_settings.max_iter = pow(10, 5);
    bfgs_settings.rel_sol_change_tol = pow(10, -7);
    bfgs_settings.grad_norm_tol = pow(10, -7);
    bfgs_settings.dim = settings.dim;
    bfgs_settings.min_bound = settings.min_bound;
    bfgs_settings.max_bound = settings.max_bound;
    const double gam_max = 2;
    BFGS bfgs;
    Result res = bfgs.minimise(f,  glob_best, bfgs_settings);
    glob_best = res.minimiser;
    f_opt = f.evaluate(glob_best);
    bfgs_time = MPI_Wtime() - bfgs_time_0;
    double t_end = MPI_Wtime();
    mean_time += t_end - t_init;
    std::cout << "Runtime: " << t_end - t_init<<"\n";
    std::cout << "\nThe minimum function value is f = " << std::setprecision(10)<< f_opt <<
      " at:\n" << res.minimum <<"\n";
    std::cout << res << std::endl;
    mean_error += (true_sol - glob_best).norm();
  }

  // Send global bests to root
  else {
    for (int i = 0; i < settings.dim; ++i){buffer[i] = glob_best(i);}
    MPI_Send(buffer, settings.dim, MPI_DOUBLE, 0, 0, comm);
  }
  MPI_Barrier(comm);
}
  if (rank == root){
    std::cout << "\nProcessors: " << size <<"\nCommunication Time: " << av_comm_time/seeds
      << "\nRuntime: " << mean_time/seeds <<"\n";
    std::ofstream file;
    file.open("data/runtimes-egg-small.csv", std::ios_base::app);
    file << size << " " << mean_time/seeds << "\n";
    file.close();
    file.open("data/errors-egg-small.csv", std::ios_base::app);
    file << size << " " << mean_error/seeds << "\n";
    file.close();
    file.open("data/bfgs_runtimes-small.csv", std::ios_base::app);
    file << size << " " << bfgs_time/seeds << "\n";
    file.close();
  }

  MPI_Finalize();
  return 0;
}
