#include <iostream>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>
#include <iomanip>
#include <stdlib.h>
#include <random>

#include "../testfun.hpp"

using namespace std;


template<typename ObjFunc>
Eigen::VectorXd particle_swarm(ObjFunc f, const int num_particles,
  const double gamma, const int dim, const double min_bound, const double max_bound,
  const int exit_bound){// The algorithm terminates after exit_bound steps without finding new global min
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
    ofstream file, file2;
    file.open("data.txt");
    file2.open("global_best.txt");
    // Loop over all iterations
    for (int m=0; m<exit_bound; ++m){
      std::cout << m << "\n";
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

      // Save output
      for (int i = 0; i < dim; ++i){
        for (int j = 0; j < num_particles; ++j){
          file << position[j](i) << " ";
        }
        file <<"\n";
        file2 << glob_best(i) <<" ";
      }
      file2 << "\n";
    }
    file.close(); file2.close();
    return glob_best;
  }

int main(){
  const int dim = 2, exit_bound = 10000, num_particles = 100; // Integer params
  const double gamma = 2, min_bound = -512, max_bound = 512; // Double params
  Eggholder f(dim);  // Objective function

  Eigen::VectorXd glob_best;
  glob_best = particle_swarm(f, num_particles, gamma, dim, min_bound, max_bound, exit_bound);

  cout << "\nThe minimum function value is f = " << setprecision(10)<< f.evaluate(glob_best) <<
    " at:\n" << glob_best <<"\n";

  return 0;
}
