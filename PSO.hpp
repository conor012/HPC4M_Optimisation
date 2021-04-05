/* PSO.HPP
Implementation of a particle swarm algorithm.
"vanilla" method has fixed bias between local and global known optimum,
decaying uses varying stepsize that biases towards the global known optimimum towards
the end of the iterations.

Should only be included indirectly through HPC_Opt.hpp.

For example usage see test/testPSO.cpp
See also:
  - misc.hpp for implementation of base classes.
  - test/testParPSO.cpp for a parallel usage
*/
#ifndef PSO
#define PSO

std::map<std::string, int> pso_methods = {
  { "vanilla"  , 1 },
  { "decaying" , 2 },
};

struct PSOSettings: OptimiserSettings
{
  int method {pso_methods["vanilla"]};
  int particles {100};
  int exit_bound {10000};
};

std::ostream& operator<<(std::ostream& os, const PSOSettings& settings)
{
  return os << "\n Using " << settings.particles << " particles \n"
            << "Will terminate after " << settings.exit_bound
            << " steps without better minima being found\n"
            << "Step size: " << settings.gamma << "\n"
            << "Save data? " << (settings.save ? "Yes" : "No") <<  "\n"
            << "Restricted to the hypercube [" << settings.min_bound << ","
            << settings.max_bound << "]";
}

class ParticleSwarm: public BaseOptimiser{
public:
  // std::ofstream file, file2;
  Result minimise(ObjectiveFunction& f, const PSOSettings settings)
  {
    Result res;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
    std::ofstream glob_best_file;
    if(settings.save)
    {
      create_save_file();
      glob_best_file.open("global_best.csv");
    }

    const int N = settings.particles;
    const int dim = settings.dim;
    Eigen::VectorXd position[N], p_best[N], velocity[N], glob_best;
    double f_pbest[N], f_opt = INFINITY; // Store f val at personal and global best
    // Initialise random position and velocities
    for (int n=0; n<N; ++n)
    {
      position[n] = settings.min_bound*Eigen::MatrixXd::Random(dim, 1);
      velocity[n] = settings.gamma*Eigen::MatrixXd::Random(dim, 1);

      // Compute initial pbest and fbest
      p_best[n] = position[n];
      f_pbest[n] = f.evaluate(position[n]);
      if (f_pbest[n] < f_opt)
      {
        f_opt = f_pbest[n]; glob_best = position[n];
      }
    }
    // Loop over all iterations
    for (int m=0; m<settings.exit_bound; ++m)
    {
      // std::cout << m << "\n";
      for (int n=0; n<N; ++n)
      {
        position[n] += velocity[n]; // Update position of particle n
        position[n] = check_bounds(position[n], settings);

        // Compare to local best
        if (f.evaluate(position[n]) < f_pbest[n])
        {
          f_pbest[n] = f.evaluate(position[n]);
          p_best[n] = position[n];
          // Compare to global best - reset counter if optimal
          if (f_pbest[n] < f_opt){f_opt = f_pbest[n]; glob_best = position[n]; m = 0;}
        }
        // Update velocity
        switch(settings.method)
        {
          case 1: // vanilla
          {
            velocity[n] += settings.gamma*distribution(generator)*(glob_best - position[n])
             + settings.gamma*distribution(generator)*(p_best[n] - position[n]);
          }
          case 2: // decaying
          {
            double w = 2*(0.4*(N-m)/pow(N, 1) + 0.0);
            double c1 = 2*(-3*m/N + 3.5);
            double c2 = 2*(3*m/N + 0.5);

            velocity[n] = w*velocity[n] + c2*distribution(generator)*(glob_best - position[n])
             + c1*distribution(generator)*(p_best[n] - position[n]);
          }
        }
      }

      res.iterations++;
      if(settings.save)
      {
        for (int j = 0; j < N; ++j)
        {
          trajectory << position[j].transpose().format(CommaInitFmt) << "\n";
        }
        glob_best_file << glob_best.transpose().format(CommaInitFmt) << "\n";
      }
    }
    if(settings.save){trajectory.close(); glob_best_file.close();}
    res.minimiser = glob_best;
    res.minimum = f.evaluate(glob_best);
    return res;
  }
};

#endif
