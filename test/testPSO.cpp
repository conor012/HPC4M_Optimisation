/* TESTPSO.CPP
Example usage of a particle swarm algorithm to minimise a function.
For implementation of PSO see PSO.hpp
See also:
  - misc.hpp for a implementation of base classes.
*/
#include "../HPC_Opt.hpp"

int main(){
  PSOSettings settings;
  settings.particles = 1000;
  settings.gamma = 2;
  settings.exit_bound = 1000;
  settings.min_bound = -512;
  settings.max_bound = 512;
  settings.save = true;

  settings.method = pso_methods["decaying"];

  std::cout << settings << std::endl;
  settings.dim = {2};
  Eggholder f(settings.dim);
  // Use the Particle Swarm algorithm to calculate the minimum.
  ParticleSwarm pso;
  Result res = pso.minimise(f, settings);

  std::cout << res  << std::endl;

  return 0;
}
