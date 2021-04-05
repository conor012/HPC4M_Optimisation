#include "HPC_Opt.hpp"

int main(){
  OptimiserSettings settings;
  settings.max_iter = pow(10,6);
  settings.dim = 2;
  settings.min_bound = -512;
  settings.max_bound = 512;

  //Eigen::VectorXd int_vals(settings.dim);
  Eggholder f(settings.dim);
  GridSearch gs;
  Result res = gs.minimise(f, settings);
  std::cout << res << std::endl;
}
