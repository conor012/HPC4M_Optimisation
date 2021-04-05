/* TESTGRIDSEARCH.CPP
Example usage ofa naive grid search to find the minimum
For implementation of GridSearch see GridSearch.hpp

*/
#include "../HPC_Opt.hpp"

int main(){
  OptimiserSettings settings;
  settings.max_iter = 100;
  settings.dim = 2;

  Eigen::VectorXd int_vals(settings.dim);
  Quadratic f(settings.dim);
  GridSearch gs;
  Result res = gs.minimise(f, int_vals, settings);
  std::cout << res << std::endl;
}
