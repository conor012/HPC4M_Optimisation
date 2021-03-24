#include "HPC_Opt.hpp"

int main(){
  BFGSSettings settings;
  settings.max_iter = 100000;
  settings.grad_norm_tol = pow(10,-7);
  settings.rel_sol_change_tol = pow(10,-10);
  settings.save = false;
  std::cout << settings << std::endl;
  // Initialise vectors to hold the intial values (which need to be inputted)
  const int d = {2};
  Eigen::VectorXd int_vals(d);
  Eggholder f(d);
  // Set intial values. This could be anything.
  int_vals << 512 ,56;
  // Use the gradient descent algorithm to calculate the minimum.
  BFGS bfgs;
  Result res = bfgs.minimise(f, int_vals, settings);

  std::cout << res  << std::endl;
  return 0;
}
