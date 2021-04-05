/* TESTGD.CPP
Example usage of a Gradient Descent algorithm to optimise a function.
For implementation of GD see GD.hpp
See also:
  - test/testParGD.cpp for a parallel usage
*/
#include "../HPC_Opt.hpp"

int main(){
  GDSettings settings;
  settings.gamma = 0.001;
  settings.max_iter = 100000;
  settings.grad_norm_tol = pow(10,-7);
  settings.rel_sol_change_tol = pow(10,-10);
  settings.save = false;
  settings.min_bound = 0;
  settings.max_bound = 10;
  settings.method = 3; // 1,2,3 or gd_methods["momentum"] etc.
  settings.par_momentum = 0.1;
  settings.par_ada_norm_term = 1;
  settings.dim = {4};
  std::cout << settings << std::endl;
  // Initialise vectors to hold the intial values (which need to be inputted)
  Eigen::VectorXd int_vals(settings.dim);
  Shekel f(settings.dim);
  // Set intial values. This could be anything.
  int_vals << 3,5,3,5;
  // Use the gradient descent algorithm to calculate the minimum.
  GradientDescent gd;
  Result res = gd.minimise(f, int_vals, settings);

  std::cout << res  << std::endl;

  return 0;
}
