#include "HPC_Opt.hpp"

int main(){
  GDSettings settings;
  settings.gamma = 0.001;
  settings.max_iter = 100000;
  settings.grad_norm_tol = pow(10,-7);
  settings.rel_sol_change_tol = pow(10,-10);
  settings.save = true;
  settings.min_bound = {0};
  settings.max_bound = {10};
  std::cout << settings << std::endl;
  // Initialise vectors to hold the intial values (which need to be inputted)
  const int d = {4};
  Eigen::VectorXd int_vals(d);
  Shekel f(d);
  // Set intial values. This could be anything.
  int_vals <<  5,5,5,5;
  // Use the gradient descent algorithm to calculate the minimum.
  GradientDescent gd;
  Result res = gd.minimise(f, int_vals, settings);

  std::cout << res  << std::endl;

  return 0;
}
