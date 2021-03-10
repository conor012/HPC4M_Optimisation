// Test BFGS method on problems (serial)
#include <iostream>
#include <cmath>
#include <fstream>
#include "../test/testfun.hpp"
#include <Eigen/Dense>

using namespace std;

template<typename ObjFunc> // Use the parameter 'max_bound' to define edge of hypercube to optimse in
Eigen::VectorXd bfgs(ObjFunc f, const int max_iter, const Eigen::VectorXd& int_val,
  const double rel_sol_change_tol, const double grad_nrm_tol, const double gam,
   const double max_bound){
    // Initialise vectors to hold xk, xk+1 and print initial value.
    Eigen::VectorXd step_prev = int_val, step;

    // Initialise dimension and gradient vectors of xk, xk+1 respectively
    const int dim = int_val.size();
    Eigen::VectorXd df_prev(dim), df(dim), s(dim), q(dim), w(dim);

    // Useful constants for updates
    double st_q, qt_d_q;

    // Initialise relative changes to large number
    double rel_sol_change = 2*rel_sol_change_tol;
    double grad_nrm = 2*grad_nrm_tol;

    // Initialise matrix D as the identity matrix
    Eigen::MatrixXd d = Eigen::MatrixXd::Identity(dim, dim);

    int iter = 0;  // Initial no. iterations

    // Compute initial gradient
    df_prev = f.gradient(step_prev);

    // While max iteraitons has not been reached and all change tolerances have
    // not been met, perform bfgs update
    while (iter < max_iter && rel_sol_change > rel_sol_change_tol && grad_nrm > grad_nrm_tol){
      // Update step
      step = step_prev - gam*d*df_prev;

      // Project iteration back into hypercube if needed
      for (int i=0; i<dim; ++i){
        if (step(i) < -max_bound){step(i) = -max_bound;}
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
      qt_d_q = abs(q.transpose()*d*q); // Small values may cause erratic negative values
      w = sqrt(qt_d_q)*(s/st_q - d*q/qt_d_q);
      d += s*s.transpose()/st_q - d*q*q.transpose()*d/qt_d_q + w*w.transpose();
      // Update iterate values
      step_prev = step;
      df_prev = df;
      ++iter;
    }
    return step;
  }

int main (){
  const int d = 2, max_iter = pow(10, 1);  // Dimensionality and max iterations
  double gam = 0.01; // Learning rate
  Eggholder f(d); // Define problem function
  double max_bound = 512;  // domain boundaries
  // Set convergence tolerances
  double grad_nrm_tol = pow(10,-10), rel_sol_change_tol = pow(10,-10);

  double minimum_f = pow(10,10), f_temp;  // Store the min f over all runs (f_temp used as placeholder below)
  Eigen::VectorXd overall_min(d);  // Store the minimum point over all runs

  for (int n=0; n<100000; ++n){  // loop for many values of n
    // Create vectors to hold initial and final values
    Eigen::VectorXd int_val(d), min_val(d);
    int_val = max_bound*Eigen::MatrixXd::Random(d,1); // Set initial point


    min_val = bfgs(f, max_iter, int_val, rel_sol_change_tol, grad_nrm_tol, gam, max_bound);

    // If the function value here is lower than all previous attempts, store final point
    f_temp = f.evaluate(min_val);
    if (f_temp < minimum_f){
      minimum_f = f_temp;
      overall_min = min_val;
      cout << "\n---------------------------------------------------------\n" <<
      "new minimum point:\n " << min_val << "\n function value: " << f_temp <<"\n";
    }
  }
  Eigen::VectorXd min_val = bfgs(f, 1000*max_iter, overall_min, rel_sol_change_tol, grad_nrm_tol, gam, max_bound);
  cout << "\n----------------------------------\n";
  cout << "\nThe global minimiser found was :\n" << min_val <<"\n with a function value of: " << minimum_f << "\n";
  return 0;
}
