#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <Eigen/Dense>
#include "testfun.hpp"

struct Result{
  Eigen::VectorXd minimiser ;
  double minimum {NAN} ;
  int iterations {0};
  double grad_norm {NAN};
  double rel_sol_change {NAN};
  // int f_evaluations {0};
  friend std::ostream& operator<<(std::ostream& os, const Result& res);
};

struct OptimiserSettings{
  int max_iter {10000};
  double rel_sol_change_tol {1e-6};
  double grad_norm_tol {1e-6};
  double gam {0.001};
  bool save {false};
  friend std::ostream& operator<<(std::ostream& os, const OptimiserSettings& settings);
};

std::ostream& operator<<(std::ostream& os, const OptimiserSettings& settings) {
  return os << "\nMax Iterations: " << settings.max_iter << std::endl
            << "Relative Solution Change Tolerance: " << settings.rel_sol_change_tol << std::endl
            << "Norm of Gradient Tolerance: " << settings.grad_norm_tol << std::endl
            << "Step size: " << settings.gam << std::endl
            << "Save data? " << (settings.save ? "Yes" : "No");
}

std::ostream& operator<<(std::ostream& os, const Result& res) {
  return os << "\nMinimiser is: \n" << res.minimiser << std::endl
            << "Minimum is: " << res.minimum << std::endl
            << "Iterations: " << res.iterations << std::endl
            << "Change in gradient at final step: " << res.grad_norm << std::endl
            << "Relative change in minimiser at final step: " << res.rel_sol_change << std::endl;
            // << "Required " << res.f_evaluations << " evaulations of the objective";
}

class BaseOptimiser{
public:
  BaseOptimiser(ObjectiveFunction& f,const Eigen::VectorXd& int_val, const OptimiserSettings settings){};
  Eigen::VectorXd step = int_val;
  const int dim = int_val.size();
  Eigen::VectorXd df(dim);

  int iter = 0;
  Result minimise(ObjectiveFunction& f, const Eigen::VectorXd& int_val, const OptimiserSettings settings){
    Result res;
    res.rel_sol_change = 2*settings.rel_sol_change_tol;
    res.grad_norm = 2*settings.grad_norm_tol;
    df = f.gradient(step);
    // Save data
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%d_%m_%Y_%H%M%S");
    auto date = oss.str();
    if(settings.save){std::cout << "trajectory"+date+".csv" << std::endl;}
    std::ofstream trajectory("trajectory" + date + ".csv");


    while (res.iterations < settings.max_iter &&
           res.rel_sol_change > settings.rel_sol_change_tol &&
           res.grad_norm > settings.grad_norm_tol){
      step = update(f, step, settings, res);
      if(settings.save){trajectory << step.transpose() << std::endl;}
    }
    res.minimiser = step;
    res.minimum = f.evaluate(step);
    trajectory.close();
    return res;
  }
private:
    // Needs to be declared here, but will be overloaded in any subclass
    virtual Eigen::VectorXd update(ObjectiveFunction& f, const Eigen::VectorXd& step,
       const OptimiserSettings& settings, Result& res) = 0;
};

class GradientDescent: public BaseOptimiser{
public:
  Eigen::VectorXd update(ObjectiveFunction& f, const Eigen::VectorXd& step,
    const OptimiserSettings& settings, Result& res)
  {
    Eigen::VectorXd next_step = step;
    Eigen::VectorXd df = f.gradient(step);
    next_step = step - settings.gam*df;
    // The relative change in solution x_i and x_{i-1}
    res.rel_sol_change = abs(f.evaluate(next_step) - f.evaluate(step)/f.evaluate(step));
    // L2 norm of change in gradient from solution x_{i-1} to x_i
    res.grad_norm =  df.norm();
    res.iterations++;
    // std::cout << res.iterations;
    return next_step;
  }
};

// class BFGS: public BaseOptimiser{
// public:
//   //BFGS Declarations
//   Eigen::VectorXd next_step = step;
//   Eigen::VectorXd next_df(dim), s(dim), q(dim), w(dim);
//   Eigen::MatrixXd D = Eigen::MatrixXd::Identity(dim, dim);
//   double st_q, qt_D_q;
//   Eigen::VectorXd p(dim);
//   const double c =1e-4, tau = 0.95;
//
//   Eigen::VectorXd update(ObjectiveFunction& f, const Eigen::VectorXd& step,
//     const OptimiserSettings& settings, Result& res)
//   {
//     // Update step
//     p= -D*df;
//     settings.gam = 2;
//     while(f.evaluate(step + gam*p) > (f.evaluate(step) + gam*c*p.transpose()*df)){
//       settings.gam *= tau;
//     }
//     next_step = step + settings.gam*p;
//     // Project iteration back into hypercube if needed
//     // for (int i=0; i<dim; ++i){
//     //   if (step(i) < -max_bound){step(i) = -max_bound;}
//     //   else if (step(i)>max_bound){step(i)=max_bound;}
//     // }
//
//     // Compute new gradient and relative changes
//     next_df = f.gradient(next_step);
//     res.rel_sol_change = abs((f.evaluate(step) - f.evaluate(next_step))/f.evaluate(step));
//     res.grad_norm = next_df.norm();
//
//     // Compute difference between points and gradients at k and k+1.
//     s = next_step - step;
//     q = next_df - df;
//
//     // Update the matrix d
//     st_q = s.transpose()*q;
//     qt_D_q = abs(q.transpose()*D*q); // Small values may cause erratic negative values
//     w = sqrt(qt_D_q)*(s/st_q - D*q/qt_D_q);
//     D += s*s.transpose()/st_q - D*q*q.transpose()*D/qt_D_q + w*w.transpose();
//     // Update iterate values
//     step = next_step;
//     df = next_df;
//     ++res.iterations;
//     return step;
//   }
// };

int main(){
  OptimiserSettings settings;
  Result res;

  const int d = {2};
  settings.gam = 0.001;
  settings.max_iter = 100000;
  settings.grad_norm_tol = pow(10,-6);
  settings.rel_sol_change_tol = pow(10,-6);
  settings.save = false;
  std::cout << settings << std::endl;
  // Initialise vectors to hold the minimum value to be found and the intial values (which need to be inputed)
  // Eigen::VectorXd min_val(d);
  Eigen::VectorXd int_vals(d);
  Quadratic f(d);
  // Set intial values. This could be anything.
  int_vals << 4,3;
  // Use the gradient descent algorithm to calculate the minimum.
  GradientDescent gd(f,int_vals, settings);
  res = gd.minimise(f, int_vals, settings);

  std::cout << res  << std::endl;
  return 0;
}
