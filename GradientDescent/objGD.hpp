#ifndef GD
#define GD

std::map<std::string, int> gd_methods = {
  { "vanilla"  , 1 },
  { "momentum" , 2 }
};
struct GDSettings: OptimiserSettings
{
  const int method {gd_methods["vanilla"]};
};

// Redefine insertion operator to include method name
std::ostream& operator<<(std::ostream& os, const GDSettings& settings)
{
  std::map<int,std::string> reverse_gd_methods = {
    { 1, "vanilla" },
    { 2, "momentum" }
  };
  return os << "\nUsing method: " << reverse_gd_methods[settings.method] <<std::endl
            << "Max Iterations: " << settings.max_iter << std::endl
            << "Relative Solution Change Tolerance: " << settings.rel_sol_change_tol << std::endl
            << "Norm of Gradient Tolerance: " << settings.grad_norm_tol << std::endl
            << "Step size: " << settings.gamma << std::endl
            << "Save data? " << (settings.save ? "Yes" : "No") << std::endl
            << "Restricted to the hypercube [" << settings.min_bound << "," << settings.max_bound << "]";
}

class GradientDescent: public BaseOptimiser{
public:
  Result minimise(ObjectiveFunction& f, const Eigen::VectorXd& int_val, const GDSettings settings){
    Result res;
    res.rel_sol_change = 2*settings.rel_sol_change_tol;
    res.grad_norm = 2*settings.grad_norm_tol;
    // Configure output format for vectors
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
    if(settings.save){create_save_file();}

    Eigen::VectorXd step = int_val;

    while (res.iterations < settings.max_iter &&
           res.rel_sol_change > settings.rel_sol_change_tol &&
           res.grad_norm > settings.grad_norm_tol)
      {
        step = update(f, step, settings, res);
        step = check_bounds(step, settings);
        if(settings.save)
        {
          trajectory << step.transpose().format(CommaInitFmt) << std::endl;
        }
      }
    res.minimiser = step;
    res.minimum = f.evaluate(step);
    if(settings.save){trajectory.close();}
    return res;
  }
private:
    Eigen::VectorXd update(ObjectiveFunction& f, const Eigen::VectorXd& step,
     const GDSettings& settings, Result& res)
     {
       Eigen::VectorXd df = f.gradient(step);
       Eigen::VectorXd next_step = step - settings.gamma*df;
       // The relative change in solution x_i and x_{i-1}
       res.rel_sol_change = abs(f.evaluate(next_step) - f.evaluate(step)/f.evaluate(step));
       // L2 norm of change in gradient from solution x_{i-1} to x_i
       res.grad_norm =  df.norm();
       res.iterations++;
       return next_step;
     }
};
#endif
