#ifndef GD
#define GD

std::map<std::string, int> gd_methods = {
  { "vanilla"  , 1 },
  { "momentum" , 2 },
  { "Nesterov accelerated gradient descent" , 3}
};
struct GDSettings: OptimiserSettings
{
  //const int method {gd_methods["vanilla"]};
  int method {gd_methods["vanilla"]};
};

// Redefine insertion operator to include method name
std::ostream& operator<<(std::ostream& os, const GDSettings& settings)
{
  std::map<int,std::string> reverse_gd_methods = {
    { 1, "vanilla" },
    { 2, "momentum" },
    { 3, "Nesterov accelerated gradient descent"}
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
    Eigen::VectorXd direc(int_val.size()), adam_vec_v(int_val.size());
    direc.fill(0);
    adam_vec_v.fill(0);
    res.adam_vec_v = adam_vec_v;
    res.direc = direc;

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
      //Eigen::VectorXd direc =res.direc;
      Eigen::VectorXd df = f.gradient(step);
      // Update direction based on if settings.method = 1 (vanilla) or 2 (momentum)
      res.direc = direc_update(f,step, settings, df, res.direc, res.adam_vec_v);
      //Eigen::VectorXd direc_new = direc_update(step, settings, df, direc);
      Eigen::VectorXd next_step = step - res.direc;//settings.gamma*df;
       // The relative change in solution x_i and x_{i-1}
       res.rel_sol_change = abs(f.evaluate(next_step) - f.evaluate(step)/f.evaluate(step));
       // L2 norm of change in gradient from solution x_{i-1} to x_i
       res.grad_norm =  df.norm();
       res.iterations++;
       return next_step;
     }
   //Updates direction for momentum / vanilla / NAG
   Eigen::VectorXd direc_update(ObjectiveFunction& f, const Eigen::VectorXd& step,
     const GDSettings& settings, const Eigen::VectorXd& df, const Eigen::VectorXd&direc,
     const Eigen::VectorXd& adam_vec_v)
     {
       Eigen::VectorXd direc_out;
       switch (settings.method)
    {
        case 1: // vanilla
        {
            direc_out = settings.gamma * df;
            break;
        }

        case 2: // momentum
        {
            // direc_out = gd_settings.par_step_size * (gd_settings.par_momentum * direc + grad_p);
            direc_out = settings.par_momentum * direc + settings.gamma * df;
            break;
        }
        case 3: // Nesterov accelerated gradient
        {
            Eigen::VectorXd NAG_grad( step.size() );
            NAG_grad = f.gradient(step - settings.par_momentum * direc);
            //direc_out = settings.gamma* (settings.par_momentum * direc + NAG_grad);
            direc_out = settings.par_momentum * direc + settings.gamma * NAG_grad;
            break;
        }
        //case 4: // AdaGrad
        //{
        //    int exponent;
        //    std::fill_n(exponent, step.size(), 2);
        //    adam_vec_v = step.array().pow(exponent).matrix();
        //    //adam_vec_v += VectorXcd::Map((df.cwiseAbs2()).data(), (df.cwiseAbs2()).size());
            //direc_out = OPTIM_MATOPS_ARRAY_DIV_ARRAY( settings.gamma * df, OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_SQRT(adam_vec_v), settings.par_ada_norm_term) );
        //    direc_out = ( settings.gamma * df) / (sqrt(adam_vec_v.cwiseSqrt()) + settings.par_ada_norm_term) );

        //    break;
        //}
     }
      return direc_out;
     }
};
#endif
