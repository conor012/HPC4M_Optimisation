#ifndef MISC
#define MISC

struct Result
{
  Eigen::VectorXd minimiser ;
  double minimum {NAN} ;
  int iterations {0};
  double grad_norm {NAN};
  double rel_sol_change {NAN};
  friend std::ostream& operator<<(std::ostream& os, const Result& res);
};

struct OptimiserSettings
{
  int max_iter {10000};
  double rel_sol_change_tol {1e-6};
  double grad_norm_tol {1e-6};
  double gamma {0.001};
  bool save {false};
  double max_bound {INFINITY};
  double min_bound {-INFINITY};
  friend std::ostream& operator<<(std::ostream& os, const OptimiserSettings& settings);
};

std::ostream& operator<<(std::ostream& os, const OptimiserSettings& settings)
{
  return os << "Max Iterations: " << settings.max_iter << std::endl
            << "Relative Solution Change Tolerance: " << settings.rel_sol_change_tol << std::endl
            << "Norm of Gradient Tolerance: " << settings.grad_norm_tol << std::endl
            << "Step size: " << settings.gamma << std::endl
            << "Save data? " << (settings.save ? "Yes" : "No") << std::endl
            << "Restricted to the hypercube [" << settings.min_bound << "," << settings.max_bound << "]";

}

std::ostream& operator<<(std::ostream& os, const Result& res)
{
  return os << "\nMinimiser is: \n" << res.minimiser << std::endl
            << "Minimum is: " << res.minimum << std::endl
            << "Iterations: " << res.iterations << std::endl
            << "Change in gradient at final step: " << res.grad_norm << std::endl
            << "Relative change in minimiser at final step: " << res.rel_sol_change << std::endl;
}

class BaseOptimiser
{
public:
  // Declare filename and stream but do not assign unless save is on
  std::ofstream trajectory;
  char const* p_filename = nullptr;
  void create_save_file()
  {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%d_%m_%Y_%H%M%S");
    auto date  = oss.str();
    std::string filename = "trajectory";
    std::cout << "Data will be saved in: " << std::endl
    << filename.append(date + ".csv") << std::endl;
    p_filename = filename.c_str() ;
    trajectory.open(p_filename);
  }


};
#endif
