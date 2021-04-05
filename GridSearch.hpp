#ifndef GRIDSEARCH
#define GRIDSEARCH

class GridSearch: public BaseOptimiser{
  public:
    Result minimise(ObjectiveFunction& f, const Eigen::VectorXd& int_val, const OptimiserSettings settings){
      Result res;
      assert(settings.dim==2 && "GridSearch only implemented for two dimensions");
      res.rel_sol_change = 2*settings.rel_sol_change_tol;
      res.grad_norm = 2*settings.grad_norm_tol;
      double side_points = pow(settings.max_iter, 1.0/settings.dim);
      double side_length = settings.max_bound - settings.min_bound;
      Eigen::VectorXd min_point {INFINITY};
      Eigen::VectorXd side {Eigen::VectorXd::LinSpaced(side_points, settings.min_bound, settings.max_bound)};
      Eigen::VectorXd point;
      for(int i=0;i<side_length;i++)
      {
        for(int j=0;j<side_length;j++)
        {
          point << side(i), side(j);
          std::cout << point << "\n";
          double f_point = f.evaluate(point);
          if(f_point<f.evaluate(min_point)){min_point=point;}
        }
      }

      // Configure output format for vectors
      return res;
    }
};
#endif
