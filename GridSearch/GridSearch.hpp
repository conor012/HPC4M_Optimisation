#ifndef GRIDSEARCH
#define GRIDSEARCH

class GridSearch: public BaseOptimiser{
  public:
    Result minimise(ObjectiveFunction& f, const OptimiserSettings settings){
      Result res;
      assert(settings.dim==2 && "GridSearch only implemented for two dimensions");
      // Number of points and length of domain
      double side_points = pow(settings.max_iter, 1.0/settings.dim);
      double side_length = settings.max_bound - settings.min_bound;
      // Vectors to hold each point and the minimum known point
      Eigen::VectorXd min_point {settings.dim}, point(settings.dim);
      
      // Side points
      Eigen::VectorXd side = Eigen::VectorXd::LinSpaced(side_points, settings.min_bound, settings.max_bound);
      
      for(int i=0;i<side_points;i++)
      {
        for(int j=0;j<side_points;j++)
        {
          point << side(i), side(j);
          double f_point = f.evaluate(point);
          if(f_point<f.evaluate(min_point)){min_point=point;}
        }
        res.minimum = f.evaluate(min_point); res.minimiser=min_point;
      }

      // Configure output format for vectors
      return res;
    }
};
#endif
