#ifndef GD
#define GD
# include "mpi.h"

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
  Result minimise(ObjectiveFunction& f, const Eigen::VectorXd& int_val, const GDSettings settings)
  {
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
  Eigen::VectorXd parallel_minimise(ObjectiveFunction& f, const GDSettings settings)
  {
    int rank;
    int size;
    const int root = 0;
    MPI_Status status;

    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Comm_size(comm, &size);
    if(size < 2){
        std::cout << "Error: Need at least 2 processes!" << std::endl;
        MPI_Finalize();
        std::terminate();
    }
    MPI_Comm_rank(comm, &rank);
    const int d = 2;
    // Initialise vectors to hold the minimum value to be found and the intial values (which need to be inputed)
    Eigen::VectorXd min_val(d);
    Eigen::VectorXd int_val[settings.num_particles];
    // Set intial values. At the moment this selects them randomly but that could be changed.
    // Seed the time differently for each process
    unsigned seed = time(0) + rank;
    // Seed the random number generator.
    srand(seed);
    // Randomise initial points
    GradientDescent gd;
    Result res;
    for (int n = 0; n < settings.num_particles; ++n){
        int_val[n] = settings.max_bound*Eigen::VectorXd::Random(d);
        for (int i=0; i<d; ++i){
            if (int_val[n](i) < settings.min_bound){ int_val[n](i) = -int_val[n](i);}
        }
      // Use the gradient descent algorithm to calculate the minimum for each particle
      res = gd.minimise(f, int_val[n], settings);
      // Update each time to find minimum of these particles.
      if(res.minimum < f.evaluate(min_val)){
          min_val = res.minimiser;  
      }
    }
    // All processors except the root send the min_val they have found to the root
    if(rank!=root){
        MPI_Send(&min_val, d, MPI_DOUBLE, 0, 1, comm);
    }

    if(rank == root)
    {
        Eigen::VectorXd buffer(d); // buffer to hold incoming min_vals
        // Root process receives min_vals in any order (this should make the code faster)...
        for(int i=1; i < size; i++)
        {
            MPI_Recv(&buffer[0], d, MPI_DOUBLE, MPI_ANY_SOURCE, 1, comm, MPI_STATUS_IGNORE);
            // ... it then evalutes the min_val recieved from each other process...
            if(f.evaluate(buffer) < f.evaluate(min_val))
            {
                    min_val = buffer;                         // .. and keeps it only if it evaluates to a lower value than the current min_val.
            }
        }
        std::cout << settings << std::endl;
        std::cout<<  "\nthe minimiser is:\n" << min_val;
        std::cout << "\nThe objective value at this point is " <<std::endl
                            << f.evaluate(min_val) << std::endl;
    }

    MPI_Finalize();

    return min_val;
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
