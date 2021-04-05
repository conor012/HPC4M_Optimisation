/* BFGS.HPP
Implementation of Broyden-Fletcher-Goldfarb-Shanno algorithms using an adaptive step size
set by the Armijo condition.

Should only be included indirectly through HPC_Opt.hpp.

For example usage see test/testBFGS.cpp
See also:
  - misc.hpp for implementation of base classes.
  - test/testParPSO.cpp for a swarm algo with BFGS final stage
*/
#ifndef BFGS_OP
#define BFGS_OP

struct BFGSSettings: OptimiserSettings{};

class BFGS: public BaseOptimiser{
public:
  double st_q, qt_D_q;
  const double max_bound = 512;
  int dim;
  Result minimise(ObjectiveFunction& f, const Eigen::VectorXd& int_val, const BFGSSettings settings)
  {
    Result res;
    // Initialise vectors to hold xk, xk+1 and print initial value.
    dim = int_val.size();
    Eigen::VectorXd next_step(dim), df(dim), next_df(dim);
    Eigen::VectorXd step = int_val;

    // Useful constants for updates
    Eigen::VectorXd s(dim), q(dim), w(dim);
    // Initialise relative changes to large number
    res.rel_sol_change = 2*settings.rel_sol_change_tol;
    res.grad_norm = 2*settings.grad_norm_tol;

    // Initialise matrix D as the identity matrix
    Eigen::MatrixXd D = Eigen::MatrixXd::Identity(dim, dim);
    Eigen::VectorXd p(dim);
    if(settings.save){create_save_file();}

    // Compute initial gradient
    df = f.gradient(step);
    // While max iteraitons has not been reached and all change tolerances have
    // not been met, perform bfgs update
    const double c =1e-4, tau = 0.95;
    while (res.iterations < settings.max_iter &&
           res.rel_sol_change > settings.rel_sol_change_tol &&
           res.grad_norm > settings.grad_norm_tol)
      {
        Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
        p= -D*df;
        // Armijo condition
        double gamma = 2;
        while(f.evaluate(step + gamma*p) > (f.evaluate(step) + gamma*c*p.transpose()*df)){
          gamma *= tau;
        }
        // Update step
        next_step = step + gamma*p;
        // Project iteration back into hypercube if needed
        next_step = check_bounds(next_step, settings);

        // Compute new gradient and relative changes
        next_df = f.gradient(next_step);
        res.rel_sol_change = abs((f.evaluate(next_step) - f.evaluate(step))/f.evaluate(step));
        res.grad_norm = df.norm();

        // Update the matrix D
        // Compute difference between points and gradients at k and k+1.
        s = next_step - step;
        q = next_df - df;
        st_q = s.transpose()*q;
        qt_D_q = abs(q.transpose()*D*q); // Small values may cause erratic negative values
        w = sqrt(qt_D_q)*(s/st_q - D*q/qt_D_q);
        D += s*s.transpose()/st_q - D*q*q.transpose()*D/qt_D_q + w*w.transpose();
        // Update iterate values
        step = next_step;
        if(settings.save)
        {
          trajectory << step.transpose().format(CommaInitFmt) << std::endl;
        }
        df = next_df;
        res.iterations++;
      }
    res.minimiser = step;
    res.minimum = f.evaluate(step);
    if(settings.save){trajectory.close();}
    return res;
  }
};
#endif
