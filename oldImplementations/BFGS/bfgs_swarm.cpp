// Test BFGS method on problems (serial)
#include <iostream>
#include <cmath>
#include <fstream>
#include<Eigen/Dense>
#include <iomanip>
#include "../testfun.hpp"

using namespace std;

template<typename ObjFunc> // Use the parameter 'max_bound' to define edge of hypercube to optimse in
Eigen::VectorXd bfgs_swarm(ObjFunc f, const int max_iter,
   const double max_bound, const int num_particles, int dim){
    // Initialise vectors to hold xk, xk+1 and print initial value.
    Eigen::VectorXd step_prev [num_particles], step [num_particles], df_prev[num_particles], df[num_particles];
    Eigen::VectorXd s(dim), q(dim), w(dim), x_opt(dim);

    double f_opt = pow(10, 16); // Store current min of f



    // Useful constants for updates
    double st_q, qt_D_q;


    // Initialise matrix D as the identity matrix (for each particle)
    Eigen::MatrixXd D [num_particles];

    int iter = 0;  // Initial no. iterations

    // Randomise initial points
    for (int n = 0; n < num_particles; ++n){
      step_prev[n] = max_bound*Eigen::MatrixXd::Random(dim,1); // Set initial point
      // Compute initial gradient
      df_prev[n] = f.gradient(step_prev[n]);
      D[n] = Eigen::MatrixXd::Identity(dim, dim);
      // Update optimal f, x if necessary
      if (f.evaluate(step_prev[n]) < f_opt){x_opt = step_prev[n]; f_opt = f.evaluate(x_opt);}
    }


    // While max iteraitons has not been reached and all change tolerances have
    // not been met, perform bfgs update
    Eigen::VectorXd p(dim);
    const double c =1e-4, tau = 0.95;
    while (iter < max_iter){
      for (int n = 0; n<num_particles; ++n){
        // Update step
        p= 0.5*(-D[n]*df_prev[n] + x_opt - step_prev[n]);
        double gam = 2;
        while(f.evaluate(step_prev[n] + gam*p) > (f.evaluate(step_prev[n]) + gam*c*p.transpose()*df_prev[n])){
          gam *= tau;
        }
        step[n] = step_prev[n] + gam*p;
        // Project iteration back into hypercube if needed
        for (int i=0; i<dim; ++i){
          if (step[n](i) < -max_bound){step[n](i) = -max_bound;}
          else if (step[n](i)>max_bound){step[n](i)= max_bound;}
        }

        // Compute new gradient and relative changes
        df[n] = f.gradient(step[n]);

        // Compute difference between points and gradients at k and k+1.
        s = step[n] - step_prev[n];
        q = df[n] - df_prev[n];

        // Update the matrix d
        st_q = s.transpose()*q;
        qt_D_q = abs(q.transpose()*D[n]*q); // Small values may cause erratic negative values
        w = sqrt(qt_D_q)*(s/st_q - D[n]*q/qt_D_q);
        D[n] += s*s.transpose()/st_q - D[n]*q*q.transpose()*D[n]/qt_D_q + w*w.transpose();
        // Update iterate values
        step_prev[n] = step[n];
        df_prev[n] = df[n];
        if (f.evaluate(step_prev[n]) < f_opt){
          x_opt = step_prev[n];
          f_opt = f.evaluate(x_opt);
          cout << "\n-----------------------\n" << "New optimal value f = " << f_opt << "\n at: \n" << x_opt << "\n";
        }
        ++iter;
      }
    }
    // cout << "gamma is" << gamma << endl;
    return x_opt;
  }

int main (){
  const int d = 2, max_iter = pow(10, 5);  // Dimensionality and max iterations
  Eggholder f(d); // Define problem function
  double max_bound = 512;  // domain boundaries
  int num_particles = 1000; // number of particles

  double minimum_f = pow(10,10), f_temp;  // Store the min f over all runs (f_temp used as placeholder below)
  Eigen::VectorXd overall_min(d);  // Store the minimum point over all runs

  overall_min = bfgs_swarm(f, max_iter, max_bound, num_particles, d);
}
