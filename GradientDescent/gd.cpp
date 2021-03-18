#include "..\testfun.hpp"
#include<iostream>
#include<Eigen/Dense>

// Use template to allow any type to be passed to gd
template<typename ObjFunc>
Eigen::VectorXd gd(ObjFunc f, const int max_iter, const Eigen::VectorXd& int_val, const double rel_sol_change_tol,
                const double grad_change_tol, const double gam){
     // Learning rate is fixed value at the moment. Could change this?
    // Intialise the step, which will be x_i and step_prev which will be x_{i-1} at...
    //.. each step of the gd algorithm
    Eigen::VectorXd step;
    Eigen::VectorXd step_prev = int_val;
    std::cout<<step_prev<<std::endl;

    //  Init df and df_prev which are \nabla F(x_i) and \nabla F(x_{i-1}) respectively.
    const int dim = int_val.size();
    Eigen::VectorXd df_prev(dim);
    Eigen::VectorXd df(dim);

    // We need to fill df_prev for the algorithm to work. Set every element equal to 100 (any big number will do)..
    df_prev.fill(100);

    // ... the same with the relative change in the solution and the change in the gradient...
    // ... for x_{i-1} to x_i
    double rel_sol_change = 100;
    double grad_change = 100;

    int iter = 0;

    // Whilst the max number of iterations has not been reached, the relative change in the solutions and ...
    // ... the change in the gradient are both less than the prescribed tol we look...
    // ... hopefully getting closer and closer to the true minimiser.
    while(iter < max_iter && rel_sol_change > rel_sol_change_tol && grad_change > grad_change_tol){
        df = f.gradient(step_prev);                                                     // Calculate \nabla F(x_{i-1})...
        step = step_prev - gam*df;
        rel_sol_change = abs(f.evaluate(step_prev) - f.evaluate(step)/f.evaluate(step));    // The relative change in solution x_i and x_{i-1}
        // L2 norm of change in gradient from solution x_{i-1} to x_i
        grad_change =  ((df - df_prev).norm())/df.norm() ;
        // Then update the steps (x_i's) and gradients (df's) for the next iteration.
        // std::cout<< " Current Step is :" << step << std::endl;
        step_prev = step;
        df_prev = df;
        iter++;
    }
    std::cout << "\nFound minimum in " << iter << " steps" << std::endl;
    return step; // When one condition is not met we return the step (x_i value) at this point.
}

int main()
{
    const int d = {4};
    double gam = 0.0001; // Set the learning rate
    int n_iter = 100000; // Set the maximum number of iterations
    double grad_change_tol = pow(10,-10); // Set the tolerance for change of gradient
    double rel_sol_change_tol = pow(10,-10); // Set the tolerance for change of solution
    // Initialise vectors to hold the minimum value to be found and the intial values (which need to be inputed)
    Eigen::VectorXd min_val(d);
    Eigen::VectorXd int_vals(d);
    Quadratic f(d);
    // Set intial values. This could be anything.
    int_vals.fill(3);
    // Use the gradient descent algorithm to calculate the minimum.
    min_val = gd(f,n_iter, int_vals, grad_change_tol, rel_sol_change_tol, gam);

    std::cout << "\nAfter " << n_iter <<
     " iterations and with a change in gradient tolerance of " <<
     grad_change_tol << "\nand a change in relative solution tolerance of " <<
     rel_sol_change_tol << "\nthe minimiser is:\n" << min_val;

    std::cout << "\nThe objective value at this point is "
                        << f.evaluate(min_val) << std::endl;
    return 0;

}
