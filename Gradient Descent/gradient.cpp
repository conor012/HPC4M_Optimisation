#include <iostream>
#include <vector>
#include <math.h> 
#include <numeric>
#include <cstdlib>

using namespace std;

const double epsilon = pow(10,-6); // Fix a small epsilon for finite difference scheme to calculate grad

// This is the function to be minimised. At the moment it is f(x) = x^2
double objfun(vector<double>& x){               // Takes a vector x...
    double output_val = 0.0;
    for(int i=0 ; i < x.size(); i++){
        output_val += pow(x[i],2);}             //... and computes x_i^2 for each coordinate...
    return output_val;                      //... and returns vector of calculation.
}

// This another function to be minimised. Uncomment out if
// you want to use it. At the moment it is f(x) = x^2 + x^4 + 15
//double objfun(vector<double>& x){               // Takes a vector x...
//    double output_val = 0.0;
//    for(int i=0 ; i < x.size(); i++){
//        output_val += pow(x[i],2) + pow(x[i],4);}             //... and computes x_i^2 +x_i^4 + 15 for each coordinate...
//    return output_val + 15;                      //... and returns vector of calculation.
//}


// Function to calculate the gradient of the objective function
vector<double> grad(vector<double>& input_vals){        // Takes a vector of type double as input values
    vector<double> grad_out;                            // Gradient vector to be returned
    vector<double> x_up = input_vals;                   // Vector to keep x_i + epsilon in the ith component
    vector<double> x_down = input_vals;                 // Vector to stall x_i - epsilon in the ith component
    for(int i=0; i < input_vals.size(); i++){        // Loop calculates derivative in each direction...
        x_up[i] += epsilon;                          //...using a finite difference scheme...
        x_down[i] -= epsilon;
        grad_out.push_back((objfun(x_up) - objfun(x_down)) / (2 * epsilon));  // (this epsilon was fixed above)
        x_up[i] -= epsilon;
        x_down[i] += epsilon;
    }
    return grad_out;                                 //... and then returns the gradient vector
}

// Function to print vectors of any length
void print_vector(vector<double> data)
{
    for(int i = 0 ; i < data.size(); i++)
    {
        cout << data[i] << "\t";
    }
    cout << endl;
}

// Function to find the L2 norm of a vector
double l2_norm(vector<double> &x)
{
    double norm = 0.0;
    for(int i=0; i<x.size();i++){
        norm += x[i] * x[i];
    }
    return sqrt(norm);
}

// Function to subtract vector y from vector x
vector<double> subtract_vect(vector<double> x, vector<double> y){
    vector<double> output;
    for(int i = 0; i < x.size();i++){
        output.push_back(x[i]-y[i]);
    }
    return output;

}

// Function to multiply every element in a vector x by a scalar k
vector<double> multiplication_vectorscal(vector<double> &x, double k){
    for (int i = 0; i < x.size() ; i++)
    {
        x[i] *= k;
    }
    return x;
}

// The gradient descent algorithm
// Takes in a maximum number of iterations, a vector of intial values,
// two tolerances (one for gradient and one for relative change in solution)
// and gam which is the learning rate.
vector<double> gd(int max_iter, vector<double> int_val, double rel_sol_change_tol,  
                double grad_change_tol, double gam){  // Learning rate is fixed value at the moment. Could change this?
    // Intialise the step, which will be x_i and step_prev which will be x_{i-1} at...
    //.. each step of the gd algorithm
    vector<double> step;
    vector<double> step_prev = int_val;

    //  Init df and df_prev which are \nablda F(x_i) and \nablda F(x_{i-1}) respectively.
    //  Along with their difference df_dfprev.
    vector<double> df_prev;
    vector<double> df;
    vector<double> df_dfprev;

    // We need to fill df_prev for the algorithm to work. Set every element equal to 100 (any big number will do)..
    for(int i=0; i < int_val.size(); i++){
        df_prev.push_back(100);}
    // ... the same with the relative change in the solution and the change in the gradient...
    // ... for x_{i-1} to x_i
    double rel_sol_change = 100;
    double grad_change = 100;

    int iter = 0;

    // Whilst the max number of iterations has not been reached, the relative change in the solutions and ...
    // ... the change in the gradient are both less than the prescribed tol we look...
    // ... hopefully getting closer and closer to the true minimiser.
    while(iter < max_iter && rel_sol_change > rel_sol_change_tol && grad_change > grad_change_tol){
        df = grad(step_prev);                                                     // Calculate \nabla F(x_{i-1})...
        step = subtract_vect(step_prev, multiplication_vectorscal(df, gam));      // ... set x_i = x_{i-1} - gamma * \nabla F(x_i)
        
        
        rel_sol_change = abs(objfun(step_prev) - objfun(step));    // The relative change in solution x_i and x_{i-1}
        df_dfprev = subtract_vect(df, df_prev);                    // The change in gradient from solution x_{i-1} to x_i...
        grad_change = l2_norm(df_dfprev);                          //... in the form of the L2 norm between them.

        // Then update the steps (x_i's) and gradients (df's) for the next iteration.
        step_prev = step;
        df_prev = df;
        iter++;
    }
    cout << "\nFound minimum in " << iter << " steps" << endl;
    return step; // When one condition is not met we return the step (x_i value) at this point.
}

int main(){
    double gam = 0.0001; // Set the learning rate
    int n_iter = 100000; // Set the maximum number of iterations
    double grad_change_tol = pow(10,-10); // Set the tolerance for change of gradient
    double rel_sol_change_tol = pow(10,-10); // Set the tolernace for change of solution
    
    // Initialise vectors to hold the minimum value to be found and the intial values (which need to be inputed)
    vector<double> min;
    vector<double> int_vals;

    // Set intial values. This could be anything. Have chosen x_i = 2^i for i=1,2,3,4.
    for (int i = 0; i < 4; i++)
    {
    int_vals.push_back(pow(2,i));
    }

    // Use the gradient descent algorithm to calculate the minimum.
    min = gd(n_iter, int_vals, grad_change_tol, rel_sol_change_tol, gam);

    cout << "\nAfter " << n_iter << " iterations and with a change in gradient tolerance of " << grad_change_tol <<
          "\nand a change in relative solution tolernace of "<< rel_sol_change_tol  << "\nthe minimiser is -\n" << endl; 
    print_vector(min);
    cout << "\nThe objective value at this point is "
                         << objfun(min) << endl;
    return 0;
}