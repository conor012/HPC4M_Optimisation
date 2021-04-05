/* TESTFUN.HPP
Optimisation test functions including base ObjectiveFunction.
If gradient is not given analytically, it will be calculated by a centred finite difference.
*/
#ifndef TESTFUN
#define TESTFUN

// Base Class for defining an objective function (from R^N to R) to be minimised.
// SHOULD NOT BE USED DIRECTLY, only inherited by objective functions
class ObjectiveFunction {
public:
    // Approximates the gradient in each direction at a point x
    Eigen::VectorXd gradient(const Eigen::VectorXd& x)
    {
        const double EPSILON = pow(10,-6);
        // Create a vector that is zero everywhere except in dimension i where it
        // is epsilon.
        Eigen::VectorXd grad(x.size());
        Eigen::VectorXd epsilon_i;
        epsilon_i.setZero(x.size());
        for(int i=0; i<x.size(); ++i){
            epsilon_i(i) = EPSILON;
            Eigen::VectorXd x_up = x + epsilon_i;
            Eigen::VectorXd x_down = x - epsilon_i;
            // First order central difference approx.
            grad(i) = (evaluate(x_up) - evaluate(x_down)) / (2*EPSILON);
            // Reset epsilon back to zero
            epsilon_i.setZero(x.size());
        }
        return grad;
    }
    // Needs to be declared here, but will be overloaded in any subclass
    virtual double evaluate(const Eigen::VectorXd& x)=0;
};

// Quadratic function x^Tx, gradient implemented exactly. Global min at x=0
class Quadratic: public ObjectiveFunction{
public:
    Quadratic(const int d){}
    double evaluate(const Eigen::VectorXd& x)
    {
        // In 2D, x^2+y^2 (or x^T * x)
        return x.squaredNorm();
    }
    Eigen::VectorXd gradient(const Eigen::VectorXd& x)
    {
        return 2*x;
    }
};

// 1 dimension double well, overloads with exact gradient. Global min x = +-1/sqrt(2)~0.70711
class DoubleWell1D: public ObjectiveFunction{
public:
    DoubleWell1D(const int d){assert(d==1 && "Only well-defined for 1 dimension");}
    double evaluate(const Eigen::VectorXd& x)
    {
        // x^4 - x^2, .sum() is a hack to return a double instead of vector
        return (x.array().pow(4) - x.array().square()).sum();
    }
    Eigen::VectorXd gradient(const Eigen::VectorXd& x)
    {
        // 4x^3 - 2x
        return 4*x.array().pow(3) - 2*x.array();
    }
};


// Double well with two minima, global min at x = (3+sqrt(41))/8 ~ 1.17539
class UnevenDoubleWell1D: public ObjectiveFunction{
public:
    UnevenDoubleWell1D(const int d){assert(d==1 && "Only well-defined for 1 dimension");}
    double evaluate(const Eigen::VectorXd& x)
    {
        // x^4 -x^3 - x^2  .sum() returns double instead of Vector
        return (x.array().pow(4) - x.array().cube() - x.array().square()).sum();
    }
    Eigen::VectorXd gradient(const Eigen::VectorXd& x)
    {
        // 4x^3 - 3x^2 - 2x
        return 4*x.array().pow(3) - 3*x.array().square() - 2*x.array();
    }
};


// Eggholder function, global minimum at x = (512,404.2319)
class Eggholder: public ObjectiveFunction{
public:
    Eggholder(const int d){assert(d==2 && "Eggholder only defined for d==2");}
    double evaluate(const Eigen::VectorXd& x)
    {
        // std::this_thread::sleep_for (std::chrono::microseconds(1));  // Uncomment to add time restriction to code
        return -(x(1)+47)*sin(sqrt(abs((x(0)/2) + x(1)+47))) - x(0)*sin(sqrt(abs(x(0) - x(1) - 47)));
    }
};

// Shekel function
// Needs restricting to  [-10,10] where minimum is at (4,4,4,4)
class Shekel: public ObjectiveFunction{
public:
    Shekel(const int d){assert(d==4 && "Shekel only defined for d==4");}

    double evaluate(const Eigen::VectorXd& x)
    {
        int m = 10;
        Eigen::MatrixXd C(4,10);
        Eigen::MatrixXd  beta(10,1);
        C <<  4, 1, 8, 6, 3, 2, 5, 8, 6, 7,
              4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6,
              4, 1, 8, 6, 3, 2, 5, 8, 6, 7,
              4 ,1, 8, 6, 7, 9, 3, 1, 2, 3.6;
        beta << 1, 2, 2, 4, 4, 6, 3, 7, 5, 5;
        beta /= 10;
        double outer = 0;
        for(int i=0; i<m;++i)
        {
            double inner = 0;
            for(int j=0; j<4;++j)
            {
                inner += (pow((x(j) - C(j,i)),2));
            }
            outer += 1/(inner+ beta(i));

        }
        outer = -outer;
        return outer;
    }
};

#endif
