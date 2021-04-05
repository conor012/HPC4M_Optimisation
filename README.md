# HPC4M Global Minima of Benchmark Optimisation Problems

Aim: Implement algorithms to find the global minima of the Eggholder function and Shekel function exploiting parallel architectures in an orderly C++ way.

> Repository for Maxwell Institute Advanced Course: High Performance Computing for Mathematicians

## Getting Started

These implementations use [Eigen 3.3.9](https://eigen.tuxfamily.org/index.php?title=Main_Page) for its linear algebra. I place Eigen in the root directory – it is ignored on pushing (See `.gitignore`). Then to compile the file `test/testGD.cpp` with Eigen using g++ on Windows,

```
g++ -I .\eigen-3.3.9 test/testGD.cpp -o gd.exe
```
On Linux you'll just need to change the file separators to `/`.

For example usage of each algorithm, see `test/test*.cpp`.

## Test Objective Functions

All the functions `f` we are interested in map from R^d to R. To use them, simply declare them as `FunctionName f(settings.dim)` where `settings.dim` is an `int` describing the dimension of the problem in the relevant `Settings` struct. For example
```
GDSettings settings
settings.dim = 10;
Quadratic f(settings.dim)
```
 There are two associated functions `f.evaluate(x)` and `f.gradient(x)` evaluating the function and its gradient at the point `x` respectively. Below is a list of current test functions:

- `Quadratic`:

    A quadratic function (convex, one minimum at x=0)

    - Defined as f : R^N \to R, f(x) = x^T x. If d=1, this is f(x) = x^2. In 2d, f(x)=  x_0^2 + x_1^2.
    - The gradient is calculated exactly, \grad_x f(x) = 2x

- `DoubleWell1D`

    A one-dimensional double well potential with two minima at x =  \pm 1/sqrt(2). Asserts that the given dimension is equal to 1.

    - Defined as f : R \to R, f(x) = x^4 - x^2
    - Gradient is calculated exactly, \grad_x f(x) = 4x^3 - 2x

- `UnevenDoubleWell1D`

    A one-dimensional double well that has one deeper well, global minimum at x = (3+sqrt(41))/8 \approx 1.1754. Asserts the given dimension is equal to 1.

    - Defined as f : R \to R, f(x) = x^4 - x^3  - x^2
    - Gradient is calculated exactly, \grad_x f(x) = 4x^3 - 3x^2 - 2x

- `Eggholder`

    The [Eggholder function](http://www.sfu.ca/~ssurjano/egg.html) in two dimensions for x \in [-512,512]^2. Global minimum at x = (512,404.2319).

    - Defined as f : R^2 \to R, f(x) = -(x_1 + 47)sin(sqrt(abs((x_0 /2) + x_1+47))) - x_0 sin(sqrt(abs(x_0 - x_1 - 47)))
    - Gradient calculated numerically using `BaseObjective::gradient`

- `Shekel`

    The [Shekel function](http://www.sfu.ca/~ssurjano/shekel.html) for x \in [0,10]^4. There are 10 local minima with the global minimum at x = (4, 4, 4, 4).

    - Gradient calculated numerically using `BaseObjective::gradient`.

## Using the Test Functions

See `testGD.cpp` for an example of using the gradient descent algorithm.

## Contributing

All of the below is subject to change as the project develops and the need for more functionality arises.

### Using Eigen

There is a [handy quick reference for Eigen](http://eigen.tuxfamily.org/dox/AsciiQuickReference.txt) . The main type used is `Eigen::VectorXd` – a vector of unknown length (`X`) containing doubles (`d`). These can be written to `cout` directly (`std::cout << v ;`).

One thing to be careful of is whether operations should be done using vector arithmetic or elementwise arithmetic. If you require the latter use `v.array()`. This is used to take elementwise powers, among other things. Some common elementwise operations have their own functions – check the documentation before using `.array()`.

### Implementing Test Functions

All test functions are classes defined in `testfun.hpp`, containing two methods:

- `double evaluate(const Eigen::VectorXd& x)`  
    - Evaluates the function at the given point `x`, f(x)
- `Eigen::VectorXd gradient(const Eigen::VectorXd& x)`
    - Calculates the gradient of the function at the given point `x` . \grad_x f(x)

The test functions all inherit the base class `BaseObjective`, which takes as it's only argument the dimension of the function `d` . *This class should not be used directly, only inherited.*   

- `BaseObjective`:

    - `gradient` is approximated using centred finite differences with step size `EPSILON`.

    - `evaluate` is a virtual function that should be overloaded in any subclass. If called directly it will terminate the program.



### Implementing Methods

Methods should be in their own file, `algo.cpp`. Methods should take:

-  an objective function `f` (`ObjectiveFunction`)
- an initial value `int_val` (Eigen::VectorXd)
- a settings structure matching the algorithm name  (`OptimiserSettings`)

They should return a vector `step` that is the point at which the minimum of the objective function `f` occurs.
The possible settings depend on the algorithm to be used. For example, the gradient descent settings  `GDSettings` are the default set `OptimiserSettings` with
- `method` an integer between 1 and 3 where 1 is a standard gradient descent, 2 is gradient descent with momentum and 3 is the Nesterov accelerated gradient descent. For ease, these can also be named using `gd_methods["vanilla"]`, `gd_methods["momentum"]` or `gd_methods["Nesterov accelerated gradient descent"]` respectively.

The standard settings are
-  `max_iter` the maximum number of iterations before stopping (int)
- `rel_sol_change_tol` the minimum relative change in the minimum after the final iteration (double)
-  `grad_norm_tol` the minimum desired change in the gradient in the final iteration (double)
-  `gamma` step size/learning rate. Some algorithms have adaptive step sizes overriding this value (double)
-  `save` whether or not to save the trajectory of the algorithm (useful for plotting) (bool)
-  `max_bound` the upper boundary of the domain of interest(double)
-  `min_bound` the lower boundary of the domain of interest (double)
-  `dim` the dimension of the objective function (int)


For a deeper dive into the theory and some scaling analysis, see `FinalReport.pdf`.


## Future Work
### Basic Algorithms

Algorithms:
- [x] Gradient Descent
- [ ] Stochastic Gradient Descent
- [x] BFGS
- [ ] Nelder-Mead

### Algorithms Designed for Parallel

- [x] Particle Swarm -- better to hunt as a swarm or packs (i.e. local or global interaction)?
- [ ] Ant/Bee Colony
- [ ] Genetic/Evolutionary Algorithms
