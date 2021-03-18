# Gradient Descent- Serial
The gradient descent (GD) algorithm aims to solve the optimisation problem
```
 \min_{x \in X} f(x)
```
for a vector space X and a convex and differentiable function f.


The updating rule works as so. Let x^{i} be the candidate solution at stage i of the algorithm and
select a step size \gamma.
At stage i:

1. Update  d^{i} via the scheme chosen from the options below- at the moment d^{i} = \gamma \times \nabla_x f ( x^{i} );
2. Update the sequence of candidate solution
```
x^{i+1} = x^{i} - d^{i}.
```
3. If the optimasation problem is on a constrained domain and x^{i+1} is outside this domain then it is projected back onto the domain.

The algorithm continues until one of the three conditions are met:

1. The maximum number of iterations are exceeded.

2. The norm of the gradient ``` \|\nabla f \| ```
is less than the prescribed gradient tolerane ```grad_change_tol```.
3. The relative change between ```x^{i+1}``` and ```x^{i}``` is less than the prescribed tolerance
```rel_sol_change_tol```.

# Gradient Descent- Parallel

If f is not a convex function then serial GD is susceptible to finding a local minima rather than a global minima.

At the moment each processor P has a random initial value, constrained to the domain relevant to the question. 

Each processor then computes a local minima using the serial GD algorithm. The root processor is then sent the local minimums from each processor. It calculates the minimum value of these local minimums and returns this value.

Note that this does not ensure we have a global minimum. We should look at how to pick the initial points effectively in order to increase the possibility of finding the global minimum.