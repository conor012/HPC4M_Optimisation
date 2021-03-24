#include "HPC_Opt.hpp"
#include <mpi.h>
int main()
{
    int rank;
    int size;
    const int root = 0;
    MPI_Status status;
    GDSettings settings;

    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Comm_size(comm, &size);
    if(size < 2){
        std::cout << "Error: Need at least 2 processes!" << std::endl;
        MPI_Finalize();
        return 0;
    }
    MPI_Comm_rank(comm, &rank);

    const int d = {2};
    settings.gamma = 0.0001; // Set the learning rate
    settings.max_iter = 100000; // Set the maximum number of iterations
    settings.grad_norm_tol = pow(10,-10); // Set the tolerance for norm gradient
    settings.rel_sol_change_tol = pow(10,-10); // Set the tolerance for change of solution
    settings.max_bound = 512; // domain boundaries
    settings.min_bound = -512;

    // Initialise vectors to hold the minimum value to be found and the intial values (which need to be inputed)
    Eigen::VectorXd min_val(d);
    Eigen::VectorXd int_vals(d);
    Eggholder f(d);

    // Set intial values. At the moment this selects them randomly but that could be changed.
    // Seed the time differently for each process
    unsigned seed = time(0) + rank;
    // Seed the random number generator.
    srand(seed);
    // Fill intial values with random integers in domain
    int_vals = settings.max_bound * Eigen::VectorXd::Random(d);
    std::cout << int_vals << std::endl;

    // Use the gradient descent algorithm to calculate the minimum (comment max_bound if not needed).
    GradientDescent gd;
    Result res = gd.minimise(f, int_vals, settings);

    // All processors except the root send the min_val they have found to the root
    if(rank!=root){
        MPI_Send(&res.minimiser[0], d, MPI_DOUBLE, 0, 1, comm);
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

    return 0;

}
