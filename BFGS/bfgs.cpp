// Test BFGS method on problems (serial)
#include <iostream>
#include <cmath>
#include <fstream>

using namespace std;

// Define function operator
// double f (double x[]){  // Eggholder function
//   return -(x[1]+47)*sin(sqrt(abs(x[0]/2 + x[1] + 47)))
//           - x[0]*sin(sqrt(abs(x[0] - x[1] - 47)));
// }

double f (double x[]){  // 1D test function
  return -(x[0]-2)*(x[0]-2);
}

int d = 1; // Problem dimensions
double tol = pow(10, -10);  // Converge when relative dist. between terms < tol
double tolSq = tol*tol;  // Used below

double  gradf (double xk[], double dx, int i){
  double df;
  double xtemp [d];
  for (int j=0; j<d; ++j){
    xtemp[j] = xk[j];
    if (j == i){
      xtemp[j] += dx;
    }
    df = (f(xtemp) - f(xk))/dx;
  }
  return df;
}

double relErr (double  xk[], double xkP1[]){
  double errN = 0, errD = 0;
  for (int i = 0; i < d; ++i){
    errN += pow(xk[i] - xkP1[i],2);
    errD += pow(xk[i], 2);
  }
  return errN/errD;
}

int main (){

  // Initialise problem parameters
  double  xk [d], xkP1 [d], dfk [d], dfkP1 [d], sk [d], qk [d], dkgf [d], xtemp0 [d], xtemp1 [d],
             wk [d];  // Vectors
  double dk [d][d];  // Matrices
  double err = 2*tolSq;  // Error between terms
  double dx = 0.001; // Distance used to approx gradient
  double alpha = 0.1, alphaOpt = 0.1, fopt;  // Used to find optimal movement parameter
  int alphaIts = 5; // No. steps to estimate alpha per iteration
  int k=0; // No. its
  ofstream fileOut;
  fileOut.open("iterations.txt");

  // Initialise matrix dk to the identity
  for (int i=0; i < d; ++i){
    dk[i][i] = 1;
  }
  for (int i=0; i <d; ++i){
    dfk[i] = gradf(xk, dx, i);  // Initial gradient
  }


  //BFGS Updates:
  while (err > tolSq){
    cout << xk[0] << " " << xk[1] << "\n";
    cout << k << "\n";
    ++k;
    // Compute D_k*grad(f)
    for (int i = 0; i < d; ++i){
      for (int j = 0; j < d; ++j){
        dkgf[i] += dk[i][j]*dfk[j];
      }
    }
    for (int j=0; j < d; ++j){
      xtemp0[j] = xk[j] - alpha*dkgf[j];
      xtemp1[j] = xk[j] - (alpha + dx)*dkgf[j];
    }

    // // Find optimal alpha
    // fopt = 2*f(xk);
    // alphaOpt = 1;
    // for (int i = 0; i < alphaIts; ++i){
    //   alpha -= (f(xtemp1) - f(xtemp0))/dx;
    //   for (int j = 0; j < d; ++j){
    //     xtemp0[j] = xtemp1[j];
    //     xtemp1[j] = xk[j] - (alpha + dx)*dkgf[j];
    //   }
    //   if (f(xtemp0) < fopt){
    //     alphaOpt = alpha;
    //     fopt = f(xtemp0);
    //   }
    // }

    // Update x
    for (int i = 0; i < d;  ++i){
      xkP1[i] = xk[i];
      for (int j = 0; j < d; ++j){
        xkP1[i] -= alphaOpt*dk[i][j]*dfk[j];
      }
    }

    // Update D
    for (int i=0; i<d; ++i){
      dfkP1[i] = gradf(xkP1, dx, i);  // Gradient f(x_k+1)
    }

    for (int i = 0; i < d; ++i){  // Update sk, qk
      sk[i] = xkP1[i] - xk[i];
      qk[i] = dfkP1[i] - dfk[i];
    }
    // Compute parameters for update of D
    double  dkqk [d], qkt_dk [d];
    double  skt_qk, qkt_dkqk;
    for (int i=0; i<d; ++i){  // dkqk
      for (int j=0; j < d; ++j){
        dkqk[i] += dk[i][j]*qk[j];
      }
    }
    for (int i=0; i<d; ++i){ // skt_qk, qkt_dkqk
      skt_qk += sk[i]*qk[i];
      qkt_dkqk += qk[i]*dkqk[i];
    }
    // Compute wk
    for (int i=0; i<d; ++i){wk[i]=sqrt(qkt_dkqk)*(sk[i]/skt_qk - dkqk[i]/qkt_dkqk);}
    // Construct Ek
    for (int i=0; i<d; ++i){
      for (int j=0; j<d; ++j){
        qkt_dk[i] += qk[j]*dk[j][i];
      }
    }
    for (int i=0; i<d; ++i){
      for (int j=0; j<d; ++j){
        dk[i][j] += sk[i]*sk[j]/skt_qk - dkqk[i]*qkt_dk[j]/qkt_dkqk + wk[i]*wk[j];
      }
    }

    // Update parameters
    err = relErr(xk, xkP1);
    fileOut << k << " ";
    for (int i = 0; i < d; ++i){
      fileOut << xkP1[i] << " ";
      xk[i] = xkP1[i];
      dfk[i] = dfkP1[i];
    }
    fileOut << "\n";
    }
  fileOut.close();
}
