// Test BFGS method on problems (serial)
#include <iostream>
#include <cmath>
#include <fstream>
#include <random>

using namespace std;

// Define function operator
double f (double x[]){  // Eggholder function
  return -(x[1]+47.0)*sin(sqrt(abs(x[0]/2.0 + x[1] + 47.0)))
          - x[0]*sin(sqrt(abs(x[0] - x[1] - 47.0)));
}

// double f (double x[]){  // 1D test function
//   return (x[0]-5)*(x[0]-5);
// }

int d = 2; // Problem dimensions
double tol = pow(10, -150);  // Converge when relative dist. between terms < tol
double tolSq = tol*tol;  // Used below

double  gradf (double xk[], double dx, int i){
  double df;
  double xtemp [d] = {};
  for (int j=0; j<d; ++j){
    xtemp[j] = xk[j];
    if (j == i){xtemp[j] += dx;}
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

double normx (double x[]){
  double norm = 0;
  for (int i=0; i<d; ++i){
    norm += pow(x[i], 2);
  }
  return norm;
}

int main (){
  // Initialise random number engine
  default_random_engine generator;
  uniform_real_distribution<double> rand(-512.0, 512.0);


  // Initialise problem parameters
  int nP = 10;  // Number of particles to use
  double xOpt [d] = {};
  double fOpt = pow(10, 16);
  ofstream fileOut;
  double dx = 0.1; // Distance used to approx gradient
  int alphaIts = 50; // No. steps to estimate alpha per iteration
  fileOut.open("iterations.csv");
  for (int n=0; n < nP; ++n){
    double  xk [d] = {}, xkP1 [d] = {}, dfk [d] = {}, dfkP1 [d] ={}, sk [d] ={},
      qk [d] ={}, dkgf [d] = {}, xtemp0 [d] ={}, xtemp1 [d] = {}, wk [d] = {};  // Vectors
    double dk [d][d] = {};  // Matrices
    double err = 2*tolSq;  // Error between terms


    // Starting point
    for (int i=0; i < d; ++i){
      xk[i] = rand(generator);
    }

    double alphas [alphaIts];  // Used to find optimal movement parameter
    for (int i=0; i<alphaIts; ++i){
      double alphat = alphaIts;
      alphas[i] = (i+1)*0.2/alphat;
    }
    int k=0; // No. its

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
        dkgf[i] = 0;
        for (int j = 0; j < d; ++j){
          dkgf[i] += dk[i][j]*dfk[j];
        }
      }

      // Find optimal alpha
      double fopt = 2*f(xk);
      double alphaOpt = 0.1;
      for (int i = 0; i < alphaIts; ++i){
        for (int j = 0; j < d; ++j){
          xtemp0[j] = xk[j] - alphas[i]*dkgf[j];
        }
        if (f(xtemp0) < fopt){
          alphaOpt = alphas[i];
          fopt = f(xtemp0);
        }
      }
      // Update x
      for (int i = 0; i < d;  ++i){
        xkP1[i] = xk[i] - alphaOpt*dkgf[i];
      }

      // Update D
      for (int i=0; i<d; ++i){
        dfkP1[i] = gradf(xkP1, dx, i);  // Gradient f(x_k+1)
        cout << "\ndfpi: " << xkP1[i] << "\n";
      }

      for (int i = 0; i < d; ++i){  // Update sk, qk
        sk[i] = xkP1[i] - xk[i];
        qk[i] = dfkP1[i] - dfk[i];
        cout <<"\n"<< i << " ski: " << sk[i] << " qki " << qk[i] << "\n";
      }
      // Compute parameters for update of D
      double  dkqk [d] = {}, qkt_dk [d] = {};
      double  skt_qk = 0, qkt_dkqk = 0;
      for (int i=0; i<d; ++i){  // dkqk
        dkqk[i] =0;
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
        qkt_dk[i] = 0;
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
      err = normx(dfkP1);
      fileOut << k << " ";
      for (int i = 0; i < d; ++i){
        fileOut << xkP1[i] << " ";
        xk[i] = xkP1[i];
        dfk[i] = dfkP1[i];
      }
      fileOut << "\n";
      }
    if (f(xk) < fOpt){
      for (int i = 0; i<d; ++i){xOpt[i] = xk[i];}
      fOpt = f(xOpt);
    }
  }
  fileOut.close();
}
