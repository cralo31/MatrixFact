#include <iostream>
#include <RcppArmadillo.h>
#include <vector>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace std;
using namespace arma;

// [[Rcpp::export]]
List ONMF(arma::mat X, const int k, arma::mat F_init, arma::mat G_init, const double error, 
          const int iter) 
{
  //___________________ Initialization ___________________
  arma::mat F = F_init;
  arma::mat G = G_init;
  arma::mat pF = F_init;
  arma::mat pG = G_init;
  arma::vec F_j, h, tempF, tempG, all_res(iter), all_tol(iter), all_orth(iter);
  int count = 1, i_e = 0;
  double eps = 1e-10;
  arma::mat A, B, C, D, pFG, FG, F_orig, F_1, F_2, Z, Q, info;
  arma::mat I = eye(k, k);
  
  //___________________ Begin Algorithmn ___________________
  
  arma::vec one = ones<vec>(k);
  arma::mat U = F * one;
  
  FG = F * G.t();
  
  all_res.fill(0); all_orth.fill(0); all_tol.fill(0);
  all_tol(i_e) = 1;
  all_res(i_e) = accu(square(X - FG)) / X.size();
  all_orth(i_e) = accu((F.t() * F - I) % (F.t() * F - I));
  
  while (abs(all_tol(i_e)) > error && count < iter) {
    
    i_e += 1;
    
    // Update F in column-wise fashion
    A = X * G;
    B = G.t() * G;
    for (int j = 0; j < k; ++j) {
      F_j = U - F.col(j);
      h = A.col(j) - F * B.col(j) + F.col(j) * B(j,j);
      tempF = F.col(j);
      tempF = h - F_j * (((F_j.t() * h) / (F_j.t() * F_j)));
      tempF(find(tempF < 0)).fill(eps);
      F.col(j) = tempF;
      F.col(j) = F.col(j) * inv(sqrt(F.col(j).t() * F.col(j)));  // Normalize columns of F as you update
      U = F_j + F.col(j);
    }
    
    // Update G in column-wise fashion
    C = X.t() * F;
    D = F.t() * F;
    for (int j = 0; j < k; ++j) {
      tempG = G.col(j);
      tempG = C.col(j) - G * D.col(j) + G.col(j) * D(j,j);
      tempG(find(tempG < 0)).fill(eps);
      G.col(j) = tempG;
    }
    
    // Update Error
    FG = F * G.t();
    
    all_res(i_e) = accu(square(X - FG)) / X.size();
    all_tol(i_e) = all_res(i_e - 1) - all_res(i_e);
    all_orth(i_e) = accu((F.t() * F - I) % (F.t() * F - I));
    
    count += 1;
  }
  
  info = join_rows(all_tol, all_res); info = join_rows(info, all_orth);
  
  return List::create(_["F"] = F, _["G"] = G, _["info"] = info,
                      _["iter"] = count, _["final_tol"] = all_tol[i_e], _["final_res"] = all_res[i_e], 
                        _["final_orth"] = all_orth[i_e]);
}




