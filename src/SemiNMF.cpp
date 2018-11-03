#include <iostream>
#include <RcppArmadillo.h>
#include <vector>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace std;
using namespace arma;

// [[Rcpp::export]]
List SemiNMF(arma::mat X, const int k, arma::mat F_init, arma::mat G_init, const double error, const int iter) 
{
  mat F, F_p, G, G_p, FG, t1, t2, b1, b2, ratio, info, X_F, F_F;
  vec all_res(iter), all_tol(iter);
  int count = 1, i_e = 0;
  
  // Initialize F & G
  G_p = G_init;
  F_p = F_init;
  FG = F_p * G_p.t();
  
  all_tol(i_e) = 1;
  all_res(i_e) = mean(mean(square((X - FG))));
  
  // Update F & G through iteration until convergence
  while (abs(all_tol(i_e)) > error && count != iter) {
    
    i_e += 1;
    
    // Perform matrix-wise update Scheme
    
    X_F = X.t()*F_p;
    F_F = F_p.t()*F_p;
    t1 = (abs(X_F) + X_F)/2;
    t2 = G_p*((abs(F_F) - F_F)/2);
    b1 = (abs(X_F) - X_F)/2;
    b2 = G_p*((abs(F_F) + F_F)/2);
    ratio = (t1 + t2) / (b1 + b2);
    G = G_p % sqrt(ratio);  
    F = X*G*inv(G.t()*G);
    
    // Update information
    FG = F * G.t();
    all_res(i_e) = mean(mean(square((X - FG))));
    all_tol(i_e) = all_res(i_e - 1) - all_res(i_e);
    G_p = G;
    F_p = F;
    
    count += 1;
  }
  
  info = join_rows(all_tol, all_res);
  
  return List::create(_["F"]=F, _["G"]=G, _["info"] = info, 
                      _["final_tol"] = all_tol[i_e], _["final_res"] = all_res[i_e], _["iter"] = count);
}

