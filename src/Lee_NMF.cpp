#include <iostream>
#include <RcppArmadillo.h>
#include <vector>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace std;
using namespace arma;

// [[Rcpp::export]]
List NMF(arma::mat X, const int k, arma::mat F_init, arma::mat G_init, const double error, 
            const int iter) 
{
  //___________________ Initialization ___________________
  arma::mat F_p = F_init;
  arma::mat G_p = G_init;
  arma::vec all_res(iter), all_tol(iter);
  int count = 1, i_e = 0;
  double eps = 1e-10;
  arma::mat F, G, FG, info;
  
  //___________________ Begin Algorithmn ___________________
  
  FG = F_p * G_p.t();
  
  all_res.fill(0);  all_tol.fill(0);
  all_tol(i_e) = 1;
  all_res(i_e) = accu(square(X - FG)) / X.size();
  
  
  while (all_tol(i_e) > error && count < iter) {
    
    i_e += 1;
    
    // Update G using multiplicative updates
    G = ((X.t() * F_p) / (G_p * F_p.t() * F_p + eps)) % G_p;
    G_p = G;
    
    // Update F using multiplicative updates
    F = F_p % ((X * G) / (F_p * G.t() * G + eps));
    F_p = F;
    
    // Update Error
    FG = F * G.t();
    
    all_res(i_e) = accu(square(X - FG)) / X.size();
    all_tol(i_e) = all_res(i_e - 1) - all_res(i_e);
    count += 1;
  
  }
  
  // Enforce all values less than a certain threshold to be 0.
  F(find(F == eps)).fill(0);
  G(find(G == eps)).fill(0);
  
    
  info = join_rows(all_tol, all_res);
  
  return List::create(_["F"] = F, _["G"] = G, _["info"] = info,_["iter"] = count, _["final_tol"] = all_tol[i_e], 
                        _["final_res"] = all_res[i_e]);
}




