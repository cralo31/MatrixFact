#include <iostream>
#include <RcppArmadillo.h>
#include <vector>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace std;
using namespace arma;

// [[Rcpp::export]]
List log_test(arma::mat X, const int k, arma::mat F_init, arma::mat G_init, arma::mat prob_t,
             const double error, const int iter, double step) 
{
  //___________________ Initialization ___________________
  arma::mat F_p = F_init;
  arma::mat G_p = G_init;
  arma::vec tempG, all_cost(iter), all_tol(iter), all_prob(iter);
  int count = 1, i_e = 0;
  double final_cost;
  arma::mat FG, F, G, info, proj_F, proj_G;
  arma::mat I = eye(k, k);
  arma::mat I_2 = eye(2*k, 2*k);
  arma::mat one = ones<mat>(size(X));
  
  //___________________ Begin Algorithmn ___________________
  
  FG = F_p * G_p.t();
  
  // Initiate vectors to track errors
  all_cost.fill(0); all_tol.fill(0);
  all_tol(i_e) = 1;
  all_cost(i_e) = mean(mean(log(one + exp(FG)) - X % FG));
  all_prob(i_e) = sqrt(accu(square(one / (one + exp(-(FG))) - prob_t)));
  
  while (all_tol(i_e) > error && count < iter) {
    
    i_e += 1;
    
    // Update F
    F = F_p + step * ((2 * X - one) / (one + exp((2*X - one) % (F_p * G_p.t())))) * G_p;
    F(find(F < 0)).zeros();
    
    
    // Update G 
    G = G_p + step * ((2 * X - one) / (one + exp((2*X - one) % (F * G_p.t())))).t() * F;
    
    F_p = F; G_p = G;
    
    // Update probability and projection error
    FG = F_p * G_p.t();
    
    all_cost(i_e) = mean(mean(log(one + exp(FG)) - X % FG));
    all_prob(i_e) = sqrt(accu(square(one / (one + exp(-(FG))) - prob_t)));
    all_tol(i_e) = all_cost(i_e - 1) - all_cost(i_e);
    count += 1;
  }
  
  FG = F * G.t();
  final_cost = mean(mean(log(one + exp(FG)) - X % FG));
  
  info = join_rows(all_tol, all_cost); info = join_rows(info, all_prob);
  
  return List::create(_["F"] = F, _["G"] = G, _["info"] = info, _["iter"] = count, _["final_res"] = final_cost);
}
