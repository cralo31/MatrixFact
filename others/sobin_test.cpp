#include <iostream>
#include <RcppArmadillo.h>
#include <vector>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace std;
using namespace arma;

// [[Rcpp::export]]
List sobin_test(arma::mat X, const int k, arma::mat F_init, arma::mat G_init, arma::mat prob_t,
             const double error, const int iter, double tau, const int factor, double step) 
{
  //___________________ Initialization ___________________
  arma::mat F = F_init;
  arma::mat G_p = G_init;
  arma::vec tempG, all_cost(iter), all_tol(iter), all_orth(iter), all_tau(iter), all_counter(iter), all_prob(iter);
  double prev_cost, temp_cost;
  int count = 1, i_e = 0, counter;
  double tol = 1;
  arma::mat C, D, FG, F_new, R, FG_temp, U, V, nume, deno, G, info;
  arma::mat I = eye(k, k);
  arma::mat I_2 = eye(2*k, 2*k);
  arma::mat one = ones<mat>(size(X));

  
  //___________________ Begin Algorithmn ___________________
  
  FG = F * G_p.t();
  
  // Initiate vectors to track errors
  all_cost.fill(0); all_orth.fill(0); all_tol.fill(0);
  all_tol(i_e) = 1;
  all_cost(i_e) = mean(mean(log(one + exp(FG)) - X % FG));
  all_orth(i_e) = accu((F.t() * F - I) % (F.t() * F - I));
  all_prob(i_e) = sqrt(accu(square(one / (one + exp(-(FG))) - prob_t)));
  all_tau(i_e) = tau;
  all_counter(i_e) = 1;
  prev_cost = all_cost(0);
  
  while (all_tol(i_e) > error && count < iter) {
    
    i_e += 1;
    
    // Update G using Newton-Rhapson
    FG = F * G_p.t();
    nume = (one / (one + exp(-(FG))) - X).t() * F;
    deno = (exp(FG) / square(one + exp(FG))).t() * square(F);
    G = G_p - step * (nume / deno);
    G(find(G < 0)).zeros();
    G_p = G;
    
    // Solve for F
    // Set gradient of F
    R = (one / (one + exp(-(F * G.t()))) - X) * G;
    
    // Set up U and V where Z = U(V^T)
    U = join_rows(R, F);
    V = join_rows(F, -R);
    
    // Find orthogonal Q via Crank-Nicolson, then use Q to project onto original F_1
    // to create corrected columns
    counter = 0;
    while (tol < 0 || counter == 0) {
      F_new = F - tau * U * inv(I_2 + tau/2 * V.t() * U) * V.t() * F;
      FG_temp = F_new * G.t();
      temp_cost = mean(mean(log(one + exp(FG_temp)) - X % FG_temp));
      tol = prev_cost - temp_cost;
      counter += 1;
      
      if (tol > 0) {
        all_cost(i_e) = temp_cost;all_tol(i_e) = tol; all_tau(i_e) = tau; all_counter(i_e) = counter;
        tau = tau * factor;
        F = F_new;
        prev_cost = temp_cost;
      } else if (tol <= 0 && counter <= 50) {
        tau = tau / factor;
      } else {
        break;
      }
    }
    
    // Update probability and projection error
    FG = F * G.t();
    all_prob(i_e) = sqrt(accu(square(one / (one + exp(-(FG))) - prob_t)));
    all_orth(i_e) = accu((F.t() * F - I) % (F.t() * F - I));
    count += 1;
    
  }
  
  info = join_rows(all_tol, all_cost);
  info = join_rows(info, all_prob);
  info = join_rows(info, all_orth); 
  
  return List::create(_["F"] = F, _["G"] = G, _["info"] = info, _["iter"] = count,
                      _["final_res"] = all_cost[i_e], _["final_ortho"] = all_orth[i_e]);
}


