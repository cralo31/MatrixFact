#include <iostream>
#include <RcppArmadillo.h>
#include <vector>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace std;
using namespace arma;

// [[Rcpp::export]]
List SO_NMF(arma::mat X, const int k, arma::mat F_init, arma::mat G_init, const double error, 
           const int iter, double tau, const int factor) 
{
  //___________________ Initialization ___________________
  arma::mat F = F_init, G = G_init;
  arma::vec tempG, all_res(iter), all_tol(iter), all_orth(iter), all_tau(iter), all_counter(iter), all_res2(iter);
  double prev_mse, temp_mse;
  int count = 1, i_e = 0, counter;
  double eps = 0, tol = 1;
  arma::mat C, D, FG, F_new, R, FG_temp, U, V, info, V_U, V_F;
  arma::mat I = eye(k, k);
  arma::mat I_2 = eye(2*k, 2*k);
  arma::vec one = ones<vec>(k);
  
  //___________________ Begin Algorithmn ___________________
  
  FG = F * G.t();
  prev_mse = mean(mean(square((X - FG))));
  
  all_res.fill(0); all_orth.fill(0); all_tol.fill(0);
  all_tol(i_e) = 1;
  all_res(i_e) = prev_mse;
  all_orth(i_e) = accu((F.t() * F - I) % (F.t() * F - I));
  all_tau(i_e) = 1/2;
  all_counter(i_e) = 1;
  all_res2(i_e) = accu(square(X - FG)) / accu(square(X));
  
  
  while (abs(all_tol(i_e)) > error && count < iter) {
    
    i_e += 1;
    
    // Update G in column-wise fashion
    G = X.t() * F;
    G(find(G < 0)).fill(eps);
    
    // Solve for F
    // Set the residual in the direction of the gradient
    R = X * G * (-2) + F * G.t() * G * 2;
    
    // Set up U and V where Z = U(V^T)
    U = join_rows(R, F);
    V = join_rows(F, -R);
    
    // Find orthogonal Q via Crank-Nicolson, then use Q to project onto original F_1
    // to create corrected columns
    counter = 0;
    V_U = V.t() * U;
    V_F = V.t() * F;
    while (tol < 0 || counter == 0) {
      F_new = F - U * inv(I_2 + V_U * tau/2) * V_F * tau;
      FG_temp = F_new * G.t();
      temp_mse = accu(square(X - FG_temp)) / X.size();
      tol = prev_mse - temp_mse;
      counter += 1;
      
      if (tol > 0) {
        tau = tau * factor;
        F = F_new;
        all_tau(i_e) = tau; all_counter(i_e) = counter; all_res(i_e) = temp_mse;
        prev_mse = temp_mse;
      } else if (tol <= 0 && counter <= 50) {
        tau = tau / factor;
      } else {
        break;
      }
    }
    
    // Update Error
    FG = F * G.t();
    all_tol(i_e) = all_res(i_e - 1) - all_res(i_e);
    all_orth(i_e) = accu((F.t() * F - I) % (F.t() * F - I));
    
    count += 1;
    
  }
  
  // Merge info into one matrix
  info = join_rows(all_tol, all_res); info = join_rows(info, all_orth); 
  info = join_rows(info, all_tau); info = join_rows(info, all_counter);
  
  return List::create(_["F"] = F, _["G"] = G, _["info"] = info, _["iter"] = count, _["final_tol"] = all_tol[i_e], 
                        _["final_res"] = all_res[i_e], _["final_orth"] = all_orth[i_e]);
}




