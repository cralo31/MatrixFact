#include <iostream>
#include <RcppArmadillo.h>
#include <vector>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace std;
using namespace arma;

// [[Rcpp::export]]
List lognmf(arma::mat X, const int k, arma::mat F_init, arma::mat G_init,
             const double error, const int iter, double tau, double step) 
{
  //___________________ Initialization ___________________
  arma::mat F_p = F_init;
  arma::mat G_p = G_init;
  arma::vec tempG, all_cost(iter), all_tol(iter), all_tau(iter);
  int count = 1, i_e = 0, counter;
  double tol = 1, prev_cost, temp_cost, final_cost;
  arma::mat FG, F, G, info, F_temp, FG_temp;
  arma::mat I = eye(k, k);
  arma::mat I_2 = eye(2*k, 2*k);
  arma::mat one = ones<mat>(size(X));
  
  //___________________ Begin Algorithmn ___________________
  
  FG = F_p * G_p.t();
  
  // Initiate vectors to track errors
  all_cost.fill(0); all_tol.fill(0);
  all_tol(i_e) = tol;
  all_cost(i_e) = mean(mean(log(one + exp(FG)) - X % FG));
  all_tau(i_e) = tau;
  prev_cost = all_tol(0);
  // all_prob(i_e) = sqrt(accu(square(one / (one + exp(-(FG))) - prob_t)));

  while (all_tol(i_e) > error && count < iter) {
    
    i_e += 1;
    
    // Update G 
    G = G_p + step * ((2 * X - one) / (one + exp((2*X - one) % (F_p * G_p.t())))).t() * F_p;
    G_p = G;
    
    // Update F
    counter = 0;
    while (tol < 0 || counter == 0) {
      F_temp = F_p + tau * ((2 * X - one) / (one + exp((2*X - one) % (F_p * G.t())))) * G;
      F_temp(find(F_temp < 0)).zeros();
      FG_temp = F_temp * G.t();
      temp_cost = mean(mean(log(one + exp(FG_temp)) - X % FG_temp));
      tol = prev_cost - temp_cost;
      counter += 1;
      
      if (tol <= 0 && counter <= 50) {
        tau = tau / 2;
      } else if (tol > 0){
        F = F_temp;
        tau = tau * 2;
        prev_cost = temp_cost;
        all_cost(i_e) = temp_cost; all_tol(i_e) = tol; all_tau(i_e) = tau;
      } else {
        cout << "Cannot find suitable step size" << endl;
        F = F_p;
        break;
      }
      
    }
    
    F_p = F;
    
    FG = F * G.t();
    final_cost = mean(mean(log(one + exp(FG)) - X % FG));
  
    
    // Update probability and projection error
    // all_prob(i_e) = sqrt(accu(square(one / (one + exp(-(FG))) - prob_t)));
    count += 1;
    
  }
  
  info = join_rows(all_tol, all_cost); info = join_rows(info, all_tau);
  
  
  //colnames(info) = ("Tol", "Cost", "Ortho", "Prob", "Tau", "Counter");
  
  return List::create(_["F"] = F, _["G"] = G, _["info"] = info, _["iter"] = count,
                      _["final_res"] = final_cost);
}


