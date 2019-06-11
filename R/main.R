##########################################################################################
################################ Main Function to Perform NMF ############################

#' 
#' Algorithms for Nonnegative Matrix Factorization (Vanilla)
#' 
#' Function to apply various methods of NMF on the input matrix for both continuous and binary entries.
#' 
#' @useDynLib MatrixFact, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @import pracma
#' @import irlba
#' 
#' @param X Matrix; An n-by-p matrix with either continous or binary entries.
#' @param mode Integer; Define the type of the input matrix. 1 for continuous, 2 for binary.
#' @param k Clusters; Number of clusters / factors to solve for. Must not exceed the minimum of n or p.
#' @param method String; Determine which NMF algorithm to use. Choices include 'nmf', 'onmf', 'semi', 'sonmf' for
#' continuous matrices and "so_bin", "log_bin" for binary matrices. Defaults to 'nmf'.
#' @param init String; Determine the type of initialization to use. Choices include 'random', 'kmeans', and 'svd'. Defaults to 'random'.
#' @param iter Integer; Number of iterations to run. Defaults to 200. 
#' @param tol Double; Stop the algorithm when the difference between two iterations is less than this specified 
#' threshold. Defaults to 1e-5.
#' @param tau Double; Initial step size for the line search algorithm in sonmf. Defaults to 0.5.
#' @param step_bin Double; Step size for the update algorithm in binary SONMF. Defaults to 0.05.
#' @param step_log Double; Step size for both gradient descent updates for logNMF. Defaults to 0.001. This value should be tuned with caution,
#' as convergence performance is rather unstable. Recommend to leave it as default.
#' @param factor Double; The factor in which the step size in sonmf is increased/decreased during the line search. Defaults to 2.
#' @param sparse_svd Boolean; Determine whether to use the exact SVD decomposition from \code{svd()} or the fast-truncated SVD 
#' from \code{irlba()} for 'svd' initialization.
#' @param seed Integer; Set seed for reproducibility. Defaults to no seed set.
#' 
#' @details The Non-negative Matrix Factorization aims to factorize/approximate a target matrix X as the product of two lower-rank
#' matrices, F and G. 
#' 
#' @return A \code{MatrixFact} object; a list consisting of
#' \item{F}{The final F matrix}
#' \item{G}{The final G matrix}
#' \item{info}{A table with the tolerance, averaged residual, and orthogonal residual(if applicable) at each iteration}
#' \item{final_res}{A vector with the final factorized residual and orthogonal residual(if applicable) and the number
#' of iterations}
#' 
#' @references Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. In Advances in neural information processing systems (pp. 556-562).
#' DOI: \url{http://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization}.
#' @references Ding, C. H., Li, T., & Jordan, M. I. (2010). Convex and semi-nonnegative matrix factorizations. IEEE transactions on pattern analysis and machine intelligence, 32(1), 45-55.
#' DOI: \url{http://dx.doi.org/10.1109/TPAMI.2008.277}
#' @references Wen, Z. and Yin, W., "A feasible method for optimization with orthogonality constraints." Mathematical Programming 142.1-2 (2013): 397-434.
#' DOI: \url{https://doi.org/10.1007/s10107-012-0584-1}. 
#' @references Kimura, K., Tanaka, Y., & Kudo, M. (2015, February). A fast hierarchical alternating least squares algorithm for orthogonal nonnegative matrix factorization. In Asian Conference on Machine Learning (pp. 129-141).
#' DOI: \url{http://proceedings.mlr.press/v39/kimura14.pdf}
#' @references Tomé, A. M., Schachtner, R., Vigneron, V., Puntonet, C. G., & Lang, E. W. (2015). A logistic non-negative matrix factorization approach to binary data sets. Multidimensional Systems and Signal Processing, 26(1), 125-143.
#' DOI: \url{https://doi.org/10.1007/s11045-013-0240-9}
#' @references Li, J. Y., Zhu, R., Qu, A., Ye, H., & Sun, Z. (2018). Semi-Orthogonal Non-Negative Matrix Factorization. arXiv preprint arXiv:1805.02306.
#' DOI: \url{https://arxiv.org/abs/1805.02306}
#' 
#' 
#' @examples 
#' 
#' ### Create an arbitrary 100-by-100 non-negative matrix to factorize. ###
#' 
#' # Run Regular NMF with random initialization #
#' 
#' n = 100 
#' X = matrix(rnorm(n * n, 0, 1), n, n)
#' X[X < 0] = 0
#' mode = 1
#' k = 10
#' method = "nmf"
#' init = "random"
#' iter = 200
#' tol = 1e-5
#' 
#' result.1 = nmf.main(X, mode, k, method, init, iter, tol)
#' 
#' # Run SONMF with SVD initialization using the same X as above #
#' 
#' method = "sonmf"
#' init = "svd"
#' 
#' result.2 = nmf.main(X, mode, k, method, init, iter, tol)
#' 
#' # Run binary SONMF with SVD initialization for binary X.
#' 
#' n = 100
#' X = matrix(rbinom(n^2, 1, runif(n^2, 0.25, 0.75)), n, n)  
#' mode = 2
#' k = 10
#' method = "so_bin"
#' init = "svd"
#' iter = 200
#' tol = 1e-6
#' 
#' result.3 = nmf.main(X, mode, k, method, init, iter, tol)
#' 
#' @export

nmf.main = function(X, mode = 1, k, method = "nmf", init = "random", iter = 200, tol = 1e-5, tau = 0.1, 
                    step_bin = 0.05, step_log = 0.001, factor = 2, sparse_svd = TRUE, seed = 0) {
  
  # Convert input X into matrix
  X = as.matrix(X)
  
  # Check if any input parameter is invalid and prints either an error or warning message accordingly
  error.mess(X, mode, k, method, init, iter, tol, tau, step_bin, step_log, seed)
  
  # Set seed for reproducibility 
  if (seed > 0) {
    set.seed(seed)
  }
  
  # Initialize basic parameters
  n = nrow(X); p = ncol(X)
  eps = 1e-10
  
  # Initialization for F and G
  if (init == "random") {
    
    if (method != "sonmf") {
      F.init = matrix(rnorm(n * k, 0, 1), n, k)
    } else {
      F.init = randortho(n)[,1:k]
      warning("Random initialization for sonmf is inefficient. Recommend using 'svd' initilization instead.")
    }
    G.init = matrix(rnorm(p * k, 0, 1), p, k)
    
  # K-means initialization
  } else if (init == "kmeans") {
    
    fit = kmeans(t(X), k)
    G.kmeans = matrix(0, ncol(X), k)
    for(i in 1:length(fit$cluster)) {
      G.kmeans[i, fit$cluster[i]] = 1
    }
    G.init = G.kmeans + 0.2
    F.init = fit$centers
    
  # SVD-based initialization  
  } else {
    
    # Initialize with svd
    if (sparse_svd) { svd.X = irlba(as.matrix(X), nu = k, nv = k) } else { svd.X = svd(as.matrix(X), nu = k, nv = k) }
    
    # Set up F init, force to be the same sign for consistency
    if (svd.X$u[1,1] < 0) {F.init = svd.X$u[,1:k] * -1} else {F.init = svd.X$u[,1:k]}
    
    G.init = t(X) %*% F.init
  }
  
  F.init = as.matrix(F.init)
  G.init = as.matrix(G.init)
  
  # Continuous Matrix 
  if (mode == 1) {
    
    # Set initialization of G to be non-negative
    G.init[G.init < 0] = eps
    
    # NMF
    if (method == "nmf") {
      F.init[F.init < 0] = eps
      solution = NMF(X, k, F.init, G.init, tol, iter)
      
      # Semi NMF    
    } else if (method == "semi") {
      G.init = G.init + 0.2
      F.init = X %*% G.init %*% pinv(t(G.init) %*% G.init)
      solution = SemiNMF(X, k, F.init, G.init, tol, iter)
      
      # ONMF  
    } else if (method == "onmf") {
      F.init[F.init < 0] = 0
      solution = ONMF(X, k, F.init, G.init, tol, iter)
      
      # SONMF  
    } else {
      
      if (init == "kmeans") {
        stop("Use random or SVD initialization for SONMF.")
      }
      
      solution = SO_NMF(X, k, F.init, G.init, tol, iter, tau, 2)
    }
    
    
  # Matrix Factorization for binary matrices   
  } else {
    
    
    # Binary SONMF
    if (method == "so_bin") {
      G.init[G.init < 0] = eps;
      solution = SO_BIN3(X, k, F.init, G.init, tol, iter, tau, factor, step_bin)  
      
      # Tome's log_bin    
    } else if (method == "log_nmf") {
      F.init[F.init < 0] = eps;
      solution = NMF_LOG(X, k, F.init, G.init, tol, iter, step_log)
      
    } else if (method == "line_lognmf") {
      F.init[F.init < 0] = eps
      solution = lognmf(X, k, F.init, G.init, tol, iter, tau, step_log)
    }
    
  }
  
  # Force all elements beneath a threshold to 0.
  solution$F[which(abs(solution$F) <= 1e-10)] = 0
  solution$G[which(abs(solution$G) <= 1e-10)] = 0
  
  # Reorganize result and return an NMF object
  solution = organize(method, solution)
  
  return(solution)
  
}

#############################################################
# Error function 
# 
# Outputs error and warning messages
# This function examines the input parameters and output corresponding error or warning message.

error.mess = function(X, mode, k, method, init, iter, tol, tau, step_bin, step_log, seed) {
  
  ### X ###
  if(!is.matrix(X)) { stop("X must be a matrix.") } # Input X is not a matrix.
  if(!is.numeric(X)) { stop("X must be numerical.") } # Input X is not a numerical matrix.
  
  ### mode ###
  if (!(mode %in% c("1","2"))) {
    stop("Input mode must be either '1' for continuous matrix and '2' for binary matrix.")
  }
  
  ### k ###
  if(k < 0) { stop("The input k is negative. k must be a positive value.") }
  if(k > nrow(X) | k > ncol(X)) { stop("k must not be larger than max[nrow(X), ncol(X)].") }
  
  
  ### method ###

  # The input method is not one of the six implemented methods.
  if (!(method %in% c("nmf","onmf","semi","sonmf","so_bin","log_nmf"))) {
    stop ("Input method needs to be either 'nmf','onmf','semi','sonmf','so_bin or 'log_nmf'.") 
  }
  
  # Input method does not match with the input mode. For example, a method for continuous matrix  
  if (mode == 1) {
    if (method %in% c("so_bin","log_nmf")) {
      stop("Input mode is for continuous matrix but method is for binary matrix. Please input a method for continuous matrix.")
    }
  } else if (mode == 2) {
    if (method %in% c("nmf","onmf","semi","sonmf")) {
      stop("Input mode is for binary matrix but method is for continuous matrix. Please input a method for binary matrix.")
    }
  }
  
  ### init ###
  if (!(init %in% c("random","kmeans","svd"))) {
    stop("Input initialization needs to be either 'random','kmeans', or 'svd'.")
  }
  
  ### iter ###
  if (iter < 0) {
    stop("Number of iterations set to run must be a positive number.")
  }
  
  ### tol ###
  if (tol <= 0) {
    stop("Convergence threshold needs to be a positive number.")
  } else if (tol > 0.1) {
    warning("Convergence threshold is too large. Algorithm may not have converged.
            It is recommended to set this positive value to at most 0.0001.")
  }
  
  ### tau ###
  if (tau < 0) {
    stop("Step size for sonmf needs to be positive value.")
  } else if (tau > 2) {
    warning("Step size for sonmf is too large. Please choose a smaller step size.")
  }
  
  ### step_bin ###
  if (step_bin < 0) {
    stop("Step size for the update of G (so_bin) needs to be a positive value.")
  }
  if (method == 'so_bin') {
    if (step_bin > 0.5) {
      warning('The input step size is quite large. Please choose a smaller step size for more stable performance.')
    } else if (step_bin > 1) {
      stop('The input step size is too large. Convergence issue may arise. Please choose a smaller step size.')
    }
    
  } else if (method == "log_nmf") {
    if (step_log > 0.001) {
      warning("The input step size quite large. Please choose a smaller step size for more stable performance.")
    } else if(step_log > 0.01) {
      stop('The input step size is too large. Convergence issue may arise. Please choose a smaller step size.')
    }
  }
}

###########################################################
# Reorganize result from NMF and returns an NMF object 

organize = function(method, nmf.res) {
  
  
  # For methods with orthogonal residuals
  if (method %in% c("sonmf", "onmf", "so_bin")) {
    
    iteration_info = nmf.res$info[1:nmf.res$iter, 1:3]
    colnames(iteration_info) = c("Tolerance", "Averaged Residual", "Orthogonal Residual")
    info = c(nmf.res$final_res, nmf.res$final_orth, floor(nmf.res$iter))
    names(info) = c("Averaged Residual", "Orthogonal Residual", "Iterations")
    
  # For methods without orthogonal residuals  
  } else {
    
    iteration_info = nmf.res$info[1:nmf.res$iter, 1:2]
    colnames(iteration_info) = c("Tolerance", "Averaged Residual")
    info = c(nmf.res$final_res, floor(nmf.res$iter))
    names(info) = c("Averaged Residual", "Iterations")
    
  }
  
  res = list(F = nmf.res$F, G = nmf.res$G, info = iteration_info, final_result = info)
  class(res) = c("F", "G", "info", "final_result")
  
  return(res)
}  

############################################################
#' Function Specific for Binary Simulation
#' 
#' This function is for simulation purposes only.
#' 
#' This function is the same as nmf.main(), with the additional parameter of an input
#' true probability matrix for simulation purpose. This is not applicable in actual
#' scenarios, since there is no way of knowing the true underlying probability matrix
#' beforehand.
#' 
#' @useDynLib MatrixFact, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @import pracma
#' @import irlba
#' 
#' @export

bin.test = function(X, k, F.init, G.init, method, prob, iter = 500, tol = 1e-5, tau = 0.5, 
                    step_bin = 0.05, step_log = 0.001, factor = 2) {
  
  if (method == "so_bin") {
    res = sobin_test(X, k, F.init, G.init, prob, tol, iter, tau, factor, step_bin)
  } else if (method == "log_nmf") {
    res = log_test(X, k, F.init, G.init, prob, tol, iter, step_log)
  } else {
    stop("Input method must be either 'so_bin' or 'log_nmf'.")
  }
  return (res)
}
  
