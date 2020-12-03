# MatrixFact
R Package for SONMF and other NMF methods

# Description
This R package is developed for the Semi-orthogonal Non-negative Matrix Factorization method proposed in the "Topic Modeling on Triage Notes with Semi-orthognal Non-negative Matrix Factorization" manuscript. 

# Installation
Execute the following code in R to install the package.
```{r}
# Need the "devtools" library to install packages from Github
install.packages("devtools")

# Install the "MatrixFact" library
devtools::install_github('cralo31/MatrixFact')
```

# Example 
It is recommended to run the following examples to have an idea of how SONMF works. 

```{r}
n = 100 
X = matrix(rnorm(n^2, 0, 1), n, n)   # Construct a 100-by-100 matrix with random elements
X[X < 0] = 0                         # Truncate all negative values to create a non-negative matix
mode = 1                             # Set mode to 1 for matrix with continuous entries
k = 5                              # Set target factorization rank to be k
iter = 200                           # Set number of iterations to be 200 
tol = 1e-5                          # Set convergence threshold to 1e-5

# Run SONMF with svd initialization
sonmf.X = nmf.main(X, mode, k, "sonmf", "svd", iter, tol); sonmf.X

# Run NMF with random initialization
nmf.X = nmf.main(X, mode, k, "nmf", "random", iter, tol); nmf.X

# Run ONMF with random initialization
onmf.X = nmf.main(X, mode, k, "onmf", "random", iter, tol); onmf.X

# Run Semi-NMF with kmeans initialization
semi.X = nmf.main(X, mode, k, "semi", "kmeans", iter, tol); semi.X

## Binary Case 

X.bin = matrix(rbinom(n^2, 1, runif(n^2, 0.25, 0.75)), n, n)  # Construct a 100-by-100
                                                              # matrix with random binary elements
mode.2 = 2

# Run Binary SONMF with svd initialization
sobin.X = nmf.main(X.bin, mode.2, k, "so_bin", "svd", iter, tol); sobin.X

# Run logNMF with random initialization
lognmf.X = nmf.main(X.bin, mode.2, k, "log_nmf", "random", iter, tol); lognmf.X
```
