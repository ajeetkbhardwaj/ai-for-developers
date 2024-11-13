# Eigen Value and Vecotr

Definition :
Consider a square matrix $A=[a_{ij}]_{n{\times}n}$ and a nonzero vector $v$ of length $n$ Then there exit a scalar $\lambda \in R$ such that $Av = \lambda v$ 
.
Where $v$ is called eigenvector corresponding to eigenvalue $\lambda$ of matrix $A$.

Remark : 
1. Linear transformation $T : R^n \to R^n$ is equivalent to the square matrix $A$ of order $n \times n$. thus given a basis set of the vector space can be defined as set of eigen vectors of matrix $A$ for linear transformation $T$.

2. Eigenvectors and eigenvalues exits in a wide range of applications  like stability and vibration analysis of dynamical systems, atomic orbits, facial recognition and matrix diagonalization.

## Faddeev-LeVerrier Algorithm

Faddeev-LeVerrier algorithm is a recursive method to calculate the coefficients of the characteristic polynomial 
$p_A(\lambda)=\det (\lambda I_n - A)$
 of a square matrix, A. solving characteristic polynomial gives eigen values of matrix A as a roots of it and matrix polynomial of matrix A  vanishes i.e p(A) = 0 by Cayley-Hamilton Theorem. Faddeev-Le Verrier algorithm works directly with coefficients of matrix $A$.

Problem : Given a 