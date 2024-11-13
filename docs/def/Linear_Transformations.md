---
marp: true
theme: gaia
class: invert
size: 16:9
math: mathjx
footer: Ajeet Kumar Bhardwaj | ajeetskbp9843@gmail.com | [Linkedin](https:\\www.linkedin.com\in\ajeetkumar09)
---
# Linear Transformations

Given two linear space V and W over a field F, a linear map $T:V\to W$ that preserve addition and scalar multiplication such that

$$
\begin{equation}
T(\mathbf u + \mathbf v)=T(\mathbf u)+T(\mathbf v) 
\\ 
\quad T(a \mathbf v)=aT(\mathbf v)
\end{equation}
$$

or we can write in general via linear combination as

$$
\\ \;
\begin{equation}
T(a \mathbf u + b \mathbf v)= T(a \mathbf u) + T(b \mathbf v) = aT(\mathbf u) + bT(\mathbf v) 
\end{equation}
$$

for all $u, v \in V$ and scaler $a, b \in F$

remark-1 : If $V = W$ are the same linear space then linear map $T : V → V$ is also known as a linear operator on $V$.

---

remark-2 : A bijective linear map between two linear spaces  is an isomorphism because it preserves linear structure and two isomorphic linear spaces are same algebraically means we can't make any distiction between these two spaces using linear space properties.

remark-3 : How to check wheather a linear map is isomorphic or not. If it is non-isomorphic then we find its range space(set of elements which have non-zero images) and null space(set of elements which have zero images also called kernel of T).
