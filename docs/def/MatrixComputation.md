# Matrix Computation

Understanding Errors in Numerical Computation

Errors in numerical computation arise from various sources, such as modeling inaccuracies, measurement noise, manufacturing imperfections, and limitations in computational precision. These errors are compounded when data must be represented in finite precision formats like floating-point arithmetic, which introduces rounding errors. Additional errors may emerge from truncating infinite mathematical processes into finite iterative computations. These challenges are particularly relevant in matrix computations, where the sensitivity of algorithms to errors depends on factors like vector and matrix norms, floating-point representation standards, condition numbers, and numerical stability. Norms quantify the size of vectors and matrices, providing a way to express error bounds, while condition numbers measure how much small changes in input can affect computational results. Numerical stability ensures that rounding errors introduced during computation do not grow excessively, safeguarding the reliability of results even under finite precision constraints.

Ensuring Reliable Computations Through Standards and Methods
Floating-point arithmetic, as standardized by IEEE (e.g., IEEE 754 and IEEE 854), defines how real numbers are approximated and manipulated in computer systems. These standards provide a framework for understanding how rounding errors arise and propagate during computations. Tools like norms, condition numbers, and perturbation analysis are used to analyze and mitigate these errors. For example, a matrix's condition number determines whether the problem it represents is well-posed or ill-posed, influencing the choice of algorithm. Stable algorithms, which minimize the amplification of rounding errors, are critical in ensuring accurate results even for large or sensitive problems. Iterative methods like the power method, inverse iteration, and orthogonal iteration play significant roles in computing eigenvalues and eigenvectors, balancing accuracy, stability, and computational efficiency. When coupled with robust theoretical frameworks, such as those outlined in seminal works by Wilkinson, Higham, and others, these tools enable effective handling of errors in numerical computations, ensuring precision in both real and complex vector spaces.


### Conditioning and Condition Numbers
Errors are inevitable in real-world data due to measurement imprecision, equipment degradation, manufacturing tolerances, and limitations of floating-point arithmetic. These errors propagate through calculations, influencing the precision of computational results. The field of conditioning studies how errors in input data affect the results of computations.

### Conditioning and Condition Numbers

Errors are inevitable in real-world data due to measurement imprecision, equipment degradation, manufacturing tolerances, and limitations of floating-point arithmetic. These errors propagate through calculations, influencing the precision of computational results. The field of *conditioning* studies how errors in input data affect the results of computations. Below is a summary of the key concepts:


1. **Computational Problem:**  
   A computational problem involves evaluating a function $P: \mathbb{R}^n \to \mathbb{R}^m$ at a given data point $z \in \mathbb{R}^n$. In practice, $z$ is often approximated due to errors, represented as $\hat{z} \in \mathbb{R}^n$.

2. **Error Measures:**  
   - **Absolute Error:**  
     $
     \text{Absolute error} = \|z - \hat{z}\|.
     $
   - **Relative Error:**  
     $
     \text{Relative error} = \frac{\|z - \hat{z}\|}{\|z\|}, \quad z \neq 0.
     $
     If $z = 0$, the relative error is undefined.

3. **Conditioning of Data:**  
   - **Well-Conditioned Data:**  
     Small relative perturbations in $ z $ lead to small relative perturbations in $ P(z) $.
   - **Ill-Conditioned Data:**  
     Even small relative perturbations in $ z $ may cause large relative perturbations in $ P(z) $.  
   The distinction depends on the context and precision required for the task.

---

#### Condition Number:
To quantify the sensitivity of a problem to input perturbations, the *condition number* is defined.

1. **Relative Condition Number:**  
   For $z \in \mathbb{R}^n $ and $ P(z) \neq 0 $, the relative condition number is given by:
   $$
   \text{cond}_P(z) = \lim_{\epsilon \to 0} \sup \frac{\|P(z + \delta z) - P(z)\| / \|P(z)\|}{\|\delta z\| / \|z\|}, \quad \|\delta z\| \leq \epsilon.
   $$
   This measures the largest possible relative change in $ P(z) $ resulting from small relative changes in $ z $.

2. **Extension for $ z = 0 $ or Isolated Roots:**  
   When $ z = 0 $ or $ z $ is an isolated root of $ P(z) $, the condition number can be generalized as:
   $$
   \text{cond}_P(z) = \limsup_{x \to z} \text{cond}_P(x).
   $$

3. **Key Notes:**  
   - The condition number depends on the function $ P $, the input data $ z $, and the norm used in the computation.
   - It characterizes the sensitivity of $ z $, not $ P(z) $, and not the algorithm used to compute $ P(z) $.

---

#### Importance of Conditioning:
- Well-conditioned data ensures reliable computational results with limited precision.
- Ill-conditioned data may lead to significant errors or instability in computations, requiring additional measures such as robust algorithms or improved precision in inputs.

### Key Facts About Conditioning and Condition Numbers

#### 1. Impact of Finite Precision and Rounding Errors:
- **Ubiquity of Rounding Errors:**  
  In any finite precision computation, the best achievable result is $ P(z + \delta z) $, where $ \|\delta z\| \leq \epsilon \|z\| $, and $ \epsilon $ is a small multiple of the machine's floating-point unit round-off.  
  (Refer to Section 37.6 for more details.)

---

#### 2. Relative Condition Number and Error Bounds:
- **Asymptotic Relative Error Bound:**  
  The relative condition number provides an asymptotic bound for the relative error:  
  $$
  \frac{\|P(z + \delta z) - P(z)\|}{\|P(z)\|} \leq \text{cond}_P(z) \frac{\|\delta z\|}{\|z\|} + o\left(\frac{\|\delta z\|}{\|z\|}\right),
  $$  
  as $ \|\delta z\| \to 0 $.  
- **Significant Digits and Condition Number:**  
  If the condition number satisfies $ \text{cond}_P(z) \approx 10^s $, then roughly $ s $ significant digits are lost in the computed result $ P(z) $. For example, if the input data $ z $ has $ p $ correct digits, the result $ P(z) $ retains approximately $ p - s $ correct digits.

---

#### 3. Condition Number via Fréchet Derivatives:
- **Definition via Derivatives:**  
  For $ P $ having a Fréchet derivative $ D(z) $ at $ z \in \mathbb{F}^n $, the relative condition number is:  
  $$
  \text{cond}_P(z) = \frac{\|D(z)\| \cdot \|z\|}{\|P(z)\|}.
  $$
- **Condition Number for Scalar Functions:**  
  If $ f(x) $ is a smooth real function of a real variable $ x $, the condition number simplifies to:  
  $$
  \text{cond}_f(z) = \left| \frac{z f'(z)}{f(z)} \right|.
  $$  
  This measures the sensitivity of $ f(z) $ to small perturbations in $ z $.

  ### Example of Conditioning for $ P(x) = \sin(x) $

1. **Error Amplification in $ \sin(z) $:**  
   Given $ z = \frac{22}{7} $, the data point has an uncertainty of approximately $ \pi - \frac{22}{7} \approx 0.00126 $.  
   Since $ \sin(x) $ is a periodic function with high sensitivity near certain values (e.g., multiples of $ \pi $), the relative error in $ \sin(z) $ can be significant.  
   Specifically, for $ z = \frac{22}{7} $, the relative error in $ \sin(z) $ can reach **100%**, making $ z $ highly **ill-conditioned** with respect to $ \sin(z) $. 

---

2. **Condition Number for $ \sin(x) $:**  
   The condition number of $ z $ with respect to $ \sin(z) $ is given by:
   $$
   \text{cond}_{\sin}(z) = |z \cot(z)|.
   $$
   For $ z = \frac{22}{7} $, we calculate:
   $$
   \text{cond}_{\sin}(22/7) \approx 2485.47.
   $$

---

3. **Relative Error Bound for Perturbation:**  
   If $ z $ is perturbed to $ z + \delta z = \pi $, the asymptotic relative error bound from **Fact 2** is:
   $$
   \frac{\sin(z + \delta z) - \sin(z)}{\sin(z)} \leq \text{cond}_{\sin}(z) \frac{\delta z}{z} + o\left(\frac{\delta z}{z}\right).
   $$
   Substituting the values:
   $$
   \frac{\sin(z + \delta z) - \sin(z)}{\sin(z)} = 1.
   $$
   This shows the actual relative error reaches its theoretical upper bound, confirming the ill-conditioned nature of $ z = \frac{22}{7} $ with respect to $ \sin(z) $.


   ### Conditioning Examples in Numerical Computation

---

#### 2. **Subtractive Cancellation:**

For $ x \in \mathbb{R}^2 $, define the computational problem:  
$$
P(x) = [1, -1]x = x_1 - x_2.
$$

- **Gradient of $ P(x) $:**  
  The gradient is constant and independent of $ x $:
  $$
  \nabla P(x) = [1, -1].
  $$

- **Condition Number:**  
  Using the $ \infty $-norm and applying **Fact 3**, the condition number is:
  $$
  \text{cond}_{P}(x) = \frac{\|\nabla P(x)\|_\infty \|x\|_\infty}{|P(x)|}.
  $$
  Substituting the expressions:
  $$
  \text{cond}_{P}(x) = \frac{2 \max(|x_1|, |x_2|)}{|x_1 - x_2|}.
  $$

- **Analysis:**  
  This condition number becomes large when $ x_1 \approx x_2 $, indicating **ill-conditioning**.  
  This reflects the challenge of **subtractive cancellation**, where small differences between nearly equal values $ x_1 $ and $ x_2 $ can lead to significant relative errors in $ P(x) $.

---

#### 3. **Conditioning of Matrix–Vector Multiplication:**

For a fixed matrix $ A \in \mathbb{F}^{m \times n} $, define the computational problem:  
$$
P(x) = Ax, \quad \text{where } x \in \mathbb{F}^n.
$$

- **Relative Condition Number:**  
  The condition number of $ x $ with respect to $ P(x) $ is:
  $$
  \text{cond}(x) = \frac{\|A\| \|x\|}{\|Ax\|},
  $$
  where the matrix norm $ \|A\| $ is the **operator norm** induced by the chosen vector norm $ \|\cdot\| $.

- **Special Case (Square and Nonsingular $ A $):**  
  If $ A $ is square and nonsingular, the relative condition number is bounded by:
  $$
  \text{cond}(x) \leq \|A\| \|A^{-1}\|.
  $$
  Here, $ \|A\| \|A^{-1}\| $ is the **condition number of the matrix $ A $**, which measures the sensitivity of the solution to small perturbations in $ A $ or $ x $.

  ### 4. **Conditioning of Polynomial Zeros**

Let $ q(x) = x^2 - 2x + 1 $, a quadratic polynomial with a double root at $ x = 1 $. The computational task is to find the roots of $ q(x) $ based on its power basis coefficients $[1, -2, 1]$. Here are the key observations:

- **Perturbation and Root Sensitivity:**  
  If $ q(x) $ is perturbed by a small error $ \epsilon $, the polynomial becomes $ q(x) + \epsilon = x^2 - 2x + 1 + \epsilon $.  
  The double root at $ x = 1 $ splits into two roots:
  $$
  x = 1 \pm \sqrt{\epsilon}.
  $$

  A **relative error** of $ \epsilon $ in the coefficients leads to a **relative error** of $ \sqrt{\epsilon} $ in the roots.

- **Infinite Condition Number:**  
  For small $ \epsilon $, the roots change dramatically, even for tiny perturbations. Specifically, as $ \epsilon \to 0 $, the **rate of change** of the roots becomes infinite.  
  The condition number of the coefficients $[1, -2, 1]$ for finding the roots is thus **infinite**.

- **Insight:**  
  The example highlights that **polynomial root finding is highly ill-conditioned** when the polynomial has multiple or near-multiple roots.  
  However, strictly speaking, it is the **coefficients** that are ill-conditioned, not the roots themselves, as the coefficients serve as the input data for this calculation.

---

### 5. **Wilkinson Polynomial**

The Wilkinson polynomial is defined as:
$$
w(x) = (x - 1)(x - 2)\cdots(x - 20),
$$
or equivalently:
$$
w(x) = x^{20} - 210x^{19} + 20615x^{18} - \cdots + 2432902008176640000.
$$

- **Ill-Conditioning of Roots:**  
  Although the roots $ 1, 2, 3, \ldots, 20 $ are distinct, they are highly sensitive to small changes in the polynomial's coefficients, particularly the coefficient of $ x^{19} $.

- **Perturbation Example:**  
  Perturb the $ x^{19} $-coefficient from $ -210 $ to $ -210 - 2^{-23} $ ($ \approx -210 - 1.12 \times 10^{-7} $).  
  This small change causes drastic shifts in some roots:
  - Roots $ 16 $ and $ 17 $ shift to a **complex conjugate pair** approximately equal to:
    $$
    16.73 \pm 2.81i.
    $$

- **Condition Numbers of Perturbed Roots:**  
  For the root near $ 16 $ (denoted $ P_{16}(z) $) and the root near $ 17 $ ($ P_{17}(z) $), the condition numbers with respect to the perturbed coefficient $ z = 210 $ are:
  $$
  \text{cond}_{16}(210) \approx 3 \times 10^{10}, \quad \text{cond}_{17}(210) \approx 2 \times 10^{10}.
  $$

- **Asymptotic Region Failure:**  
  The condition numbers are so large that even a perturbation as small as $ 2^{-23} $ falls **outside the asymptotic region** where the higher-order terms $ o(\delta z / z) $ in Fact 2 can be neglected.

---

### Insights:
1. Polynomial root finding from power basis coefficients is inherently **ill-conditioned**, especially for polynomials with closely spaced, multiple, or near-multiple roots.
2. In practice, numerical algorithms must take such sensitivities into account, often using alternative representations (e.g., Chebyshev polynomials or orthogonal bases) to mitigate ill-conditioning.

### **Numerical Stability and Instability**

Numerical stability is a crucial concept in numerical analysis, concerning how well an algorithm handles errors introduced during computation, such as rounding or truncation errors. Here's an outline based on the definitions provided:

---

### **1. Definitions**

#### **Forward Error**
- **Definition:** The difference between the exact function evaluation $ f(x) $ and the perturbed function evaluation $ \hat{f}(x) $:
  $$
  \text{Forward error} = f(x) - \hat{f}(x).
  $$

#### **Backward Error**
- **Definition:** A vector $ e \in \mathbb{R}^n $ of the smallest norm for which:
  $$
  f(x + e) = \hat{f}(x).
  $$
  If no such $ e $ exists, the backward error is undefined.
- **Interpretation:** The backward error measures how far the input $ x $ would need to be perturbed to make the perturbed output $ \hat{f}(x) $ an exact evaluation.

---

### **2. Numerical Stability**

#### **Forward Stability**
- **Definition:** An algorithm is forward stable if the **forward relative error** is small for all valid inputs $ x $:
  $$
  \frac{|f(x) - \hat{f}(x)|}{|f(x)|} \leq \epsilon,
  $$
  where $ \epsilon $ is a modest multiple of the unit roundoff error or truncation error.
- **Interpretation:** The computed solution is close to the exact solution, within the error limits dictated by the precision of the input.

#### **Backward Stability**
- **Definition:** An algorithm is backward stable if the backward error exists and satisfies:
  $$
  \frac{|e|}{|x|} \leq \epsilon,
  $$
  for all valid inputs $ x $, where $ \epsilon $ is small.
- **Interpretation:** A backward stable algorithm effectively computes the exact solution to a nearby problem. The perturbation in the input is proportional to the inherent errors in the computation.

#### **Strong Stability:**  
Backward stability is sometimes referred to as strong stability due to its rigorous error guarantees.

---

### **3. Practical Implications**

- **Stable Algorithms:** Produce results that are **as accurate as the input data allows**. Errors introduced during computation do not grow disproportionately.
- **Unstable Algorithms:** Amplify rounding or truncation errors, potentially leading to results that are much less accurate than the errors in the input data would suggest.

---

### **4. Examples**

- **Forward Stable Algorithms:** Gaussian elimination with partial pivoting is forward stable for most practical problems.
- **Backward Stable Algorithms:** Many algorithms for solving linear systems or eigenvalue problems (e.g., LU decomposition or QR decomposition) are designed to be backward stable, ensuring numerical robustness.

---

### **5. Error Size**
The term **"small"** in these definitions usually means a **modest multiple of the size of the input errors**, often determined by the machine precision or rounding unit.

---

### **Summary**
Numerical stability ensures that computational errors introduced during algorithm execution do not significantly degrade the accuracy of the results. Forward stability focuses on small output errors, while backward stability ensures that the computed result corresponds to the exact solution of a slightly perturbed problem.


### **Weak Numerical Stability vs Numerical Stability**

Numerical stability in algorithms characterizes how errors from rounding and truncation affect the accuracy of results. Here's a detailed breakdown based on the provided definitions:

---

### **1. Weak Numerical Stability**

#### **Definition:**
A numerical algorithm is **weakly numerically stable** if rounding and truncation errors cause it to evaluate a perturbed function $ \hat{f}(x) $ satisfying:
$$
\frac{|f(x) - \hat{f}(x)|}{|f(x)|} \leq \epsilon \cdot \text{cond}(f(x)),
$$
for all valid inputs $ x $.

#### **Key Characteristics:**
- The **forward error** is **proportional to the condition number** of the input $ x $ and a small error factor $ \epsilon $ (such as the machine precision).
- **Forward Error Bound:** The error magnitude is no worse than what would result from **perturbing the data by a small multiple of the unit roundoff.**
- **No Backward Error Guarantee:** Weak numerical stability does not ensure the existence of a backward error or that the computed result corresponds to a small perturbation of the input.
- **Weaker Variant:** An even weaker form of weak stability guarantees the relative error bound only when the data is **well-conditioned.**

#### **Implications:**
Weak numerical stability allows for larger errors in ill-conditioned inputs, where the condition number is high. This makes the method less robust in general compared to numerically stable algorithms.

---

### **2. Numerical Stability**

#### **Definition:**
A numerical algorithm is **numerically stable** if rounding and truncation errors cause it to evaluate a perturbed function $ \hat{f}(x) $ satisfying:
$$
\frac{|f(x + e) - \hat{f}(x)|}{|f(x)|} \leq \epsilon,
$$
where $ e $ is a small **relative-to-$ x $** backward error such that $ |e| \leq \epsilon |x| $.

#### **Key Characteristics:**
- **Backward Error Guarantee:** The computed value $ \hat{f}(x) $ corresponds to the exact function $ f(x) $ evaluated at a nearby data point $ x + e $.
- **Robustness:** Numerical stability ensures that the result is consistent with a slight perturbation in the input.
- **Behavior for Ill-Conditioned Inputs:** While the forward error can still be large for ill-conditioned inputs (due to high condition numbers), the algorithm remains robust because the backward error is small.

---

### **3. Comparisons**

| **Aspect**                  | **Weak Numerical Stability**                                    | **Numerical Stability**                                     |
|-----------------------------|---------------------------------------------------------------|-----------------------------------------------------------|
| **Forward Error**            | Proportional to $ \epsilon \cdot \text{cond}(f(x)) $.       | Proportional to $ \epsilon $, independent of condition number. |
| **Backward Error**           | No guarantee or may not exist.                               | Guaranteed to exist and be small ($ |e| \leq \epsilon |x| $).  |
| **Error Interpretation**     | May not correspond to small perturbations of the input.       | Corresponds to a small perturbation of the input.          |
| **Handling Ill-Conditioned Data** | Errors can grow with condition number.                       | Backward error remains small; forward error may still be large. |

---

### **4. Visualization (Figure 37.2)**

- **Weak Stability:**  
  The computed value $ \hat{f}(x) $ lies within a larger circle around $ f(x) $, which includes errors consistent with the condition number. The result may not correspond to a small input perturbation.
  
- **Numerical Stability:**  
  The computed value $ \hat{f}(x) $ lies **near** or inside the image of the small perturbations $ x + e $. This implies that the result corresponds to a slightly perturbed input.

- **Backward Stability:**  
  The computed value $ \hat{f}(x) $ lies entirely within the shaded image of the small perturbations of $ x $.

---

### **5. Summary**

- **Weak Stability** provides weaker guarantees and is more error-prone, especially for ill-conditioned data, as it allows larger forward errors without requiring backward stability.
- **Numerical Stability** ensures more robust performance by tightly linking the computed result to a small perturbation in the input, making it preferable for reliable computations.


### **Key Facts about Numerical Stability and Backward Stability**

#### **1. Numerical Stability and Ill-Conditioned Problems**
- **Ill-Conditioned Problems:** Even a numerically stable algorithm may yield **inaccurate results** when applied to an ill-conditioned problem. This is because small errors in the input data (e.g., rounding or measurement errors) can amplify into large errors in the solution due to the problem's inherent sensitivity.
- **No Algorithmic Correction:** A numerical algorithm cannot compensate for errors already present in the input data or create information that wasn't implicitly there. For ill-conditioned problems, inaccuracies are often unavoidable.

---

#### **2. Backward Stability**
- **Perturbation Equivalent:** A **backward stable algorithm** ensures that its rounding and truncation errors are equivalent to solving the problem with a slightly perturbed input. The computed results are **realistic** and consistent with an exact arithmetic solution for the perturbed data.
- **Negligibility:** In most cases, this additional error due to backward instability is negligible compared to the inherent data errors.

---

#### **3. Forward Error in Backward Stability**
- **Condition Number Dependency:** The **forward error** in a backward stable algorithm follows the **condition number bound** (Fact 2, Section 37.4). This means the forward error can grow in proportion to the problem's condition number, especially for ill-conditioned data.

---

#### **4. Examples of Backward Stable Algorithms**
The following are well-known backward stable algorithms and their properties:

1. **Single Floating Point Operations:**  
   - Any single floating-point operation (e.g., addition, multiplication) is **both forward and backward stable**.

2. **Dot Product Algorithm (Naive):**  
   - The naive dot product algorithm is **backward stable**, but not generally forward stable due to potential **cancellation of significant digits** during summation.

3. **Gaussian Elimination:**  
   - **With complete pivoting:** Strictly backward stable.  
   - **With partial pivoting:** Not strictly backward stable, but considered **“backward stable in practice”** since instability cases are extraordinarily rare.

4. **Triangular Back Substitution:**  
   - The back-substitution algorithm computes a solution $ \hat{x} $ such that it solves a nearby system $ (T + E)\hat{x} = b $, where $ |e_{ij}| \leq |t_{ij}| $.  
   - **Backward Stable:** Ensures the computed solution is consistent with a slightly perturbed system.

5. **QR Factorization (Householder and Givens Methods):**  
   - Both methods for $ A = QR $, where $ Q $ is orthogonal and $ R $ is upper triangular, are **backward stable**.

6. **Singular Value Decomposition (SVD):**  
   - The Golub–Kahan–Reinsch algorithm is **backward stable** for computing $ A = U \Sigma V^T $, where $ U $ and $ V $ are orthogonal matrices, and $ \Sigma $ is diagonal.

7. **Least-Squares Problems:**  
   - The following methods for solving least-squares problems are **backward stable**:  
     - Householder QR Factorization.  
     - Givens QR Factorization.  
     - Singular Value Decomposition (SVD).

8. **Eigenvalue Computations:**  
   - The implicit double-shift QR iteration is **backward stable** for finding eigenvalues.

---

### **Practical Implications**
- **Selection of Algorithms:** Backward stable algorithms are highly preferred in practice because they ensure realistic results for small perturbations, even if forward errors are amplified for ill-conditioned data.
- **Understanding Stability:** Recognizing whether an algorithm is backward or forward stable helps predict its behavior in different numerical contexts.

### **Procedures to Compute Stability and Conditioning**

To analyze the **numerical stability** of algorithms and the **conditioning** of computational problems, follow these structured steps:

---

### **1. Compute Problem Conditioning**

#### **Definition:**  
Conditioning measures how sensitive a problem's output is to small perturbations in the input.

#### **Steps to Compute Conditioning**
1. **Define the Function $ f(x) $:**  
   Let $ f : \mathbb{R}^n \to \mathbb{R}^m $ represent the function you are analyzing.

2. **Determine Sensitivity:**  
   Compute the **condition number**:
   $$
   \text{cond}_f(x) = \frac{\|D(x)\| \cdot \|x\|}{\|f(x)\|}
   $$
   where $ D(x) $ is the derivative (or Jacobian) of $ f(x) $.

3. **Interpret Results:**
   - If $ \text{cond}_f(x) $ is large, the problem is **ill-conditioned**, meaning small input errors may lead to large output errors.
   - If $ \text{cond}_f(x) $ is small, the problem is **well-conditioned**.

#### **Example: Polynomial Root Finding**
For $ q(x) = x^2 - 2x + 1 $:
- Double root at $ x = 1 $.
- Perturbation $ \epsilon $ in coefficients causes root error $ \sqrt{\epsilon} $, leading to infinite condition number for the coefficients near multiple roots.

---

### **2. Assess Algorithm Stability**

#### **Definition:**  
Stability measures how rounding and truncation errors affect the accuracy of the algorithm's result.

#### **Steps to Compute Algorithm Stability**
1. **Forward Error Analysis:**
   Evaluate the forward error:
   $$
   \text{Forward Error} = \frac{|f(x) - \hat{f}(x)|}{|f(x)|}
   $$
   - If the error is small for all inputs, the algorithm is **forward stable**.

2. **Backward Error Analysis:**
   Compute the smallest perturbation $ e $ such that:
   $$
   f(x + e) = \hat{f}(x)
   $$
   - If $ \|e\| \leq \epsilon \|x\| $, the algorithm is **backward stable**.

3. **Weak Stability:**
   If rounding errors satisfy:
   $$
   \frac{|f(x) - \hat{f}(x)|}{|f(x)|} \leq \epsilon \cdot \text{cond}_f(x)
   $$
   the algorithm is **weakly stable**.

#### **Example: Naive Dot Product**
Given $ x = [a, b] $ and $ P(x) = a - b $:
- For $ a \approx b $, significant digit cancellation may occur.
- Backward stability holds, but forward stability fails because the subtraction magnifies relative errors.

---

### **3. Numerical Conditioning of Matrix Algorithms**

#### **Matrix Multiplication Conditioning:**
Condition number for $ A x $ is given by:
$$
\text{cond}(x) = \frac{\|A\| \cdot \|x\|}{\|Ax\|}
$$
For nonsingular $ A $:
$$
\text{cond}(x) \leq \|A\| \cdot \|A^{-1}\|
$$

#### **Example: Gaussian Elimination**
- Complete pivoting ensures backward stability.
- Partial pivoting may not be strictly backward stable but is considered stable in practice.

---

### **4. Numerical Stability in Least Squares Problems**

#### **Method:**
- Solve $ Ax = b $ using QR factorization or SVD.
- Stability depends on:
  - Orthogonal $ Q $ properties in QR decomposition.
  - Singular values of $ A $ in SVD.

#### **Example: QR Factorization**
Given $ A = QR $, where $ Q $ is orthogonal and $ R $ is upper triangular:
- Both Householder and Givens methods are backward stable.

---

### **5. Eigenvalue Problems**

#### **Method:**
- Compute eigenvalues using iterative techniques like QR iterations.

#### **Example: Wilkinson Polynomial**
Perturbing the $ x^{19} $ coefficient in $ w(x) = (x-1)(x-2)...(x-20) $ drastically changes eigenvalues.  
Condition numbers $ \text{cond}_{16}(210) \approx 3 \times 10^{10} $ and $ \text{cond}_{17}(210) \approx 2 \times 10^{10} $ illustrate ill-conditioning.

---

### **Conclusion**
To assess stability and conditioning:
1. Compute **condition numbers** to measure problem sensitivity.
2. Evaluate **forward** and **backward errors** for algorithm stability.
3. Analyze examples such as matrix operations, polynomial roots, or iterative eigenvalue computations to understand practical implications.

### **Examples of Numerical Stability and Conditioning**

---

### **1. Forward Stable but Not Backward Stable: Outer Product Algorithm**  
**Example:** Compute $ A = xy^T $, where $ x, y \in \mathbb{R}^n $.  
- **Forward Stability:**  
  - The computed $ A $ (outer product of $ x $ and $ y $) is correctly rounded, ensuring the forward relative error is small.
  - Example: If $ x = [1, 2] $, $ y = [3, 4] $, then $ A = \begin{bmatrix} 3 & 4 \\ 6 & 8 \end{bmatrix} $.
- **Backward Instability:**  
  - Rounding errors perturb $ A $ into a matrix of higher rank, making it impossible to represent $ A $ as $ xy^T $ for perturbed $ x $ and $ y $.

---

### **2. Backward vs. Forward Errors: Taylor Series Approximation**  
**Example:** Evaluate $ f(x) = e^x $ at $ x = 1 $ using the Taylor series approximation:  
$$
\hat{f}(x) = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!}.
$$
- **Forward Error:**  
  $$
  \text{Forward Error} = |f(1) - \hat{f}(1)| \approx 2.7183 - 2.6667 = 0.0516.
  $$
- **Backward Error:**  
  Solve for $ y $ such that $ f(y) = \hat{f}(1) $. The backward error is:
  $$
  1 - \ln(\hat{f}(1)) \approx 1 - \ln(2.6667) = 0.0192.
  $$

---

### **3. Numerically Unstable Algorithm: Logarithm Near Zero**  
**Example:** Evaluate $ f(x) = \ln(1 + x) $ for $ x \approx 0 $ using $ \text{fl}(\ln(1 \oplus x)) $:  
- For $ x = 10^{-16} $ in 16-digit arithmetic:
  - $ \ln(1 \oplus 10^{-16}) $ is computed as $ \ln(1) = 0 $ due to rounding.  
  - Exact value $ f(x) = 10^{-16} $, so relative error = 100%.

- **Analysis:**
  - The function is well-conditioned ($ \text{cond}_f(x) = 1 $ as $ x \to 0 $), but the algorithm is numerically unstable due to precision loss.

---

### **4. Improved Logarithm Algorithm: Taylor Series**  
**Example:** Use the Taylor expansion:
$$
f(x) = \ln(1 + x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots.
$$
- **Advantages:**
  - For $ x \approx 0 $, higher precision is achieved with a few terms.
- **Limitations:**
  - Converges slowly for $ |x| \approx 1 $.
  - Alternative algorithms like $ \text{fl}(\ln(1 \oplus x)) $ may be required for $ |x| > 1 $.

---

### **5. Gaussian Elimination Without Pivoting**  
**Example:** Solve:
$$
\begin{aligned}
10^{-10}x_1 + x_2 &= 1, \\
x_1 + 2x_2 &= 3,
\end{aligned}
$$
using 9-digit arithmetic.  
- **Steps:**
  - Eliminate $ x_1 $:
    $$
    x_2 = 1, \quad x_1 = 0.
    $$
  - Exact solution: $ x_1 = 1 - 2 \cdot 10^{-10}, \; x_2 = 1 - 3 \cdot 10^{-10} $.  
- **Analysis:**
  - Large error is due to numerical instability, not ill-conditioning ($ \kappa(A) \approx 9 $).

---

### **6. Unstable Algorithm for Eigenvalue Computations**  
**Example:** Compute eigenvalues of $ A = \text{diag}(1, 2, 3, \ldots, 20) $ by finding roots of the characteristic polynomial.  
- **Problem:**
  - The Wilkinson polynomial has highly ill-conditioned roots, even though the eigenvalues themselves are well-conditioned ($ \kappa \leq \|E\|_F $ for perturbation $ E $).
  - Transformation to companion form introduces instability.  

---

### **7. Alternative Eigenvalue Algorithms**  
Use iterative QR methods to avoid polynomial root finding instability:
- Implicit double-shift QR iteration is **backward stable**.
- It directly computes eigenvalues without forming ill-conditioned polynomials.

---

### **Conclusion**
- Numerical stability and conditioning depend on both the algorithm and the problem.  
- Forward stability focuses on relative error in results, while backward stability ensures results align with a slightly perturbed input.  
- Examples highlight the importance of carefully choosing stable algorithms for ill-conditioned problems.