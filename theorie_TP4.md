# TP4: Particle filtering for terrain-aided navigation (theory summary)

## 1. The problem

An aircraft flies in **uniform rectilinear motion**. Its state is
$X_k = (x_k, y_k, z_k, \dot x_k, \dot y_k, \dot z_k)^T$.

$$
\begin{cases}
X_k = \Phi\, X_{k-1} + \eta_k & \text{(dynamics)}\\[2pt]
Y_k = z_k - h_{\text{MNT}}(x_k, y_k) + \epsilon_k & \text{(measurement)}
\end{cases}
\qquad
\Phi = \begin{pmatrix} I_3 & dt\, I_3 \\ 0 & I_3 \end{pmatrix}
$$

- The **measurement** is the radio-altimeter height above ground: the aircraft altitude minus the terrain height from the Digital Terrain Model (MNT). The noise $\epsilon_k \sim \mathcal N(0, \sigma^2)$ is Gaussian.
- $h_{\text{MNT}}$ is a **lookup table, strongly nonlinear and non-differentiable**. The posterior is often **multimodal**, because several places can have the same terrain profile. A Kalman filter or EKF does not handle this well, so the TP uses a **particle filter**.

> Implementation note: the terrain array is indexed `MNT(y, x)`.

## 2. Particle filter (SIR: Sampling Importance Resampling)

The posterior is approximated by $N$ weighted samples (particles):

$$
p(x_k \mid Y_{1:k}) \approx \sum_{j=1}^{N} w_k^j\, \delta(x - X_k^j)
$$

**Algorithm**
1. **Initialization:** draw $X_0^j \sim p(x_0)$ (here $\mathcal N(\cdot, P_0)$) and set $w_0^j = 1/N$.
2. **Prediction:** $X_{k|k-1}^j = \Phi X_{k-1}^j + \eta_k^j$. This samples from $p(x_k \mid x_{k-1})$.
3. **Correction:** multiply each weight by the likelihood, then normalize:
   $$\tilde w_k^j = w_{k-1}^j\, p(Y_k \mid X_{k|k-1}^j),\qquad
   p(Y_k\mid X) \propto \exp\!\Big(-\tfrac{(Y_k - z + h_{\text{MNT}}(x,y))^2}{2\sigma^2}\Big),\qquad
   w_k^j = \frac{\tilde w_k^j}{\sum_i \tilde w_k^i}$$
4. **Estimation:** compute the weighted mean and covariance:
   $$\hat X_k = \sum_j w_k^j X_k^j,\qquad P_k = \sum_j w_k^j (X_k^j - \hat X_k)(X_k^j - \hat X_k)^T$$
5. **Resampling, only when needed:** if $N_{\text{eff}} < N_{\text{th}}$, resample and reset all weights to $1/N$.

## 3. Degeneracy and resampling

- **Degeneracy:** weights are multiplied together at every step. After a while, nearly all the weight sits on **one particle (trajectory)** and the other particles contribute nothing.
- **Detection:** use the effective sample size
  $N_{\text{eff}} = 1 / \sum_j (w_k^j)^2$. It lies between 1 (fully degenerate) and $N$ (uniform weights).
  Resample when $N_{\text{eff}} < N_{\text{th}} = c \cdot N$ (in the TP, `CoefResampling = 0.4`).
- **Multinomial resampling:** draw $N$ times **with replacement**, choosing particle $j$ with probability $w^j$. Heavy particles get duplicated and light ones disappear. Afterwards every weight is $1/N$.

**Drawbacks of plain resampling**
- After resampling, many particles are **identical copies**. Diversity only comes back through the dynamic noise $\eta_k$. If that noise is **small**, the filter explores the state space poorly (sample impoverishment).
- Modes can be **lost** when the posterior is strongly multimodal.
- The method struggles when the predicted cloud and the likelihood **overlap little**. This happens with precise measurements or a terrain signal that varies a lot.

## 4. Regularized Particle Filter (RPF)

Idea: **smooth** the discrete (Dirac) approximation with a kernel, then resample from that **continuous** density. New particles are spread around the old ones instead of being exact copies.

$$
\hat p(x) = \sum_j w^j\, \delta(x - X^j)
\;\;\longrightarrow\;\;
\hat p * K_h(x) = \sum_j w^j K_h(x - X^j),\qquad K_h(x) = \frac{1}{h^d} K\!\left(\frac{x}{h}\right)
$$

**Kernel $K$:** a probability density on $\mathbb R^d$ with $K \ge 0$, $\int K = 1$, $\int xK = 0$ (zero mean) and $\int \|x\|^2 K < \infty$.

**Choosing the bandwidth $h$.** The mean squared error has two terms. One is a bias term in $h^{2s}$, which grows with $h$. The other is a variance term in $\frac{1}{Nh}$, which shrinks with $h$. Minimizing the sum gives:

| Kernel | $h_{\text{opt}} = A(K)\, N^{-1/(d+4)}$ |
|---|---|
| Epanechnikov (MSE-optimal): $K(x) = \frac{d+2}{2c_d}(1-\|x\|^2)\,\mathbb 1_{\|x\|<1}$ | $A(K) = \big[8 c_d^{-1}(d+4)(2\sqrt\pi)^d\big]^{1/(d+4)}$ |
| **Gaussian** (simpler, used in practice) | $A(K) = \big[4/(d+2)\big]^{1/(d+4)}$ |

Here $c_d$ is the volume of the unit ball in $\mathbb R^d$.
In practice, use $h = \text{Coef}\times h_{\text{opt}}$ with Coef < 1 (`CoefHopt = 0.4`) to **avoid over-smoothing**.

**Regularization step**, applied instead of plain resampling when $N_{\text{eff}} < N_{\text{th}}$. For each new particle $i$:
1. Draw an index $I$ with $P(I = j) = w_k^j$ (multinomial draw).
2. Draw $\epsilon \sim K$ (Gaussian or Epanechnikov).
3. Set $X_k^i = X_{k|k-1}^I + h_{\text{opt}}\, \Gamma_k^T \epsilon$, where $\Gamma_k^T\Gamma_k = P_k$ (for example a Cholesky factor, written $P_k^{1/2}$).

Then reset all weights to $1/N$.

Scaling by $P_k^{1/2}$ makes the kernel follow the shape of the particle cloud. The RPF is the **SIR filter with jittered resampling**, and it is most useful when the **dynamic noise is weak**.

## 5. TP scenario

- Speed 300 km/h, heading 45°, flight length 60 km, $dt = 1$ s.
- True initial state: $(150\text{ km}, 150\text{ km}, 8000\text{ m}, v\cos\psi, v\sin\psi, 0)$.
- Measurement noise: $\sigma = 30$ m.
- Initial uncertainty ($P_0$): $\sigma_{xy} = 3000$ m, $\sigma_z = 500$ m, $\sigma_{v} = 5$ m/s.
- Initialization: the cloud centre is drawn uniformly around the true state, using $\text{true} + P_0^{1/2}(2U-1)$. Then $N = 5000$ particles are drawn as $\mathcal N(\text{centre}, P_0)$.
- RPF settings: Gaussian kernel, `CoefHopt = 0.4`, `CoefResampling = 0.4`.
- Outputs to plot: position and velocity errors (with $\pm$ bounds from $P_k$), absolute-altitude error, and the resampling instants.
