# TP4 project plan: regularized particle filter for terrain-aided navigation

Five main questions for the report (binôme, M1 level). Because the filter is random, every numerical comparison should use Monte Carlo runs (for example 50 seeds) and report the RMSE and the divergence rate.

## 1. Why a particle filter for this problem?

- Write the model: uniform rectilinear motion, measurement $Y_k = z_k - h_\text{MNT}(x_k, y_k) + \epsilon_k$.
- Explain why a Kalman filter or an EKF is not suited: the terrain is a strongly nonlinear lookup table, the initial error is 3 km, and the posterior can be multimodal.
- Explain why the position is only observable through the variation of the terrain along the flight.

## 2. How does the SIR filter work, and why does it degenerate?

- Derive the weight update $w_k^j \propto w_{k-1}^j\, p(Y_k \mid X_k^j)$.
- Define $N_\text{eff} = 1/\sum_j (w^j)^2$, show that $1 \le N_\text{eff} \le N$, and explain the resampling criterion.
- Explain multinomial resampling and its drawback: identical copies, hence poor exploration when the process noise is small.

## 3. What does regularization add?

- Present the kernel smoothing of the particle cloud and the conditions on the kernel $K$.
- Explain the bias–variance trade-off that gives $h_\text{opt} = A(K)\, N^{-1/(d+4)}$, and compute it for $d = 6$, $N = 5000$.
- Explain why the kernel is scaled by $P_k^{1/2}$ and why a coefficient below 1 is applied to $h_\text{opt}$.

## 4. Does the filter converge on the TP scenario?

- Run the baseline with the statement's settings: $N = 5000$, Gaussian kernel, `CoefHopt = 0.4`, `CoefResampling = 0.4`, $\sigma = 30$ m.
- Produce the figures of the slides: true versus measured relative altitude, errors in x, y, z, vx, vy with ±3σ, and the resampling instants.
- Check consistency: does the true error stay inside ±3σ?
- Note: the provided code uses 0.5 and 0.3 for the two coefficients and inflates the measurement variance by 1.5². Align it with the statement or justify the change.

## 5. How sensitive is the filter to its parameters?

- Number of particles: $N$ from 500 to 20000 (accuracy, divergence rate, computing time).
- Regularization: `CoefHopt` from 0 (plain SIR) to 1, and Gaussian versus Epanechnikov kernel.
- Resampling threshold: `CoefResampling` from 0.1 to 1.
- Noise: process noise $Q$ (including $Q = 0$, where SIR should collapse and the RPF survive) and measurement noise $\sigma$.
- Conclude on the limits of the method: flat terrain, large initial uncertainty, multimodality.
