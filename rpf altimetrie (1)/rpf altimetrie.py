"""
Terrain-aided navigation by radar-altimeter measurements with a
regularized particle filter (RPF, Epanechnikov kernel).
Scenario data: TP4 2026 ENSTA (300 km/h, heading 45°, 60 km, z0 = 8000 m).

Usage: python rpf_altimetrie.py [carte.mat] [seed]
In Spyder: Run (F5); the 2 figures appear in the Plots pane (no file export).
"""
import sys
import os
import numpy as np
from math import gamma, pi, sqrt
from scipy.io import loadmat

# Spyder / IPython: force inline backend so that figures stay docked in the
# Plots pane (no floating windows), whatever the Spyder graphics preference.
try:
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821
except NameError:
    pass   # plain Python: default backend

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Terrain model
# ---------------------------------------------------------------------------
class Terrain:
    def __init__(self, matfile):
        m = loadmat(matfile)
        self.h_MNT = np.asarray(m["h_MNT"], dtype=float)
        self.pasx = float(m["pasx_reel"].squeeze())
        self.pasy = float(m["pasy_reel"].squeeze())
        self.x_MNT = np.asarray(m["x_MNT"], dtype=float).ravel()
        self.y_MNT = np.asarray(m["y_MNT"], dtype=float).ravel()
        self.nrow = self.h_MNT.shape[0]
        # column-major flat view to reproduce MATLAB linear indexing exactly
        self.hF = self.h_MNT.ravel(order="F")

    def extent_km(self):
        return [self.x_MNT[0] * self.pasx / 1000, self.x_MNT[-1] * self.pasx / 1000,
                self.y_MNT[-1] * self.pasy / 1000, self.y_MNT[0] * self.pasy / 1000]

    def hobs(self, X):
        """Bilinear interpolation of terrain height at (X[0], X[1]).
        x = column index, y = row index of h_MNT (image convention)."""
        ncol = self.h_MNT.shape[1]
        # clamp to the map (particles leaving the DEM would index out of bounds)
        lam = np.clip(X[0] / self.pasx, 0.0, ncol - 1 - 1e-9)
        ix = np.floor(lam)
        lam = lam - ix
        mu = np.clip(1.0 + X[1] / self.pasy, 1.0, self.nrow - 1e-9)
        iy = np.floor(mu)
        mu = mu - iy
        k = (self.nrow * ix + iy).astype(np.int64) - 1   # MATLAB 1-based -> 0-based
        n = self.nrow
        hF = self.hF
        return ((1 - lam) * (1 - mu) * hF[k] + lam * (1 - mu) * hF[k + n]
                + (1 - lam) * mu * hF[k + 1] + lam * mu * hF[k + n + 1])


# ---------------------------------------------------------------------------
# Epanechnikov sampler (Deheuvels; Devroye p. 236)
# density ∝ (1 - ||x||^2) on the unit ball of R^d
# ---------------------------------------------------------------------------
def gene_epanech(d, n, rng):
    # ||X||^2 ~ Beta(d/2, 2). The MATLAB code draws it with Cheng's rejection
    # algorithm; numpy's Beta sampler gives the same distribution.
    beta = rng.beta(d / 2.0, 2.0, size=n)
    g = rng.standard_normal((d, n))
    return g * (np.sqrt(beta) / np.linalg.norm(g, axis=0))


def select(p, rng):
    """Multinomial resampling."""
    c = np.cumsum(p)
    c[-1] = 1.0
    return np.searchsorted(c, rng.random(p.size))


# ---------------------------------------------------------------------------
def main(matfile, seed=None, ChoixNoyau=2, coefR=1.5, plot=True, verbose=True):
    rng = np.random.default_rng(seed)
    terr = Terrain(matfile)
    d = 6

    # ======================= Données du scénario (TP4) =======================
    VitAvion = 300 * 1000 / 3600           # vitesse avion (m/s)
    CapAvion = 45 * pi / 180               # cap avion (rad)
    LongTraject = 60 * 1000                # longueur du vol (m)
    EtatVraiInitial = np.array([150 * 1000, 150 * 1000, 8000,
                                VitAvion * np.cos(CapAvion),
                                VitAvion * np.sin(CapAvion), 0.0])
    dt = 1.0                               # période de mesure (s)
    SigmaBruit = 30.0                      # écart-type bruit de mesure (m)
    # Incertitude initiale : σxy = 3000 m, σz = 500 m, σv = 5 m/s  (donne P0)
    P0 = np.diag(np.array([3000, 3000, 500, 5, 5, 5], dtype=float) ** 2)
    RacineP0 = np.linalg.cholesky(P0)
    T = int(round(LongTraject / (VitAvion * dt)))   # nombre de pas (720)

    # ===================== Filtre particulaire régularisé ====================
    NbPart = 5000                          # nb de particules
    Np = NbPart
    # ChoixNoyau : 1 = Epanechnikov, 2 = gaussien (TP4 : gaussien)
    Nth = 0.5 * Np          # critère de redistribution (non modifié)
    mu = 0.3                # coefficient du h optimal (non modifié)
    # ==========================================================================

    F = np.eye(d)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    Q = np.diag(np.array([.5, .5, .1, .1, .1, .01]) ** 2)
    LQ = np.linalg.cholesky(Q)

    # True trajectory (deterministic: straight line at constant velocity)
    Xt = np.zeros((d, T))
    Xt[:, 0] = EtatVraiInitial
    for t in range(1, T):
        Xt[:, t] = F @ Xt[:, t - 1]

    # Radar-altimeter measurements: h = z - h_terrain(x, y) + v
    R = SigmaBruit ** 2
    h_mes = Xt[2] - terr.hobs(Xt) + sqrt(R) * rng.standard_normal(T)

    # Filter-side measurement variance, inflated (tuned on the first scenario)
    R_f = (coefR ** 2) * R

    # Optimal bandwidth (Gaussian target), times the over-smoothing factor mu
    if ChoixNoyau == 1:      # Epanechnikov
        cd = pi ** (d / 2) / gamma(d / 2 + 1)
        KA = ((8 / cd) * (d + 4) * (2 * sqrt(pi)) ** d) ** (1 / (d + 4))
        h_RPF = mu * KA * Np ** (-1 / (d + 4))
    else:                    # Gaussian
        h_RPF = mu * (4 / (Np * (d + 2))) ** (1 / (d + 4))

    def kernel(n):
        if ChoixNoyau == 1:
            return gene_epanech(d, n, rng)
        return rng.standard_normal((d, n))

    # Initialisation : centre du nuage tiré uniformément dans un carré,
    # particules gaussiennes autour de ce centre, de covariance P0
    Xe = np.zeros((d, T))
    Xstd = np.zeros((d, T))                 # weighted particle std (posterior)
    CentreNuageInitial = EtatVraiInitial + RacineP0 @ (2 * rng.random(d) - 1)
    Xp = CentreNuageInitial[:, None] + RacineP0 @ rng.standard_normal((d, Np))
    p = np.full(Np, 1.0 / Np)
    Xe[:, 0] = Xp @ p
    Xstd[:, 0] = np.sqrt(((Xp - Xe[:, 0][:, None]) ** 2) @ p)

    Neff_hist = np.full(T, np.nan)
    Neff_hist[0] = Np
    resampled = np.zeros(T, dtype=bool)

    for t in range(1, T):
        # Prediction
        Xp = F @ Xp + LQ @ rng.standard_normal((d, Np))
        # Correction (log-domain for numerical safety)
        inno = h_mes[t] - Xp[2] + terr.hobs(Xp)
        logw = np.log(p) - inno ** 2 / (2 * R_f)
        logw -= logw.max()
        p = np.exp(logw)
        p /= p.sum()
        Xe[:, t] = Xp @ p
        Xstd[:, t] = np.sqrt(((Xp - Xe[:, t][:, None]) ** 2) @ p)

        Neff = 1.0 / np.sum(p ** 2)
        Neff_hist[t] = Neff
        msg = f"Obs. {t + 1:3d}: Neff = {Neff:.4f}"
        if Neff < Nth:
            Xc = Xp - Xe[:, t][:, None]
            At = (Xc * p) @ Xc.T
            LA = np.linalg.cholesky(At)
            Xp = Xp[:, select(p, rng)] + h_RPF * LA @ kernel(Np)
            p = np.full(Np, 1.0 / Np)
            resampled[t] = True
            msg += " resampling"
        if verbose:
            print(msg)

    temps = np.arange(1, T + 1) * dt
    err = Xt - Xe

    if not plot:
        return Xt, Xe

    # ================================================================ plots
    ext = terr.extent_km()

    # --- Fig. 1: map, full view + zoom on the trajectory -------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    for a in (a1, a2):
        im = a.imshow(terr.h_MNT, extent=ext, origin="upper", aspect="equal", cmap="terrain")
        a.plot(Xt[0] / 1000, Xt[1] / 1000, "-k", lw=1.5, label="vraie")
        a.plot(Xe[0] / 1000, Xe[1] / 1000, ".r", ms=3, label="estimée")
        a.plot(Xt[0, 0] / 1000, Xt[1, 0] / 1000, "ok", mfc="none", ms=8, label="départ")
        a.set_xlabel("km"); a.set_ylabel("km"); a.grid(True)
    a1.set_title("44° N, 3° E")
    pad = 5.0
    a2.set_xlim(Xt[0].min() / 1000 - pad, Xt[0].max() / 1000 + pad)
    a2.set_ylim(Xt[1].max() / 1000 + pad, Xt[1].min() / 1000 - pad)
    a2.set_title("zoom sur la trajectoire"); a2.legend(loc="lower left")
    fig.colorbar(im, ax=(a1, a2), label="altitude du relief (m)", shrink=0.8)

    # --- Fig. 2: errors with ±3σ particle envelopes ------------------------
    labels = ["x (m)", "y (m)", "z (m)", "vx (m/s)", "vy (m/s)", "vz (m/s)"]
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for i, a in enumerate(axs.T.ravel()):
        a.plot(temps, err[i], "b", label="erreur réelle")
        a.plot(temps, 3 * Xstd[i], "r", lw=0.8, label="±3σ particules")
        a.plot(temps, -3 * Xstd[i], "r", lw=0.8)
        a.set_ylabel(labels[i]); a.grid(True)
        s = np.nanpercentile(np.abs(np.r_[err[i, 20:], 3 * Xstd[i, 20:]]), 99)
        a.set_ylim(-1.5 * s, 1.5 * s)      # zoom on the post-transient part
    for a in axs[-1]:
        a.set_xlabel("temps (s)")
    axs[0, 0].legend(loc="upper right")
    fig.suptitle("Erreur réelle (bleu) et écart-type des particules ±3σ (rouge)")
    fig.tight_layout()

    # ---------------------------------------------------------------- summary
    tail = slice(T // 2, T)
    print("\nRésumé")
    print(f"  h_RPF = {h_RPF:.4f}, rééchantillonnages : {resampled.sum()}/{T - 1}")
    names = ["x", "y", "z", "vx", "vy", "vz"]
    print("  erreur finale      :", ", ".join(f"{n}={e:+.2f}" for n, e in zip(names, err[:, -1])))
    print("  RMSE (2e moitié)   :", ", ".join(
        f"{n}={np.sqrt(np.mean(err[i, tail] ** 2)):.2f}" for i, n in enumerate(names)))
    return Xt, Xe


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1]:
        mat = sys.argv[1]
    else:
        # look for the map next to this script (Spyder: Run without arguments)
        here = os.path.dirname(os.path.abspath(__file__))
        mat = next((os.path.join(here, f) for f in ("carte.mat", "carte_2.mat")
                    if os.path.exists(os.path.join(here, f))), "carte.mat")
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(mat, sd)
    plt.show()
