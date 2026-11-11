"""
Filtre particulaire régularisé (RPF) pour la navigation par corrélation de
terrain (TP4). Coeur du filtre, adapté de « rpf altimetrie.py » (script fourni).

Les réglages par défaut sont ceux de l'énoncé :
N = 5000, noyau gaussien, CoefHopt = 0.4, CoefResampling = 0.4, sigma = 30 m.
"""
import os
from math import gamma, pi, sqrt

import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
CARTE = os.path.join(HERE, "..", "rpf altimetrie (1)", "carte.mat")


class Terrain:
    def __init__(self, matfile=CARTE):
        m = loadmat(matfile)
        self.h_MNT = np.asarray(m["h_MNT"], dtype=float)
        self.pasx = float(m["pasx_reel"].squeeze())
        self.pasy = float(m["pasy_reel"].squeeze())
        self.x_MNT = np.asarray(m["x_MNT"], dtype=float).ravel()
        self.y_MNT = np.asarray(m["y_MNT"], dtype=float).ravel()
        self.nrow = self.h_MNT.shape[0]
        self.hF = self.h_MNT.ravel(order="F")

    def extent_km(self):
        return [self.x_MNT[0] * self.pasx / 1000, self.x_MNT[-1] * self.pasx / 1000,
                self.y_MNT[-1] * self.pasy / 1000, self.y_MNT[0] * self.pasy / 1000]

    def hobs(self, X):
        """Interpolation bilinéaire du relief en (X[0], X[1]).
        x = indice de colonne, y = indice de ligne de h_MNT : MNT(y, x)."""
        ncol = self.h_MNT.shape[1]
        lam = np.clip(X[0] / self.pasx, 0.0, ncol - 1 - 1e-9)
        ix = np.floor(lam)
        lam = lam - ix
        mu = np.clip(1.0 + X[1] / self.pasy, 1.0, self.nrow - 1e-9)
        iy = np.floor(mu)
        mu = mu - iy
        k = (self.nrow * ix + iy).astype(np.int64) - 1
        n = self.nrow
        hF = self.hF
        return ((1 - lam) * (1 - mu) * hF[k] + lam * (1 - mu) * hF[k + n]
                + (1 - lam) * mu * hF[k + 1] + lam * mu * hF[k + n + 1])


def gene_epanech(d, n, rng):
    """Tirage du noyau d'Epanechnikov sur la boule unité de R^d :
    direction uniforme sur la sphère, ||X||^2 ~ Beta(d/2, 2)."""
    beta = rng.beta(d / 2.0, 2.0, size=n)
    g = rng.standard_normal((d, n))
    return g * (np.sqrt(beta) / np.linalg.norm(g, axis=0))


def select(p, rng):
    """Rééchantillonnage multinomial."""
    c = np.cumsum(p)
    c[-1] = 1.0
    return np.searchsorted(c, rng.random(p.size))


def h_opt(N, d=6, kernel=2):
    """Pas optimal (cible gaussienne blanchie)."""
    if kernel == 1:      # Epanechnikov
        cd = pi ** (d / 2) / gamma(d / 2 + 1)
        A = ((8 / cd) * (d + 4) * (2 * sqrt(pi)) ** d) ** (1 / (d + 4))
    else:                # gaussien
        A = (4 / (d + 2)) ** (1 / (d + 4))
    return A * N ** (-1 / (d + 4))


_TERR = None


def terrain():
    global _TERR
    if _TERR is None:
        _TERR = Terrain()
    return _TERR


def run(seed=0, N=5000, coef_h=0.4, coef_res=0.4, kernel=2, q_scale=1.0,
        sigma_true=30.0, coef_R=1.0, sigma_xy0=3000.0, snaps=()):
    """Une réalisation du filtre.

    coef_h   : coefficient multiplicatif de h_opt (0 -> SIR, sans régularisation)
    coef_res : seuil de redistribution, N_th = coef_res * N
    kernel   : 1 = Epanechnikov, 2 = gaussien
    q_scale  : facteur multiplicatif de la covariance Q du bruit d'état
    coef_R   : rapport sigma_filtre / sigma_vrai
    snaps    : indices de temps où l'on conserve le nuage (x, y) et les poids
    """
    rng = np.random.default_rng(seed)
    terr = terrain()
    d = 6

    Vit = 300 * 1000 / 3600
    Cap = 45 * pi / 180
    X0 = np.array([150e3, 150e3, 8000.0, Vit * np.cos(Cap), Vit * np.sin(Cap), 0.0])
    dt = 1.0
    T = int(round(60e3 / (Vit * dt)))
    P0 = np.diag(np.array([sigma_xy0, sigma_xy0, 500, 5, 5, 5], dtype=float) ** 2)
    L0 = np.linalg.cholesky(P0)

    F = np.eye(d)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    LQ = sqrt(q_scale) * np.diag([.5, .5, .1, .1, .1, .01])

    Xt = np.zeros((d, T))
    Xt[:, 0] = X0
    for t in range(1, T):
        Xt[:, t] = F @ Xt[:, t - 1]

    h_vrai = Xt[2] - terr.hobs(Xt)
    h_mes = h_vrai + sigma_true * rng.standard_normal(T)
    R_f = (coef_R * sigma_true) ** 2

    h = coef_h * h_opt(N, d, kernel)

    centre = X0 + L0 @ (2 * rng.random(d) - 1)
    Xp = centre[:, None] + L0 @ rng.standard_normal((d, N))
    p = np.full(N, 1.0 / N)

    Xe = np.zeros((d, T))
    Xstd = np.zeros((d, T))
    Neff = np.full(T, float(N))
    resampled = np.zeros(T, dtype=bool)
    distinct = np.full(T, N, dtype=int)
    clouds = {}

    Xe[:, 0] = Xp @ p
    Xstd[:, 0] = np.sqrt(((Xp - Xe[:, 0][:, None]) ** 2) @ p)
    if 0 in snaps:
        clouds[0] = (Xp[:2].copy(), p.copy())

    for t in range(1, T):
        # Prédiction
        Xp = F @ Xp + LQ @ rng.standard_normal((d, N))
        # Correction (poids calculés en logarithme)
        inno = h_mes[t] - Xp[2] + terr.hobs(Xp)
        logw = np.log(p) - inno ** 2 / (2 * R_f)
        logw = np.where(np.isnan(logw), -np.inf, logw)
        logw -= logw.max()
        p = np.exp(logw)
        p /= p.sum()
        # Estimation
        Xe[:, t] = Xp @ p
        Xc = Xp - Xe[:, t][:, None]
        Xstd[:, t] = np.sqrt((Xc ** 2) @ p)
        Neff[t] = 1.0 / np.sum(p ** 2)
        if t in snaps:
            clouds[t] = (Xp[:2].copy(), p.copy())
        # Redistribution / régularisation
        if Neff[t] < coef_res * N:
            idx = select(p, rng)
            Xp = Xp[:, idx]
            if h > 0:
                At = (Xc * p) @ Xc.T
                At += 1e-9 * np.diag(np.diag(At) + 1e-12)
                LA = np.linalg.cholesky(At)
                eps = gene_epanech(d, N, rng) if kernel == 1 else rng.standard_normal((d, N))
                Xp = Xp + h * LA @ eps
            p = np.full(N, 1.0 / N)
            resampled[t] = True
        distinct[t] = np.unique(Xp[0]).size

    return dict(Xt=Xt, Xe=Xe, Xstd=Xstd, Neff=Neff, resampled=resampled,
                distinct=distinct, h_vrai=h_vrai, h_mes=h_mes, clouds=clouds,
                h=h, dt=dt, N=N, coef_res=coef_res)
