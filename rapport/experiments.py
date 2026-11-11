"""
Expériences numériques du rapport TP4 : génère rapport/figures/*.pdf et
rapport/results/summary.json.   Usage : python experiments.py
"""
import json
import os
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rpf

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
NMC = 50            # réalisations Monte-Carlo par configuration
SEUIL_DIV = 500.0   # m : erreur horizontale finale au-delà de laquelle on déclare la divergence

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def metrics(kw):
    r = rpf.run(**kw)
    e = r["Xt"] - r["Xe"]
    eh = np.hypot(e[0], e[1])
    T = eh.size
    inside = (np.abs(e[:, 100:]) <= 3 * r["Xstd"][:, 100:]).mean(axis=1)
    return dict(eh=eh, ez=np.abs(e[2]), final=eh[-1],
                rmse_half=np.sqrt(np.mean(eh[T // 2:] ** 2)),
                rmse_z_half=np.sqrt(np.mean(e[2, T // 2:] ** 2)),
                rmse_v_half=np.sqrt(np.mean(e[3:5, T // 2:] ** 2)),
                inside=inside, nres=int(r["resampled"].sum()),
                distinct_end=int(r["distinct"][-1]),
                distinct_min=int(r["distinct"].min()))


def agg(lst):
    final = np.array([m["final"] for m in lst])
    ok = final < SEUIL_DIV
    rm = np.array([m["rmse_half"] for m in lst])
    return dict(
        div=float(100 * (1 - ok.mean())),
        rmse=float(np.median(rm[ok])) if ok.any() else float("nan"),
        rmse_z=float(np.median(np.array([m["rmse_z_half"] for m in lst])[ok])) if ok.any() else float("nan"),
        rmse_v=float(np.median(np.array([m["rmse_v_half"] for m in lst])[ok])) if ok.any() else float("nan"),
        nres=float(np.mean([m["nres"] for m in lst])),
        inside=[float(v) for v in 100 * np.mean([m["inside"] for m in lst], axis=0)],
        distinct_end=float(np.median([m["distinct_end"] for m in lst])),
        rmse_t=np.sqrt(np.mean(np.array([m["eh"] for m in lst]) ** 2, axis=0)),
        med_t=np.median(np.array([m["eh"] for m in lst]), axis=0),
    )


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    summary = {}

    # ------------------------------------------------------------------
    # 1. Réalisation de référence (réglages de l'énoncé)
    # ------------------------------------------------------------------
    snaps = (0, 30, 100, 400)
    r = rpf.run(seed=0, snaps=snaps)
    terr = rpf.terrain()
    Xt, Xe, Xstd = r["Xt"], r["Xe"], r["Xstd"]
    T = Xt.shape[1]
    temps = np.arange(T) * r["dt"]
    err = Xt - Xe
    eh = np.hypot(err[0], err[1])
    summary["ref"] = dict(
        h=float(r["h"]), hopt_gauss=float(rpf.h_opt(5000, 6, 2)),
        hopt_epan=float(rpf.h_opt(5000, 6, 1)), T=int(T),
        nres=int(r["resampled"].sum()), nres_100=int(r["resampled"][:100].sum()),
        centre_err=float(np.hypot(*(Xt[:2, 0] - Xe[:2, 0]))),
        final=[float(v) for v in err[:, -1]], final_h=float(eh[-1]),
        rmse_h_half=float(np.sqrt(np.mean(eh[T // 2:] ** 2))),
        rmse_half=[float(np.sqrt(np.mean(err[i, T // 2:] ** 2))) for i in range(6)],
        t_under_100=float(temps[np.argmax([np.all(eh[k:] < 100) for k in range(T)])]),
        std_end=[float(v) for v in Xstd[:, -1]],
        relief_min=float(np.nanmin(terr.hobs(Xt))), relief_max=float(np.nanmax(terr.hobs(Xt))),
    )

    # F1 : carte + nuages
    ext = terr.extent_km()
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.3))
    a = axs[0]
    im = a.imshow(terr.h_MNT, extent=ext, origin="upper", cmap="terrain")
    a.plot(Xt[0] / 1e3, Xt[1] / 1e3, "-k", lw=1.2, label="vraie")
    a.plot(Xe[0] / 1e3, Xe[1] / 1e3, "-r", lw=0.8, label="estimée")
    a.set_xlim(130, 215); a.set_ylim(215, 130)
    a.set_title("trajectoire"); a.set_xlabel("x (km)"); a.set_ylabel("y (km)")
    a.legend(loc="lower left", fontsize=7); a.grid(False)
    for a, k in zip(axs[1:], snaps):
        xy, w = r["clouds"][k]
        cx, cy = Xt[0, k] / 1e3, Xt[1, k] / 1e3
        half = max(1.0, 1.15 * np.max(np.abs(xy / 1e3 - np.array([[cx], [cy]]))))
        a.imshow(terr.h_MNT, extent=ext, origin="upper", cmap="terrain", alpha=0.75)
        a.scatter(xy[0] / 1e3, xy[1] / 1e3, s=1.5, c="m", alpha=0.35, rasterized=True, linewidths=0)
        a.plot(cx, cy, "k+", ms=10, mew=1.8)
        a.plot(Xe[0, k] / 1e3, Xe[1, k] / 1e3, "rx", ms=7, mew=1.5)
        a.set_xlim(cx - half, cx + half); a.set_ylim(cy + half, cy - half)
        a.set_title(f"t = {k} s"); a.set_xlabel("x (km)"); a.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f1_carte_nuages.pdf"), dpi=200)
    plt.close(fig)

    # F2 : mesures
    fig, a = plt.subplots(figsize=(6.2, 2.6))
    a.plot(temps, r["h_mes"], "r", lw=0.6, label="hauteur mesurée")
    a.plot(temps, r["h_vrai"], "b", lw=1.6, label="hauteur vraie")
    a.set_xlabel("temps (s)"); a.set_ylabel("hauteur sol (m)"); a.legend()
    fig.savefig(os.path.join(FIG, "f2_mesures.pdf")); plt.close(fig)

    # F3 : erreurs ± 3 sigma
    labels = ["x (m)", "y (m)", "z (m)", "$v_x$ (m/s)", "$v_y$ (m/s)", "$v_z$ (m/s)"]
    fig, axs = plt.subplots(2, 3, figsize=(10, 4.4), sharex=True)
    for i, a in enumerate(axs.ravel()):
        a.plot(temps, err[i], "b", lw=0.9, label="erreur")
        a.plot(temps, 3 * Xstd[i], "r", lw=0.7, label=r"$\pm 3\sigma$")
        a.plot(temps, -3 * Xstd[i], "r", lw=0.7)
        s = np.percentile(np.abs(np.r_[err[i, 40:], 3 * Xstd[i, 40:]]), 99)
        a.set_ylim(-1.3 * s, 1.3 * s)
        a.set_ylabel(labels[i])
    for a in axs[-1]:
        a.set_xlabel("temps (s)")
    axs[0, 0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f3_erreurs.pdf")); plt.close(fig)

    # F4 : Neff et instants de redistribution
    fig, a = plt.subplots(figsize=(6.2, 2.6))
    a.plot(temps, r["Neff"], "b", lw=0.8, label=r"$N_{\mathrm{eff}}$")
    a.axhline(r["coef_res"] * r["N"], color="k", ls="--", lw=0.8, label=r"$N_{\mathrm{th}}$")
    tr = temps[r["resampled"]]
    a.vlines(tr, 0, 250, color="r", lw=0.8, label="redistribution")
    a.set_ylim(0, 6300)
    a.set_xlabel("temps (s)"); a.set_ylabel("particules"); a.legend(loc="upper center", ncol=3, fontsize=7)
    fig.savefig(os.path.join(FIG, "f4_neff.pdf")); plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Études Monte-Carlo
    # ------------------------------------------------------------------
    configs = {}
    Ns = [500, 1000, 2000, 5000, 10000, 20000]
    for N in Ns:
        configs[f"N={N}"] = dict(N=N)
    coefs_h = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0]
    for k in (1, 2):
        for c in coefs_h:
            configs[f"h={c},K={k}"] = dict(coef_h=c, kernel=k)
    qs = [0.0, 0.01, 1.0, 100.0]
    for q in qs:
        for c in (0.0, 0.4):
            configs[f"q={q},h={c}"] = dict(q_scale=q, coef_h=c)
    crs = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    for c in crs:
        configs[f"res={c}"] = dict(coef_res=c)
    sigs = [5.0, 15.0, 30.0, 60.0, 100.0]
    for s in sigs:
        configs[f"sig={s}"] = dict(sigma_true=s)
    cRs = [0.5, 1.0, 1.5, 3.0]
    for c in cRs:
        configs[f"cR={c}"] = dict(coef_R=c)
    s0s = [1000.0, 3000.0, 6000.0, 10000.0]
    for s in s0s:
        configs[f"s0={s}"] = dict(sigma_xy0=s)

    jobs = [(tag, dict(kw, seed=1000 + s)) for tag, kw in configs.items() for s in range(NMC)]
    t0 = time.time()
    with Pool(9) as pool:
        out = pool.map(metrics, [j[1] for j in jobs], chunksize=4)
    print(f"{len(jobs)} réalisations en {time.time() - t0:.0f} s")
    res = {}
    for (tag, _), m in zip(jobs, out):
        res.setdefault(tag, []).append(m)
    A = {tag: agg(lst) for tag, lst in res.items()}

    # temps de calcul (séquentiel)
    tps = {}
    for N in Ns:
        t0 = time.perf_counter()
        for s in range(3):
            rpf.run(seed=s, N=N)
        tps[N] = (time.perf_counter() - t0) / 3

    for tag, a in A.items():
        summary[tag] = {k: v for k, v in a.items() if k not in ("rmse_t", "med_t")}
        summary[tag]["rmse_t_100"] = float(a["rmse_t"][100])
        summary[tag]["med_t_100"] = float(a["med_t"][100])
        summary[tag]["med_t_end"] = float(a["med_t"][-1])
    summary["temps_calcul"] = {str(k): float(v) for k, v in tps.items()}
    summary["NMC"] = NMC

    # F5 : nombre de particules
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.9))
    for N in Ns:
        axs[0].semilogy(temps, A[f"N={N}"]["med_t"], lw=0.9, label=f"N = {N}")
    axs[0].set_xlabel("temps (s)"); axs[0].set_ylabel("erreur horizontale médiane (m)")
    axs[0].legend(fontsize=6, ncol=2)
    axs[1].plot(Ns, [A[f"N={N}"]["div"] for N in Ns], "o-r")
    axs[1].set_xscale("log"); axs[1].set_xlabel("N"); axs[1].set_ylabel("taux de divergence (%)")
    axs[2].plot(Ns, [A[f"N={N}"]["rmse"] for N in Ns], "o-b", label="RMSE horizontale (m)")
    axs[2].set_xscale("log"); axs[2].set_xlabel("N"); axs[2].set_ylabel("RMSE horizontale (m)", color="b")
    a2 = axs[2].twinx(); a2.grid(False)
    a2.plot(Ns, [tps[N] for N in Ns], "s--k"); a2.set_ylabel("temps de calcul (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f5_nb_particules.pdf")); plt.close(fig)

    # F6 : coefficient de h_opt et noyau
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.9))
    for k, nom, st in ((2, "gaussien", "o-b"), (1, "Epanechnikov", "s--g")):
        axs[0].plot(coefs_h, [A[f"h={c},K={k}"]["div"] for c in coefs_h], st, label=nom)
        axs[1].plot(coefs_h, [A[f"h={c},K={k}"]["rmse"] for c in coefs_h], st, label=nom)
        axs[2].plot(coefs_h, [A[f"h={c},K={k}"]["inside"][0] for c in coefs_h], st, label=nom)
    axs[0].set_ylabel("taux de divergence (%)")
    axs[1].set_ylabel("RMSE horizontale (m)")
    axs[2].set_ylabel(r"erreur en $x$ dans $\pm3\sigma$ (%)")
    for a in axs:
        a.set_xlabel(r"coefficient de $h_{\mathrm{opt}}$"); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f6_regularisation.pdf")); plt.close(fig)

    # F7 : bruit d'état, SIR contre RPF
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.9))
    xs = np.arange(len(qs)); w = 0.38
    for j, (c, nom, col) in enumerate(((0.0, "SIR", "tab:orange"), (0.4, "RPF", "tab:blue"))):
        axs[0].bar(xs + (j - .5) * w, [A[f"q={q},h={c}"]["div"] for q in qs], w, label=nom, color=col)
        axs[1].bar(xs + (j - .5) * w, [A[f"q={q},h={c}"]["rmse"] for q in qs], w, label=nom, color=col)
        axs[2].bar(xs + (j - .5) * w, [A[f"q={q},h={c}"]["inside"][0] for q in qs], w, label=nom, color=col)
    axs[0].set_ylabel("taux de divergence (%)")
    axs[1].set_ylabel("RMSE horizontale (m)")
    axs[2].set_ylabel(r"erreur en $x$ dans $\pm3\sigma$ (%)")
    for a in axs:
        a.set_xticks(xs); a.set_xticklabels([f"{q:g}" for q in qs])
        a.set_xlabel(r"facteur sur $Q$")
    axs[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f7_bruit_etat.pdf")); plt.close(fig)

    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # impression lisible
    for tag in A:
        s = summary[tag]
        print(f"{tag:16s} div={s['div']:5.1f}%  rmse={s['rmse']:7.1f}  rmse_z={s['rmse_z']:6.1f} "
              f"rmse_v={s['rmse_v']:5.2f} nres={s['nres']:6.1f}  in3s={np.round(s['inside'], 0)} "
              f"dist={s['distinct_end']:.0f} med100={s['med_t_100']:.0f}")
    print("temps", tps)
    print(json.dumps(summary["ref"], indent=1))


if __name__ == "__main__":
    main()
