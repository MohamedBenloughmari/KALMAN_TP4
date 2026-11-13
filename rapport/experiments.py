"""
Figures et tableaux du rapport TP4.   Usage : python experiments.py
Les figures sont enregistrées dans figures/ et les tableaux affichés dans le terminal.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rpf

np.seterr(all="ignore")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)
NB_ESSAIS = int(os.environ.get("NB_ESSAIS", 20))   # nombre de tirages pour les tableaux
SEUIL_DIV = 500.0   # m : erreur horizontale finale au-delà de laquelle on dit que le filtre a divergé


def sauver(nom):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, nom), dpi=150)
    plt.close()


def erreur_horizontale(r):
    e = r["Xt"] - r["Xe"]
    return np.hypot(e[0], e[1])


# ----------------------------------------------------------------------
# 1. Scénario de référence (réglages de l'énoncé)
# ----------------------------------------------------------------------
instants = (0, 30, 100, 400)
r = rpf.run(seed=0, snaps=instants)
terr = rpf.terrain()
Xt, Xe, Xstd = r["Xt"], r["Xe"], r["Xstd"]
T = Xt.shape[1]
t = np.arange(T)
err = Xt - Xe
ext = terr.extent_km()

# carte et trajectoire
plt.figure(figsize=(6, 5))
plt.imshow(terr.h_MNT, extent=ext, origin="upper", cmap="terrain")
plt.colorbar(label="altitude du relief (m)")
plt.plot(Xt[0] / 1000, Xt[1] / 1000, "k", label="trajectoire vraie")
plt.plot(Xe[0] / 1000, Xe[1] / 1000, "r", label="trajectoire estimée")
plt.xlim(120, 220); plt.ylim(220, 120)
plt.xlabel("x (km)"); plt.ylabel("y (km)"); plt.legend()
sauver("carte.png")

# mesures
plt.figure(figsize=(7, 3.5))
plt.plot(t, r["h_mes"], "r", lw=0.8, label="altitude relative mesurée (m)")
plt.plot(t, r["h_vrai"], "b", lw=2, label="altitude relative vraie (m)")
plt.xlabel("temps (s)"); plt.grid(); plt.legend()
sauver("mesures.png")

# nuage de particules
plt.figure(figsize=(12, 3.4))
for i, k in enumerate(instants):
    xy, w = r["clouds"][k]
    plt.subplot(1, 4, i + 1)
    plt.plot(xy[0] / 1000, xy[1] / 1000, ".", color="gray", ms=1)
    plt.plot(Xt[0, k] / 1000, Xt[1, k] / 1000, "b+", ms=12, mew=2, label="vrai")
    plt.plot(Xe[0, k] / 1000, Xe[1, k] / 1000, "rx", ms=8, mew=2, label="estimé")
    plt.axis("equal"); plt.gca().invert_yaxis()
    plt.title(f"t = {k} s"); plt.xlabel("x (km)")
    if i == 0:
        plt.ylabel("y (km)"); plt.legend()
sauver("nuage.png")

# erreurs en position et en vitesse
plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.plot(t, err[0], "b", label="erreur en x (m)")
plt.plot(t, err[1], "r", label="erreur en y (m)")
plt.xlabel("temps (s)"); plt.grid(); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(t, err[3], "b", label="erreur en vx (m/s)")
plt.plot(t, err[4], "r", label="erreur en vy (m/s)")
plt.xlabel("temps (s)"); plt.grid(); plt.legend()
sauver("erreurs_xy.png")

# erreur en altitude et instants de redistribution
plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.plot(t, err[2], "b", label="erreur en z (m)")
plt.xlabel("temps (s)"); plt.grid(); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(t, r["resampled"], "b", lw=0.8, label="instants de redistribution")
plt.xlabel("temps (s)"); plt.legend(loc="upper right")
sauver("erreur_z_redistribution.png")

# erreurs en x et y avec l'incertitude à 3 sigma
plt.figure(figsize=(11, 4))
for i, (j, nom) in enumerate(((0, "x (m)"), (1, "y (m)"))):
    plt.subplot(1, 2, i + 1)
    plt.fill_between(t, -3 * Xstd[j], 3 * Xstd[j], color="lightgray", label=r"incertitude ($3\sigma$)")
    plt.plot(t, err[j], "b", label="erreur en " + nom)
    plt.xlabel("temps (s)"); plt.grid(); plt.legend()
sauver("erreurs_3sigma.png")

eh = erreur_horizontale(r)
print("Référence (graine 0)")
print("  h =", round(r["h"], 3), " nb redistributions =", r["resampled"].sum(),
      " dont avant 100 s :", r["resampled"][:100].sum())
print("  erreur initiale horizontale :", round(eh[0]), "m")
print("  erreurs finales :", np.round(err[:, -1], 2))
print("  erreur horizontale moyenne (2e moitié) :", round(eh[T // 2:].mean()), "m")
print("  3 sigma finaux x, y :", np.round(3 * Xstd[:2, -1]))


# ----------------------------------------------------------------------
# 2. Influence des paramètres
# ----------------------------------------------------------------------
def tableau(titre, nom_param, valeurs):
    """Pour chaque valeur du paramètre : NB_ESSAIS tirages, nombre de divergences
    et erreur horizontale moyenne (2e moitié du vol) sur les essais qui convergent."""
    print("\n" + titre)
    for v in valeurs:
        nb_div, erreurs, nb_red = 0, [], []
        for s in range(NB_ESSAIS):
            res = rpf.run(seed=100 + s, **{nom_param: v})
            e = erreur_horizontale(res)
            nb_red.append(res["resampled"].sum())
            if not e[-1] < SEUIL_DIV:
                nb_div += 1
            else:
                erreurs.append(e[T // 2:].mean())
        moy = f"{np.mean(erreurs):6.0f} m" if erreurs else "   -    "
        print(f"  {nom_param} = {v:<8} divergences : {nb_div:2d}/{NB_ESSAIS}   erreur : {moy}"
              f"   redistributions : {np.mean(nb_red):5.0f}")


def courbes(nom_param, valeurs, etiquettes, fichier, seed=0):
    plt.figure(figsize=(7, 4))
    for v, lab in zip(valeurs, etiquettes):
        res = rpf.run(seed=seed, **{nom_param: v})
        e = erreur_horizontale(res)
        print(f"  courbe {nom_param} = {v} : erreur finale {e[-1]:.0f} m")
        plt.semilogy(t, e, label=lab)
    plt.ylim(10, 5e4)
    plt.xlabel("temps (s)"); plt.ylabel("erreur horizontale (m)")
    plt.grid(which="both", alpha=0.4); plt.legend()
    sauver(fichier)


courbes("N", [500, 2000, 5000], ["N = 500", "N = 2000", "N = 5000"], "influence_N.png")
tableau("Nombre de particules", "N", [500, 1000, 2000, 5000, 10000])

courbes("coef_h", [0.0, 0.4, 1.0],
        ["sans régularisation (SIR)", "CoefHopt = 0.4", "CoefHopt = 1"], "influence_h.png")
tableau("Coefficient de h_opt (0 = SIR)", "coef_h", [0.0, 0.2, 0.4, 0.7, 1.0])

tableau("Noyau (1 = Epanechnikov, 2 = gaussien)", "kernel", [1, 2])
tableau("Seuil de redistribution", "coef_res", [0.2, 0.4, 0.6, 0.8, 1.0])
tableau("Ecart-type du bruit de mesure", "sigma_true", [5.0, 30.0, 100.0])

# ----------------------------------------------------------------------
# 3. Influence de la position de départ (relief survolé)
# ----------------------------------------------------------------------
departs = [(60, 280), (60, 60), (130, 240), (150, 150), (160, 150)]   # en km
couleurs = ["m", "darkorange", "b", "k", "r"]

# carte avec les 5 trajectoires
plt.figure(figsize=(6.5, 6.5))
plt.imshow(terr.h_MNT, extent=ext, origin="upper", cmap="terrain")
plt.colorbar(label="altitude du relief (m)", shrink=0.8)
reliefs = []
for i, (x0, y0) in enumerate(departs):
    res = rpf.run(seed=0, N=100, depart=(x0 * 1000, y0 * 1000))   # sert seulement à avoir la trajectoire
    reliefs.append(8000 - res["h_vrai"])
    plt.plot(res["Xt"][0] / 1000, res["Xt"][1] / 1000, color=couleurs[i], lw=2.5)
    plt.plot(x0, y0, "o", color=couleurs[i])
    plt.text(x0 - 13, y0 - 5, str(i + 1), fontsize=13, fontweight="bold")
plt.xlabel("x (km)"); plt.ylabel("y (km)")
sauver("depart_carte.png")

# relief survolé
plt.figure(figsize=(7, 4))
for i in range(len(departs)):
    plt.plot(t, reliefs[i], color=couleurs[i], label=f"départ {i + 1}")
plt.xlabel("temps (s)"); plt.ylabel("altitude du relief survolé (m)")
plt.grid(); plt.legend()
sauver("depart_relief.png")

# NB_ESSAIS tirages par départ
print("\nPosition de départ")
plt.figure(figsize=(7, 4))
for i, (x0, y0) in enumerate(departs):
    nb_div, erreurs, nb_red, toutes = 0, [], [], []
    for s in range(NB_ESSAIS):
        res = rpf.run(seed=100 + s, depart=(x0 * 1000, y0 * 1000))
        e = erreur_horizontale(res)
        toutes.append(e)
        nb_red.append(res["resampled"].sum())
        if not e[-1] < SEUIL_DIV:
            nb_div += 1
        else:
            erreurs.append(e[T // 2:].mean())
    mediane = np.nanmedian(np.array(toutes), axis=0)
    plt.semilogy(t, mediane, color=couleurs[i], label=f"départ {i + 1}")
    moy = f"{np.mean(erreurs):6.0f} m" if erreurs else "   -    "
    print(f"  départ {i + 1} ({x0}, {y0}) km : relief {reliefs[i].min():.0f}-{reliefs[i].max():.0f} m, "
          f"écart-type {reliefs[i].std():.0f} m | divergences {nb_div:2d}/{NB_ESSAIS} | erreur {moy} | "
          f"redistributions {np.mean(nb_red):.0f} | médiane à 100 s : {mediane[100]:.0f} m, "
          f"à 300 s : {mediane[300]:.0f} m, finale : {mediane[-1]:.0f} m")
plt.ylim(10, 5e4)
plt.xlabel("temps (s)"); plt.ylabel("erreur horizontale médiane (m)")
plt.grid(which="both", alpha=0.4); plt.legend()
sauver("depart_erreur.png")
