#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MODELLO MONODIMENSIONALE DI ACCRESCIMENTO DELLA BRINA SU SUPERFICIE FREDDA
Script di calcolo allegato alla tesi -- versione riproducibile
===============================================================================

CONTENUTO
---------
Lo script risolve il modello di frosting monodimensionale (crescita dello
spessore + densificazione interna + conduzione nello strato poroso) e produce
tutti i risultati numerici e i grafici riportati nella tesi:

  ANALISI A -- Caso base (Ta = -1 C, Tw = -15 C, Wa = 0.00339 kg/kg)
               Fig. 1  evoluzione dello spessore nel tempo
               Fig. 2  profilo di temperatura alla soglia h = 0.6 mm
               Fig. 3  distribuzione della densita' nello strato di brina
               Fig. 4  densita' media in funzione dello spessore
               Fig. 5  porosita' media in funzione dello spessore
               + stato alla soglia 0.6 mm salvato in formato .npz
                 (condizione iniziale per il modello di defrost)

  ANALISI B -- Sensibilita' all'umidita' assoluta dell'aria
               Fig. 6  spessore nel tempo per tre valori di Wa

  ANALISI C -- Verifica di indipendenza dalla discretizzazione
               Fig. 7  spessore finale e tempo di soglia al variare di
                       Ny (griglia spaziale) e di dt (passo temporale)

COME ESEGUIRE
-------------
Lo script e' autosufficiente: non richiede file di input, non legge dati
esterni e non contiene percorsi assoluti.

  1) da terminale, nella cartella che contiene questo file:

         python frost_model.py

  2) per visualizzare i grafici a schermo oltre a salvarli:

         python frost_model.py --show

  3) per scegliere la cartella dei risultati:

         python frost_model.py --outdir /percorso/di/destinazione

Lo script funziona anche se incollato/eseguito in una cella di Jupyter
Notebook o in un IDE (Spyder, PyCharm, IDLE, VS Code): in quel caso i
grafici vengono mostrati automaticamente.

RISULTATI PRODOTTI
------------------
Tutti i file vengono scritti nella sottocartella "risultati_frost", creata
automaticamente accanto a questo script:

    frost_1_spessore.png
    frost_2_temperatura_06mm.png
    frost_3_densita.png
    frost_4_densita_vs_h.png
    frost_5_porosita_vs_h.png
    frost_6_influenza_Wa.png
    frost_7_indipendenza_griglia_dt.png
    frost_state_06mm.npz          (stato alla soglia di defrost)
    riepilogo.txt                 (copia integrale dell'output a schermo)

REQUISITI
---------
Python >= 3.9 e le tre librerie scientifiche standard:

    python -m pip install numpy scipy matplotlib

Versioni usate per produrre i risultati della tesi:
    Python 3.14.7, numpy 2.5.2, scipy 1.18.1, matplotlib 3.11.1
Lo script e' compatibile anche con versioni precedenti
(numpy >= 1.20, scipy >= 1.6, matplotlib >= 3.4).

NOTE SULLA RIPRODUCIBILITA'
---------------------------
Il modello e' completamente deterministico: non utilizza numeri casuali,
ne' stati globali, ne' dati di ingresso esterni. A parita' di parametri
l'esecuzione restituisce sempre gli stessi valori numerici, su qualsiasi
sistema operativo (differenze ammesse solo all'ultima cifra significativa,
per effetto dell'aritmetica in virgola mobile).

Tempo di esecuzione complessivo: circa 30-60 s su un portatile standard.
===============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- Controllo delle dipendenze, con messaggio esplicito se mancanti --------
try:
    import numpy as np
    import scipy
    from scipy.linalg import solve_banded
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError as _err:                                 # pragma: no cover
    sys.exit(
        "\nLibreria mancante: {}\n"
        "Installare le dipendenze con il comando:\n\n"
        "    python -m pip install numpy scipy matplotlib\n".format(_err.name)
    )

# L'output a schermo contiene i simboli ° e ³: forziamo UTF-8 anche quando
# lo stdout viene rediretto su file.
try:
    sys.stdout.reconfigure(encoding="utf-8")                # Python >= 3.7
except (AttributeError, OSError):                           # pragma: no cover
    pass


# =============================================================================
# 1. PARAMETRI DEL MODELLO
#    I nomi dei simboli corrispondono a quelli usati nella tesi.
# =============================================================================

# --- Discretizzazione di riferimento ---------------------------------------
T_MAX = 4000.0        # [s]  durata simulata
NY_RIF = 41           # [-]  numero di nodi nello spessore della brina
NT_RIF = 4000         # [-]  numero di passi temporali  (dt = 1 s)

# --- Condizioni operative --------------------------------------------------
Ta = 272.15           # [K]  temperatura dell'aria umida          (-1 °C)
Tw = 258.15           # [K]  temperatura della parete fredda     (-15 °C)
W_a = 0.00339         # [kg/kg]  umidita' assoluta dell'aria      (UR ~ 98 %)

# --- Proprieta' dei materiali e costanti fisiche ---------------------------
rho_a = 1.290         # [kg/m3]   densita' dell'aria umida
rho_ice = 918.95      # [kg/m3]   densita' del ghiaccio compatto
P_tot = 101325.0      # [Pa]      pressione totale
Dv = 1.80e-5          # [m2/s]    diffusivita' del vapore in aria
lamda = 2.838e6       # [J/kg]    calore latente di sublimazione
Ka = 0.0225           # [W/(m K)] conducibilita' termica dell'aria
Ki = 2.390            # [W/(m K)] conducibilita' termica del ghiaccio
Cp_ice = 2000.0       # [J/(kg K)] calore specifico del ghiaccio
Cp_a = 1005.0         # [J/(kg K)] calore specifico dell'aria
a_cost = 1.576        # [-]  costante della correlazione di k_brina
b_cost = 0.797        # [-]  esponente della correlazione di k_brina
R_v = 461.5           # [J/(kg K)] costante del vapore d'acqua
R_d = 287.0           # [J/(kg K)] costante dell'aria secca

# --- Coefficienti di scambio e costanti di chiusura del modello ------------
Kw = 0.01             # [kg/(m2 s)] coeff. di scambio di massa (analogia di Lewis)
Le = 0.89             # [-]  numero di Lewis usato per ricavare Kw (informativo)
h_conv = 13.2         # [W/(m2 K)] coeff. convettivo (Blasius, regime laminare)
ALPHA = 0.2           # [-]  correzione del flusso diffusivo superficiale (Eq. 6)
BETA = 0.5            # [-]  correzione del termine sorgente di densificazione (Eq. 12)

# --- Condizioni iniziali ---------------------------------------------------
h_iniz = 2e-5         # [m]       spessore iniziale dello strato di brina
rho_fr_iniz = 30.0    # [kg/m3]   densita' iniziale della brina

# --- Soglia di innesco dello sbrinamento -----------------------------------
h_target_06 = 0.6e-3  # [m]  spessore che definisce il tempo di defrost

# --- Impostazioni delle analisi parametriche -------------------------------
SNAP_TIMES = (900.0, 1800.0, 2700.0, 3600.0)   # [s]  istanti dei profili di Fig. 3
SAMPLE_EVERY = 240                             # [-]  intervallo di campionamento Fig. 4-5
WA_LIST = (0.003438, 0.002307, 0.001357)       # [kg/kg]  valori di Wa per Fig. 6
NY_LIST = (11, 21, 41, 81, 161)                # [-]  griglie per lo studio di Fig. 7
NT_LIST = (1000, 2000, 4000, 8000, 16000)      # [-]  passi temporali per Fig. 7

# --- Tavolozza dei colori usata nelle figure della tesi --------------------
C_BLU = "#2980b9"
C_ROSSO = "#c0392b"
C_ARANCIO = "#e07b2a"
C_GRIGIO = "#95a5a6"
C_BLU_SCURO = "#3d6cc0"
C_VERDE = "#27ae60"


# =============================================================================
# 2. CORRELAZIONI COSTITUTIVE
# =============================================================================

def Pvs_C(T_C):
    """Pressione di saturazione del vapore [Pa] in funzione di T [°C]
    (correlazione di Magnus-Tetens)."""
    return 611.2 * np.exp(21.87 * T_C / (T_C + 265.5))


def calculation_porosity(rho_fr):
    """Porosita' della brina [-] dalla densita' locale [kg/m3]."""
    return (rho_ice - rho_fr) / (rho_ice - rho_a)


def calculation_tortuosity(eps):
    """Fattore di tortuosita' dei pori [-]."""
    return (1.0 / eps) * (1.0 - eps / 0.58 * (1.0 - eps))


def calculation_thermalcoducbility_fr(eps):
    """Conducibilita' termica efficace della brina [W/(m K)]."""
    return a_cost * Ki * (Ka / Ki) ** (b_cost * eps)


def calculation_specificheat_fr(eps, rho_fr):
    """Calore specifico efficace della brina [J/(kg K)]."""
    return (rho_ice * Cp_ice * (1.0 - eps) + rho_a * eps * Cp_a) / rho_fr


def calculation_effective_diffusiviti(tau, eps):
    """Diffusivita' efficace del vapore nella brina [m2/s]."""
    return tau * Dv * eps


def calculation_supersaturation(P_vs_fs_val, P_v_inf_l, P_vs_inf_l):
    """Grado di soprasaturazione S [-] alla superficie della brina."""
    return 0.808 * ((P_v_inf_l / P_vs_inf_l)
                    * (P_vs_fs_val / P_vs_inf_l) ** (-0.657)) - 1.0


def calculation_P_v_fs(S, P_vs_fs_val):
    """Pressione parziale del vapore alla superficie [Pa]."""
    return P_vs_fs_val * (1.0 + S)


def calculation_humidity_fs(P_v_fs):
    """Umidita' assoluta alla superficie della brina [kg/kg]."""
    return 0.622 * (P_v_fs / (P_tot - 0.378 * P_v_fs))


def umidita_relativa(W_a_loc):
    """Umidita' relativa [%] corrispondente a un'umidita' assoluta [kg/kg]."""
    P_v_loc = W_a_loc / (W_a_loc + 0.622) * P_tot
    return P_v_loc / Pvs_C(Ta - 273.15) * 100.0


# =============================================================================
# 3. SOLUTORE
#    Un'unica funzione, usata da tutte le analisi: cio' garantisce che il
#    caso base, lo studio di sensibilita' e lo studio di convergenza
#    risolvano esattamente le stesse equazioni.
# =============================================================================

def run_frost(Ny=NY_RIF, Nt=NT_RIF, T_max=T_MAX, W_a_loc=None,
              snap_times=SNAP_TIMES, sample_every=SAMPLE_EVERY,
              diagnostics=True):
    """Integra il modello di frosting su [0, T_max].

    Schema numerico
    ---------------
    * densificazione  : Eulero esplicito su flusso diffusivo di vapore,
                        diffusivita' valutata alle facce (media aritmetica);
    * spessore        : Eulero esplicito sul bilancio di massa superficiale;
    * temperatura     : schema a theta variabile (Crank-Nicolson, theta = 0.5;
                        Eulero implicito, theta = 1, nelle celle in cui il
                        numero di Fourier supera 0.4), sistema tridiagonale
                        risolto con scipy.linalg.solve_banded;
    * T di superficie : iterazione di punto fisso (5 iterazioni, sotto-
                        relazione 0.5) sul bilancio convezione-conduzione-
                        sublimazione.

    Parametri
    ---------
    Ny, Nt, T_max : discretizzazione.
    W_a_loc       : umidita' assoluta dell'aria; None usa il valore nominale.
    snap_times    : istanti [s] in cui memorizzare i profili interni.
    sample_every  : ogni quanti passi campionare densita' e porosita' medie.
    diagnostics   : se True raccoglie gli indicatori di qualita' numerica.

    Ritorna
    -------
    dict con storia dello spessore, profili alla soglia di 0.6 mm,
    snapshot, serie campionate, valori finali e diagnostiche numeriche.
    """
    W_a_l = W_a if W_a_loc is None else W_a_loc
    dt = T_max / Nt

    # Condizioni al contorno lato aria (costanti nel tempo)
    P_vs_inf_l = Pvs_C(Ta - 273.15)
    P_v_inf_l = W_a_l / (W_a_l + 0.622) * P_tot

    def _sup(P_vs_fs_val):
        return calculation_supersaturation(P_vs_fs_val, P_v_inf_l, P_vs_inf_l)

    # ---------------- stato iniziale ----------------
    rho_fr = np.full(Ny, rho_fr_iniz)
    eps_0 = calculation_porosity(rho_fr_iniz)
    k_0 = calculation_thermalcoducbility_fr(eps_0)
    T_fs_0 = (h_conv * Ta + (k_0 / h_iniz) * Tw) / (h_conv + k_0 / h_iniz)
    T_fr = np.linspace(Tw, float(np.clip(T_fs_0, Tw, Ta)), Ny)

    h_current = h_iniz
    h_hist = np.empty(Nt)
    t_hist = np.empty(Nt)

    # ---------------- strutture di uscita ----------------
    T_prof_06 = None        # profilo di T [°C] alla soglia 0.6 mm
    rho_prof_06 = None      # profilo di densita' [kg/m3] alla soglia
    h_06 = None             # spessore effettivo [mm] al passo di soglia
    t_06_raw = None         # primo istante [s] con h >= 0.6 mm
    t_06_interp = None      # istante [s] interpolato linearmente sulla soglia

    snap_rho, snap_T, snap_h = {}, {}, {}
    tempi = (np.arange(Nt) + 1) * dt
    snap_idx = {int(np.argmin(np.abs(tempi - s))): float(s) for s in snap_times}

    h_pts, rho_pts, eps_pts = [], [], []

    # diagnostiche numeriche
    theta_switch_count = 0
    theta_cells_total = 0
    n_clip_hits = 0

    A_banded = np.zeros((3, Ny))
    B_vec = np.zeros(Ny)

    # ---------------- avanzamento temporale ----------------
    for t in range(Nt):
        t_now = (t + 1) * dt
        dy = h_current / (Ny - 1)

        # (a) proprieta' locali della miscela aria-vapore
        P_v_local = Pvs_C(T_fr - 273.15)
        rho_v_local = P_v_local / (R_v * T_fr)
        rho_a_local = rho_v_local + (P_tot - P_v_local) / (R_d * T_fr)

        pvs_wall = Pvs_C(Tw - 273.15)                  # nodo a parete
        rho_v_local[0] = pvs_wall / (R_v * Tw)
        rho_a_local[0] = rho_v_local[0] + (P_tot - pvs_wall) / (R_d * Tw)

        pvs_top = Pvs_C(T_fr[-1] - 273.15)             # nodo superficiale
        rho_v_local[-1] = pvs_top / (R_v * T_fr[-1])
        rho_a_local[-1] = rho_v_local[-1] + (P_tot - pvs_top) / (R_d * T_fr[-1])

        phi = rho_v_local / rho_a_local                # frazione di massa di vapore

        eps = calculation_porosity(rho_fr)
        tau = calculation_tortuosity(eps)
        D_eff = calculation_effective_diffusiviti(tau, eps)

        # (b) densificazione interna (Eulero esplicito, forma conservativa)
        DR = D_eff * rho_a_local
        D_faccia = 0.5 * (DR[:-1] + DR[1:])
        flusso = D_faccia * np.diff(phi)

        rho_fr_new = np.copy(rho_fr)
        rho_fr_new[1:-1] = (rho_fr[1:-1]
                            + dt * BETA * (flusso[1:] - flusso[:-1]) / dy ** 2)
        rho_fr_new[0] = (rho_fr[0]
                         + dt * BETA * 2.0 * D_faccia[0] * (phi[1] - phi[0]) / dy ** 2)

        if diagnostics:
            pre_clip = rho_fr_new.copy()

        rho_fr_new = np.clip(rho_fr_new, rho_fr_iniz, 0.92 * rho_ice)
        if diagnostics and np.any(np.abs(pre_clip - rho_fr_new) > 1e-9):
            n_clip_hits += 1
        rho_fr_new[-1] = rho_fr_new[-2]                # superficie a gradiente nullo

        # (c) crescita dello spessore (bilancio di massa superficiale)
        P_vs_s = Pvs_C(T_fr[-1] - 273.15)
        P_v_s = calculation_P_v_fs(_sup(P_vs_s), P_vs_s)
        W_fs = calculation_humidity_fs(P_v_s)
        mv_fs = Kw * (W_a_l - W_fs)                                 # flusso convettivo
        grad_phi_s = (3.0 * phi[-1] - 4.0 * phi[-2] + phi[-3]) / (2.0 * dy)
        mv_diff_s = D_eff[-1] * rho_a * grad_phi_s                  # quota diffusiva

        h_prev = h_current
        dh_dt = max(0.0, (mv_fs - ALPHA * mv_diff_s) / rho_fr_new[-1])
        h_current += dh_dt * dt
        h_hist[t] = h_current
        t_hist[t] = t_now

        # (d) temperatura superficiale (punto fisso, 5 iterazioni)
        eps_cn = calculation_porosity(rho_fr_new)
        k_fr = calculation_thermalcoducbility_fr(eps_cn)
        cp_fr = calculation_specificheat_fr(eps_cn, rho_fr_new)
        cond_r = float(np.mean(k_fr)) / max(h_current, 1e-9)

        T_fs = float(np.clip(T_fr[-1], Tw, Ta))
        for _ in range(5):
            Pvs_t = Pvs_C(T_fs - 273.15)
            P_v_t = calculation_P_v_fs(_sup(Pvs_t), Pvs_t)
            W_t = calculation_humidity_fs(P_v_t)
            mv_t = max(Kw * (W_a_l - W_t), 0.0)
            T_new = (h_conv * Ta + cond_r * Tw + lamda * mv_t) / (h_conv + cond_r)
            T_fs = float(np.clip(0.5 * T_fs + 0.5 * T_new, Tw, Ta))

        # (e) conduzione nello strato (schema a theta variabile, tridiagonale)
        k_mid = 0.5 * (k_fr[1:] + k_fr[:-1])
        km = k_mid[:-1]
        kp = k_mid[1:]
        rCp = rho_fr_new[1:-1] * cp_fr[1:-1]

        Fo = dt * 0.5 * (km + kp) / (rCp * dy ** 2)
        theta = np.where(Fo > 0.4, 1.0, 0.5)
        if diagnostics:
            theta_switch_count += int(np.count_nonzero(Fo > 0.4))
            theta_cells_total += theta.size

        r_im = dt * km / (rCp * dy ** 2)
        r_ip = dt * kp / (rCp * dy ** 2)

        A_banded[:] = 0.0
        B_vec[:] = 0.0
        A_banded[0, 2:] = -theta * r_ip
        A_banded[1, 1:-1] = 1.0 + theta * (r_im + r_ip)
        A_banded[2, :-2] = -theta * r_im
        B_vec[1:-1] = (T_fr[1:-1]
                       + (1.0 - theta) * r_im * (T_fr[:-2] - T_fr[1:-1])
                       + (1.0 - theta) * r_ip * (T_fr[2:] - T_fr[1:-1])
                       + lamda * (rho_fr_new[1:-1] - rho_fr[1:-1]) / rCp)
        A_banded[1, 0] = 1.0
        B_vec[0] = Tw                                  # Dirichlet a parete
        A_banded[1, -1] = 1.0
        B_vec[-1] = T_fs                               # Dirichlet in superficie

        T_fr = solve_banded((1, 1), A_banded, B_vec)
        rho_fr = rho_fr_new

        # (f) cattura dello stato alla soglia di defrost (una sola volta)
        if T_prof_06 is None and h_current >= h_target_06:
            T_prof_06 = T_fr.copy() - 273.15
            rho_prof_06 = rho_fr.copy()
            h_06 = h_current * 1000.0
            t_06_raw = t_now
            if h_current > h_prev:
                frazione = (h_target_06 - h_prev) / (h_current - h_prev)
                t_06_interp = (t_now - dt) + frazione * dt
            else:
                t_06_interp = t_now

        # (g) profili interni agli istanti richiesti
        if t in snap_idx:
            chiave = snap_idx[t]
            snap_rho[chiave] = rho_fr.copy()
            snap_T[chiave] = T_fr.copy() - 273.15
            snap_h[chiave] = h_current * 1000.0

        # (h) campionamento per le curve densita'/porosita' vs spessore
        if sample_every and t % sample_every == 0:
            h_pts.append(h_current * 1e3)
            rho_pts.append(float(np.mean(rho_fr)))
            eps_pts.append(float(np.mean(eps_cn)))

    eps_final = calculation_porosity(rho_fr)
    return {
        "Ny": Ny, "Nt": Nt, "dt": dt, "T_max": T_max, "W_a": W_a_l,
        "t_hist_s": t_hist,
        "h_hist_mm": h_hist * 1000.0,
        "t_06_s": t_06_raw,
        "t_06_interp_s": t_06_interp,
        "h_06_mm": h_06,
        "T_prof_06_C": T_prof_06,
        "rho_prof_06": rho_prof_06,
        "snap_rho": snap_rho, "snap_T": snap_T, "snap_h": snap_h,
        "h_pts_mm": np.array(h_pts),
        "rho_pts": np.array(rho_pts),
        "eps_pts": np.array(eps_pts),
        "h_final_mm": h_current * 1000.0,
        "rho_mean_final": float(np.mean(rho_fr)),
        "eps_mean_final": float(np.mean(eps_final)),
        "theta_switch_frac": (theta_switch_count / theta_cells_total
                              if theta_cells_total else 0.0),
        "n_clip_hits": n_clip_hits,
    }


# =============================================================================
# 4. UTILITA' DI OUTPUT (log a schermo + su file, salvataggio figure)
# =============================================================================

_LOG = []


def log(msg=""):
    """Stampa a schermo e memorizza la riga per il file riepilogo.txt."""
    testo = str(msg)
    print(testo)
    _LOG.append(testo)


def salva_figura(fig, nome, outdir, show, dpi=150):
    """Salva la figura in outdir, la mostra se richiesto e la chiude."""
    percorso = Path(outdir) / nome
    fig.savefig(percorso, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return percorso


def in_notebook():
    """True se lo script sta girando in Jupyter/IPython."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
    except Exception:
        return False
    return shell is not None and shell.__class__.__name__ != "TerminalInteractiveShell"


# =============================================================================
# 5. ANALISI A -- CASO BASE E FIGURE 1-5
# =============================================================================

def analisi_caso_base(outdir, show):
    """Esegue il caso base, produce le Figure 1-5, salva lo stato di defrost
    e stampa il riepilogo numerico. Ritorna il dizionario dei risultati."""
    log("=" * 74)
    log("ANALISI A -- CASO BASE")
    log("=" * 74)
    log("Condizioni: Ta = {:.0f} °C, Tw = {:.0f} °C, Wa = {:g} kg/kg "
        "(UR ~ {:.0f} %)".format(Ta - 273.15, Tw - 273.15, W_a,
                                 umidita_relativa(W_a)))
    log("Discretizzazione: Ny = {}, Nt = {}, dt = {:g} s, T_max = {:g} s"
        .format(NY_RIF, NT_RIF, T_MAX / NT_RIF, T_MAX))
    log("Costanti di chiusura: alpha = {:g}, beta = {:g}, Kw = {:g} kg/(m2 s), "
        "h_conv = {:g} W/(m2 K)".format(ALPHA, BETA, Kw, h_conv))
    log("Calcolo in corso...")

    ris = run_frost()

    time_arr = ris["t_hist_s"] / 60.0        # [min]
    h_arr = ris["h_hist_mm"]                 # [mm]
    t_06 = ris["t_06_s"]
    h_06 = ris["h_06_mm"]
    T_prof_06 = ris["T_prof_06_C"]

    if t_06 is not None:
        log("Soglia di 0.6 mm raggiunta a t = {:.2f} min (h = {:.4f} mm)"
            .format(t_06 / 60.0, h_06))
    else:
        log("ATTENZIONE: la soglia di 0.6 mm non e' stata raggiunta entro T_max.")

    # -------------------------------------------------------------------
    # FIGURA 1 -- evoluzione dello spessore nel tempo
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(time_arr, h_arr, color=C_BLU, lw=2.2,
            label="Variation of thickness over time")
    ax.plot([0, time_arr[-1]], [0, h_arr[-1]], "k:", lw=1, alpha=0.4,
            label="Reference line")

    marker_times = [10, 20, 30, 40, 50, 60]
    marker_h = [h_arr[int(np.argmin(np.abs(time_arr - tp)))] for tp in marker_times]
    ax.plot(marker_times, marker_h, "o", color=C_BLU, ms=9,
            markerfacecolor="white", markeredgewidth=2, zorder=4,
            label="Thickness values in the range 10-60 min")

    ax.axhline(y=0.6, color=C_ROSSO, linestyle=":", lw=1.6, alpha=0.7)
    if t_06 is not None:
        ax.plot([t_06 / 60.0], [h_06], "D", color=C_ROSSO, ms=10, zorder=5,
                label="Defrost threshold 0.6 mm (t = {:.1f} min)".format(t_06 / 60.0))
        ax.annotate("t_defrost = {:.1f} min".format(t_06 / 60.0),
                    xy=(t_06 / 60.0, 0.6),
                    xytext=(t_06 / 60.0 - 14, 0.72),
                    fontsize=10, color=C_ROSSO, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C_ROSSO, lw=1.2))

    ax.set_xlabel("Time [min]", fontsize=12)
    ax.set_ylabel("Thickness [mm]", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, time_arr[-1])
    ax.set_ylim(0, h_arr.max() * 1.1)
    fig.tight_layout()
    salva_figura(fig, "frost_1_spessore.png", outdir, show)

    # -------------------------------------------------------------------
    # FIGURA 2 -- profilo di temperatura alla soglia h = 0.6 mm
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.5))
    if T_prof_06 is not None:
        y_mm = np.linspace(0, h_06, ris["Ny"])
        ax.plot(T_prof_06, y_mm, color=C_ROSSO, lw=2.3,
                label="Thickness = {:.3f} mm  (t = {:.1f} min)"
                      .format(h_06, t_06 / 60.0))
        ax.axvline(Tw - 273.15, color="black", ls="--", lw=1, alpha=0.5,
                   label="Temperature wall = {:.0f}°C".format(Tw - 273.15))
        ax.axvline(-12, color="black", ls=":", lw=1, alpha=0.5)
        ax.set_ylabel("Frost position [mm]", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
    ax.set_xlabel("Temperature [°C]", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    salva_figura(fig, "frost_2_temperatura_06mm.png", outdir, show)

    # -------------------------------------------------------------------
    # FIGURA 3 -- distribuzione della densita' (con zoom sulla zona di parete)
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    snap_colors = [C_GRIGIO, C_BLU_SCURO, C_ARANCIO, C_ROSSO]
    snap_labels = ["15 min", "30 min", "45 min", "60 min"]
    yn = np.linspace(0, 1, ris["Ny"])

    x_zoom = (0, 200)        # [kg/m3]
    y_zoom = (0, 0.2)        # profilo adimensionale
    axins = ax.inset_axes([0.42, 0.20, 0.53, 0.48])

    for ts, col, lbl in zip(SNAP_TIMES, snap_colors, snap_labels):
        if ts not in ris["snap_rho"]:
            continue
        rho_prof = ris["snap_rho"][ts]
        ax.plot(rho_prof, yn, color=col, lw=2,
                label="{}  (h={:.2f} mm, avg={:.0f} kg/m³)"
                      .format(lbl, ris["snap_h"][ts], float(np.mean(rho_prof))))
        axins.plot(rho_prof, yn, "o-", color=col, lw=1.8, ms=4,
                   markerfacecolor="white", markeredgewidth=1.3)

    axins.set_xlim(*x_zoom)
    axins.set_ylim(*y_zoom)
    axins.set_yticks(np.arange(0, 0.201, 0.05))
    axins.tick_params(labelsize=9)
    axins.grid(True, alpha=0.3)
    axins.set_facecolor("white")
    ax.indicate_inset_zoom(axins, edgecolor="0.35", alpha=0.8)

    ax.set_xlabel("Local density  [kg/m³]", fontsize=14)
    ax.set_ylabel("Standardized profile\n(0 = wall,  1 = surface)", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    salva_figura(fig, "frost_3_densita.png", outdir, show)

    # -------------------------------------------------------------------
    # FIGURA 4 -- densita' media in funzione dello spessore
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.plot(ris["h_pts_mm"], ris["rho_pts"], "o-", color=C_ROSSO, lw=2, ms=7,
            markerfacecolor="white", markeredgewidth=2)
    ax.set_xlabel("Frost thickness [mm]", fontsize=16)
    ax.set_ylabel("Average density [kg/m³]", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=20)
    fig.tight_layout()
    salva_figura(fig, "frost_4_densita_vs_h.png", outdir, show)

    # -------------------------------------------------------------------
    # FIGURA 5 -- porosita' media in funzione dello spessore
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.plot(ris["h_pts_mm"], ris["eps_pts"], "s-", color=C_BLU, lw=2, ms=7,
            markerfacecolor="white", markeredgewidth=2)
    ax.set_xlabel("Frost thickness [mm]", fontsize=16)
    ax.set_ylabel("Average porosity [-]", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0.90, 1.0)
    fig.tight_layout()
    salva_figura(fig, "frost_5_porosita_vs_h.png", outdir, show)

    # -------------------------------------------------------------------
    # Analisi della concavita' della curva h(t) e riepilogo numerico
    # -------------------------------------------------------------------
    m_i = (time_arr >= 5.0) & (time_arr <= 10.0)
    p_i = np.polyfit(time_arr[m_i], h_arr[m_i], 1)
    t_fin = min(time_arr[-1], 60.0)
    m_f = (time_arr >= t_fin - 10.0) & (time_arr <= t_fin)
    p_f = np.polyfit(time_arr[m_f], h_arr[m_f], 1)
    non_lin = (1.0 - p_f[0] / p_i[0]) * 100.0

    log("")
    log("-" * 74)
    log("RISULTATI DEL CASO BASE")
    log("-" * 74)
    if t_06 is not None:
        log("Tempo di defrost (h = 0.6 mm)      : {:8.1f} min".format(t_06 / 60.0))
        log("Temperatura di superficie a soglia : {:8.2f} °C".format(T_prof_06[-1]))
    log("Spessore finale a {:.0f} min          : {:8.3f} mm"
        .format(time_arr[-1], ris["h_final_mm"]))
    log("Densita' media finale              : {:8.1f} kg/m³".format(ris["rho_pts"][-1]))
    log("Porosita' media finale             : {:8.4f} -".format(ris["eps_pts"][-1]))
    log("Pendenza iniziale (5-10 min)       : {:8.5f} mm/min".format(p_i[0]))
    log("Pendenza finale (ultimi 10 min)    : {:8.5f} mm/min".format(p_f[0]))
    log("Concavita' (non linearita')        : {:+8.1f} %".format(non_lin))
    log("")
    log("Diagnostiche numeriche:")
    log("  celle con Fo > 0.4 (Eulero implicito): {:.2f} %"
        .format(100.0 * ris["theta_switch_frac"]))
    log("  passi con vincolo di densita' attivo : {} / {}"
        .format(ris["n_clip_hits"], ris["Nt"]))

    # -------------------------------------------------------------------
    # Stato alla soglia 0.6 mm: condizione iniziale del modello di defrost
    # -------------------------------------------------------------------
    if ris["rho_prof_06"] is not None:
        percorso_npz = Path(outdir) / "frost_state_06mm.npz"
        np.savez(percorso_npz,
                 T_fr_K=T_prof_06 + 273.15,          # [K]      profilo di temperatura
                 rho_fr=ris["rho_prof_06"],          # [kg/m3]  profilo di densita'
                 eps=calculation_porosity(ris["rho_prof_06"]),   # [-]  porosita'
                 h_0=h_06 * 1e-3,                    # [m]      spessore alla soglia
                 t_frost=t_06,                       # [s]      tempo di accrescimento
                 Ta=Ta, Tw=Tw)                       # [K]      condizioni operative
        log("")
        log("Stato alla soglia salvato in: {}".format(percorso_npz.name))
    else:
        log("")
        log("Soglia non raggiunta: frost_state_06mm.npz non e' stato creato.")
        log("Aumentare T_MAX per simulare un intervallo piu' lungo.")

    return ris


# =============================================================================
# 6. ANALISI B -- SENSIBILITA' ALL'UMIDITA' ASSOLUTA (FIGURA 6)
# =============================================================================

def analisi_sensibilita_Wa(outdir, show):
    """Ripete il calcolo per tre valori di Wa (Ta e Tw invariati) e produce
    la Figura 6."""
    log("")
    log("=" * 74)
    log("ANALISI B -- SENSIBILITA' ALL'UMIDITA' ASSOLUTA DELL'ARIA")
    log("=" * 74)
    log("{:>10} | {:>7} | {:>14} | {:>14}"
        .format("Wa [kg/kg]", "UR [%]", "t(0.6 mm) [min]", "h finale [mm]"))
    log("-" * 56)

    risultati = []
    for W in WA_LIST:
        r = run_frost(W_a_loc=W, snap_times=(), sample_every=0, diagnostics=False)
        risultati.append(r)
        t6 = r["t_06_s"]
        log("{:>10.6f} | {:>7.1f} | {:>14} | {:>14.3f}"
            .format(W, umidita_relativa(W),
                    "{:.1f}".format(t6 / 60.0) if t6 else "non raggiunta",
                    r["h_final_mm"]))

    colori = [C_BLU, C_ARANCIO, C_ROSSO]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r, col in zip(risultati, colori):
        t_min = r["t_hist_s"] / 60.0
        ax.plot(t_min, r["h_hist_mm"], color=col, lw=2.2,
                label="Wa = {:.6f}  (UR ≈ {:.0f}%)"
                      .format(r["W_a"], umidita_relativa(r["W_a"])))
        if r["t_06_s"] is not None:
            ax.plot([r["t_06_s"] / 60.0], [r["h_06_mm"]], "D", color=col, ms=8,
                    zorder=5)
            ax.annotate("{:.1f} min".format(r["t_06_s"] / 60.0),
                        xy=(r["t_06_s"] / 60.0, r["h_06_mm"]),
                        xytext=(6, -14), textcoords="offset points",
                        fontsize=9, color=col, fontweight="bold")

    ax.axhline(y=0.6, color="black", linestyle=":", lw=1.5, alpha=0.6)
    ax.set_xlabel("Time [min]", fontsize=12)
    ax.set_ylabel("Frost thickness [mm]", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, risultati[0]["t_hist_s"][-1] / 60.0)
    ax.set_ylim(0, risultati[0]["h_hist_mm"].max() * 1.1)
    fig.tight_layout()
    salva_figura(fig, "frost_6_influenza_Wa.png", outdir, show)

    return risultati


# =============================================================================
# 7. ANALISI C -- INDIPENDENZA DA GRIGLIA E PASSO TEMPORALE (FIGURA 7)
# =============================================================================

def analisi_indipendenza(outdir, show):
    """Studio di convergenza: raffinamento della griglia spaziale a dt
    costante e riduzione del passo temporale a Ny costante. Produce la
    Figura 7 e le tabelle degli scarti."""
    comune = dict(snap_times=(), sample_every=0)

    # ---- C.1 griglia spaziale (dt = 1 s costante) ----
    log("")
    log("=" * 74)
    log("ANALISI C.1 -- GRIGLIA SPAZIALE (dt = {:g} s costante)".format(T_MAX / NT_RIF))
    log("=" * 74)
    log("{:>6} | {:>13} | {:>13} | {:>12} | {:>13} | {:>11}"
        .format("Ny", "dy@0.6mm [um]", "h finale [mm]", "t06 [min]",
                "rho parete", "clip [passi]"))
    log("-" * 84)

    gh, gt, g_clip = [], [], []
    for n in NY_LIST:
        r = run_frost(Ny=n, Nt=NT_RIF, T_max=T_MAX, **comune)
        gh.append(r["h_final_mm"])
        gt.append(r["t_06_interp_s"] / 60.0)
        g_clip.append(r["n_clip_hits"])
        log("{:>6d} | {:>13.2f} | {:>13.6f} | {:>12.4f} | {:>13.2f} | {:>5d}/{:<5d}"
            .format(n, h_target_06 / (n - 1) * 1e6, r["h_final_mm"],
                    r["t_06_interp_s"] / 60.0, r["rho_prof_06"][0],
                    r["n_clip_hits"], NT_RIF))

    gh = np.array(gh)
    gt = np.array(gt)

    # ---- C.2 passo temporale (Ny = 41 costante) ----
    log("")
    log("=" * 74)
    log("ANALISI C.2 -- PASSO TEMPORALE (Ny = {} costante)".format(NY_RIF))
    log("=" * 74)
    log("{:>8} | {:>7} | {:>14} | {:>12}"
        .format("dt [s]", "Nt", "h finale [mm]", "t06 [min]"))
    log("-" * 50)

    th, tt = [], []
    for n in NT_LIST:
        r = run_frost(Ny=NY_RIF, Nt=n, T_max=T_MAX, **comune)
        th.append(r["h_final_mm"])
        tt.append(r["t_06_interp_s"] / 60.0)
        log("{:>8.3f} | {:>7d} | {:>14.7f} | {:>12.5f}"
            .format(T_MAX / n, n, r["h_final_mm"], r["t_06_interp_s"] / 60.0))

    th = np.array(th)
    tt = np.array(tt)
    dts = T_MAX / np.array(NT_LIST, dtype=float)

    # ---- C.3 tabelle degli scarti rispetto al caso piu' raffinato ----
    log("")
    log("=" * 74)
    log("SCARTI RISPETTO AL CASO PIU' RAFFINATO")
    log("=" * 74)

    log("")
    log("  Griglia spaziale (riferimento Ny = {}):".format(NY_LIST[-1]))
    log("  {:>6} | {:>13} | {:>9} | {:>12} | {:>9}"
        .format("Ny", "h [mm]", "scarto", "t06 [min]", "scarto"))
    log("  " + "-" * 60)
    for i, n in enumerate(NY_LIST):
        log("  {:>6d} | {:>13.6f} | {:>8.3f}% | {:>12.4f} | {:>8.3f}%"
            .format(n, gh[i], 100.0 * abs(gh[i] - gh[-1]) / gh[-1],
                    gt[i], 100.0 * abs(gt[i] - gt[-1]) / gt[-1]))

    log("")
    log("  Passo temporale (riferimento dt = {:g} s):".format(dts[-1]))
    log("  {:>8} | {:>13} | {:>9} | {:>12} | {:>9}"
        .format("dt [s]", "h [mm]", "scarto", "t06 [min]", "scarto"))
    log("  " + "-" * 62)
    for i, n in enumerate(NT_LIST):
        log("  {:>8.3f} | {:>13.6f} | {:>8.4f}% | {:>12.5f} | {:>8.4f}%"
            .format(T_MAX / n, th[i], 100.0 * abs(th[i] - th[-1]) / th[-1],
                    tt[i], 100.0 * abs(tt[i] - tt[-1]) / tt[-1]))

    # ---- FIGURA 7 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    # (a) griglia spaziale
    ax = axes[0]
    ax.plot(NY_LIST, gh, "o-", color=C_BLU, lw=2.2, ms=10,
            markerfacecolor="white", markeredgewidth=2.2, label="Final thickness")
    ax.set_xlabel("Cell number  $N_y$", fontsize=12)
    ax.set_ylabel("Final thickness [mm]", fontsize=12, color=C_BLU)
    ax.tick_params(axis="y", labelcolor=C_BLU)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(NY_LIST))
    ax.set_xticklabels([str(n) for n in NY_LIST])
    rif = gh[-1]
    ax.axhspan(rif * 0.99, rif * 1.01, color=C_BLU, alpha=0.10)
    ax.axhline(rif, color=C_BLU, ls=":", lw=1.2, alpha=0.7)
    ax.axvline(NY_RIF, color=C_VERDE, ls="-", lw=2.0, alpha=0.35)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(NY_LIST, gt, "s--", color=C_ROSSO, lw=2.0, ms=9,
             markerfacecolor="white", markeredgewidth=2.2, label="Defrost time")
    ax2.set_ylabel("Threshold time [min]", fontsize=12, color=C_ROSSO)
    ax2.tick_params(axis="y", labelcolor=C_ROSSO)
    ax2.ticklabel_format(axis="y", useOffset=False, style="plain")

    linee = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(linee, [l.get_label() for l in linee], fontsize=10, loc="upper right")

    # (b) passo temporale
    ax = axes[1]
    ax.plot(dts, th, "o-", color=C_BLU, lw=2.2, ms=10,
            markerfacecolor="white", markeredgewidth=2.2)
    ax.set_xlabel("Time step  $\\Delta t$  [s]", fontsize=12)
    ax.set_ylabel("Final thickness [mm]", fontsize=12, color=C_BLU)
    ax.tick_params(axis="y", labelcolor=C_BLU)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dts)
    ax.set_xticklabels(["{:g}".format(x) for x in dts])
    ax.invert_xaxis()
    rif = th[-1]
    ax.axhspan(rif * 0.99, rif * 1.01, color=C_BLU, alpha=0.10)
    ax.axhline(rif, color=C_BLU, ls=":", lw=1.2, alpha=0.7)
    ax.axvline(T_MAX / NT_RIF, color=C_VERDE, ls="-", lw=2.0, alpha=0.35)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(dts, tt, "s--", color=C_ROSSO, lw=2.0, ms=9,
             markerfacecolor="white", markeredgewidth=2.2)
    ax2.set_ylabel("Threshold time 0.6 mm  [min]", fontsize=12, color=C_ROSSO)
    ax2.tick_params(axis="y", labelcolor=C_ROSSO)
    ax2.ticklabel_format(axis="y", useOffset=False, style="plain")

    fig.suptitle("Verifica di indipendenza dalla discretizzazione — modello di frosting\n"
                 "banda ombreggiata: tolleranza $\\pm$1% sul valore piu' raffinato",
                 fontsize=12.5, fontweight="bold", y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    salva_figura(fig, "frost_7_indipendenza_griglia_dt.png", outdir, show, dpi=200)

    # ---- Avviso sul limite di validita' del raffinamento ----
    if any(c > 0 for c in g_clip):
        log("")
        log("  NOTA: per le griglie piu' fini si attiva il vincolo di densita'")
        log("  massima (0.92 * rho_ghiaccio). Oltre Ny ~ 100 tale vincolo diventa")
        log("  dominante e i risultati perdono significato fisico: il raffinamento")
        log("  non va spinto oltre.")
        for n, c in zip(NY_LIST, g_clip):
            if c > 0:
                log("    Ny = {:>3d}: vincolo attivo per {} passi ({:.1f} %)"
                    .format(n, c, 100.0 * c / NT_RIF))


# =============================================================================
# 8. PROGRAMMA PRINCIPALE
# =============================================================================

def cartella_predefinita():
    """Sottocartella 'risultati_frost' accanto allo script (o nella cartella
    di lavoro corrente se __file__ non e' definito, es. in Jupyter)."""
    try:
        base = Path(__file__).resolve().parent
    except NameError:                                        # pragma: no cover
        base = Path.cwd()
    return base / "risultati_frost"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Modello di accrescimento della brina: calcolo completo "
                    "e generazione di tutte le figure della tesi.")
    parser.add_argument("--outdir", default=None, metavar="CARTELLA",
                        help="cartella di destinazione dei risultati "
                             "(predefinita: ./risultati_frost)")
    parser.add_argument("--show", action="store_true",
                        help="mostra i grafici a schermo oltre a salvarli")
    # parse_known_args ignora gli argomenti aggiunti da Jupyter/IDE.
    args, _ignoti = parser.parse_known_args(argv)

    outdir = Path(args.outdir) if args.outdir else cartella_predefinita()
    outdir.mkdir(parents=True, exist_ok=True)
    show = args.show or in_notebook()

    log("=" * 74)
    log("MODELLO DI ACCRESCIMENTO DELLA BRINA -- ESECUZIONE COMPLETA")
    log("=" * 74)
    log("Python      : {}".format(sys.version.split()[0]))
    log("numpy       : {}".format(np.__version__))
    log("scipy       : {}".format(scipy.__version__))
    log("matplotlib  : {}".format(matplotlib.__version__))
    log("Risultati in: {}".format(outdir.resolve()))
    log("")

    analisi_caso_base(outdir, show)
    analisi_sensibilita_Wa(outdir, show)
    analisi_indipendenza(outdir, show)

    # ---- elenco finale dei file prodotti ----
    attesi = ["frost_1_spessore.png",
              "frost_2_temperatura_06mm.png",
              "frost_3_densita.png",
              "frost_4_densita_vs_h.png",
              "frost_5_porosita_vs_h.png",
              "frost_6_influenza_Wa.png",
              "frost_7_indipendenza_griglia_dt.png",
              "frost_state_06mm.npz"]
    log("")
    log("=" * 74)
    log("FILE PRODOTTI IN {}".format(outdir.resolve()))
    log("=" * 74)
    for nome in attesi:
        esiste = (outdir / nome).exists()
        log("  [{}] {}".format("OK      " if esiste else "MANCANTE", nome))
    log("  [OK      ] riepilogo.txt")

    (outdir / "riepilogo.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# In[ ]:




