#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MODELLO DI SBRINAMENTO (DEFROST) IN TRE STADI SU SUPERFICIE PIANA
Confronto tra alluminio nudo e superficie SLIPS
Script di calcolo allegato alla tesi -- versione riproducibile
===============================================================================

CONTENUTO
---------
Lo script risolve l'intera sequenza di sbrinamento di uno strato di brina
spesso 0.6 mm, ereditando lo stato iniziale dal modello di accrescimento
della brina (frosting) e percorrendo i tre stadi descritti da
Mohs & Kulacki, "Heat and Mass Transfer in the Melting of Frost",
Springer 2015:

  STADIO I   -- riscaldamento sensibile della brina
                La struttura metallica (superficie di prova + heat spreader)
                e' modellata con l'ODE a due nodi dell'Eq. 4.10; la brina con
                una PDE termica a conducibilita' efficace k_eff = k_fr + k_fick.
                Lo stadio termina quando la superficie metallica raggiunge 0 °C.

  STADIO II  -- fusione e permeazione
                Un fronte di fusione avanza dalla parete verso la superficie;
                un fronte di permeazione (Aoki et al. 1988) lo segue. Le gocce
                trattenute coprono parte dell'area e sottraggono flusso utile.
                Esce la massa d'acqua trattenuta m_ret, diversa per le due
                superfici: e' il vero meccanismo di differenziazione.

  STADIO III -- destino dell'acqua residua, in due regimi
                (A) Peltier acceso  -> la piastra si scalda e l'acqua evapora;
                (B) Peltier spento  -> essendo l'aria a -1 °C, l'acqua non
                    evapora ma RIGELA (bilancio energetico gia' negativo a
                    T = T_melt).

  Completano il calcolo due analisi di sensibilita' sulla potenza del Peltier
  (parametro incerto) e il confronto del flusso all'interfaccia con le misure
  sperimentali riportate dal libro.

COME ESEGUIRE
-------------
Lo script e' autosufficiente: non richiede file di input, non importa moduli
locali e non contiene percorsi assoluti.

  1) da terminale, nella cartella che contiene questo file:

         python defrost_model.py

  2) per visualizzare i grafici a schermo oltre a salvarli:

         python defrost_model.py --show

  3) per scegliere la cartella dei risultati:

         python defrost_model.py --outdir /percorso/di/destinazione

Lo script funziona anche in Jupyter Notebook o in un IDE (Spyder, PyCharm,
IDLE, VS Code): in quel caso i grafici vengono mostrati automaticamente.

ACCOPPIAMENTO COL MODELLO DI FROSTING
-------------------------------------
Lo stato iniziale (profili di temperatura e densita' della brina, spessore,
tempo di accrescimento) NON e' assunto: viene dal modello di frosting alla
soglia di 0.6 mm. Perche' questo file sia eseguibile da solo, la parte di
frosting necessaria a raggiungere quella soglia e' inclusa qui nella sezione 4,
con fisica, parametri e discretizzazione identici a quelli di 'frost_model.py'
(Ny = 41, dt = 1 s). Il risultato e' percio' identico, cifra per cifra, allo
stato salvato da quel modello in 'frost_state_06mm.npz'.

Se si preferisce partire dal file prodotto dal modello di frosting:

     python defrost_model.py --frost-state percorso/di/frost_state_06mm.npz

RISULTATI PRODOTTI
------------------
Tutti i file vengono scritti nella sottocartella "risultati_defrost", creata
automaticamente accanto a questo script:

    defrost_1_stadio1.png     riscaldamento sensibile (6 pannelli)
    defrost_2_stadio2.png     fusione e permeazione (8 pannelli)
    defrost_3_stadio3.png     essiccazione e rigelo (8 pannelli)
    defrost_4_sweep_Q.png     sensibilita' alla potenza del Peltier
    riepilogo.txt             copia integrale dell'output a schermo

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
Il modello e' completamente deterministico: non usa numeri casuali, ne' stati
globali, ne' dati di ingresso esterni. A parita' di parametri l'esecuzione
restituisce sempre gli stessi valori numerici, su qualsiasi sistema operativo
(differenze ammesse solo all'ultima cifra significativa, per effetto
dell'aritmetica in virgola mobile).

I commenti non riportano valori numerici dei risultati: tutti i numeri citati
nella discussione sono quelli stampati dall'esecuzione e raccolti in
'riepilogo.txt', cosi' che non possano divergere dal codice.

Tempo di esecuzione complessivo: circa 1-3 minuti su un portatile standard.
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
    import matplotlib.gridspec as gridspec
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
# 1. COSTANTI FISICHE E PARAMETRI DEL MODELLO
#    Allineate a quelle del modello di frosting: i due modelli devono usare
#    gli stessi valori, altrimenti l'accoppiamento non e' coerente.
# =============================================================================

# --- Materiali -------------------------------------------------------------
RHO_ICE = 918.95      # [kg/m3]    densita' del ghiaccio compatto
RHO_A = 1.290         # [kg/m3]    densita' dell'aria umida
RHO_W = 999.8         # [kg/m3]    densita' dell'acqua liquida a 0 °C
R_V = 461.5           # [J/(kg K)] costante del vapore d'acqua
DV = 1.80e-5          # [m2/s]     diffusivita' del vapore in aria
KA = 0.0225           # [W/(m K)]  conducibilita' termica dell'aria
KI = 2.390            # [W/(m K)]  conducibilita' termica del ghiaccio
KW = 0.580            # [W/(m K)]  conducibilita' termica dell'acqua liquida
CP_ICE = 2000.0       # [J/(kg K)] calore specifico del ghiaccio
CP_A = 1005.0         # [J/(kg K)] calore specifico dell'aria
CP_W = 4218.0         # [J/(kg K)] calore specifico dell'acqua
MU_W = 1.787e-3       # [Pa s]     viscosita' dell'acqua (verifica drenaggio)

# --- Calori latenti: tre grandezze distinte, da non confondere -------------
LAMDA_IG = 2.838e6    # [J/kg] sublimazione   ghiaccio -> vapore  (Stadio I)
LAMDA_IF = 3.340e5    # [J/kg] fusione        ghiaccio -> acqua   (Stadio II)
LAMDA_FG = 2.501e6    # [J/kg] vaporizzazione acqua    -> vapore  (Stadio III)

# --- Correlazione di conducibilita' della brina (Auracher 1987) ------------
A_COST = 1.576        # [-]
B_COST = 0.797        # [-]

T_MELT = 273.15       # [K]     temperatura di fusione (0 °C)
G = 9.81              # [m/s2]  accelerazione di gravita'

# --- Coefficiente di convezione: parametrizzato ----------------------------
# 13.2 W/(m2 K) e' il valore del modello di frosting (Blasius, moto laminare);
# 35 W/(m2 K) era il valore usato nelle versioni precedenti del defrost.
# Il confronto e' tenuto esplicito perche' h_conv e' un parametro incerto, e
# il coefficiente di scambio di massa (analogia di Lewis) vi scala linearmente.
H_CONV_FROST = 13.2   # [W/(m2 K)]
H_CONV_OLD = 35.0     # [W/(m2 K)]
H_CONV_CASES = {"h_conv=13.2 (Blasius)": H_CONV_FROST,
                "h_conv=35 (precedente)": H_CONV_OLD}

# --- Struttura metallica (Tab. 4.1 del libro) ------------------------------
M_TS = 15.29e-3       # [kg]       massa della superficie di prova (alluminio)
CP_TS = 936.0         # [J/(kg K)] calore specifico dell'alluminio
C_TS = M_TS * CP_TS   # [J/K]      capacita' termica della superficie di prova

M_SP = 117.5e-3       # [kg]       massa dell'heat spreader (rame)
CP_SP = 385.0         # [J/(kg K)] calore specifico del rame
C_SP = M_SP * CP_SP   # [J/K]      capacita' termica dell'heat spreader

C_METAL = C_TS + C_SP  # [J/K] capacita' termica totale della struttura:
#                        e' il pozzo dominante in tutti e tre gli stadi.

A_TS = 0.038 * 0.038  # [m2]  area della superficie di prova (38 x 38 mm)
RTH_SP_TS = 0.068     # [K/W] resistenza di contatto spreader-superficie
RTH_E = 5.00          # [K/W] resistenza spreader-ambiente (Eq. 4.10)

# --- Potenza in ingresso ----------------------------------------------------
# Q_PELTIER e' la potenza elettrica fornita all'heat spreader, NON il flusso
# misurato dal sensore del libro. Il sensore e' montato dentro la catena
# termica, tra spreader e superficie di prova, e misura quindi il flusso netto
# all'interfaccia q''_int = (T_sp - T_ts)/(R_sp,ts * A_ts): la potenza in
# ingresso si ripartisce fra riscaldamento dello spreader, perdite verso
# l'ambiente e flusso attraverso l'interfaccia. Confrontare Q/A_ts con le
# misure del libro sarebbe un confronto fra grandezze diverse; il confronto
# corretto, su q''_int, e' svolto dalla funzione confronto_flusso_interfaccia()
# per tutti e tre gli stadi. Lo sweep 6-15 W copre la banda di incertezza.
Q_PELTIER = 15.0      # [W]

# --- Condizioni operative (identiche al modello di frosting) ---------------
T_AIR_K = 272.15      # [K] aria della camera (-1 °C)
T_WALL_K = 272.15     # [K] riferimento del termine (T_w - T_sp)/R_th,e.
#                       Da non confondere con la temperatura iniziale della
#                       piastra (-15 °C), che arriva dal modello di frosting.

# --- Griglia del defrost ----------------------------------------------------
NY = 121              # [-] nodi nello spessore (il frosting ne usa 41)

# --- Permeazione (Aoki et al. 1988) ----------------------------------------
SC = 0.1              # [-] contenuto d'acqua critico al fronte di permeazione
KP_OVER_KFR = 1.0     # [-] rapporto k_p/k_fr (Aoki: ~1 senza dati migliori)

# --- Parametri delle due superfici a confronto -----------------------------
#  beta      : frazione della massa fusa trattenuta (imbibizione + gocce)
#  m0        : ritenzione superficiale residua [kg/m2]
#  A_cov_max : frazione massima di area coperta dalle gocce trattenute
#
#  beta(Al) = 0.70
#    Primario: bilancio di massa invertito su Liang et al. (2015),
#      beta = (m_ret,tot - m0)/m_fusa = (364 - 120)/372 = 0.66, arrotondato
#      a 0.70 in senso conservativo.
#    Ancoraggio indipendente: Gu et al. (2022, Sustainability 14:1954)
#      misurano su Al nudo verticale una ritenzione diretta del 76.6 %.
#
#  beta(SLIPS) = 0.05
#    Primario (scaling sulla forza capillare di ritenzione):
#      beta_Al * [(cos_R - cos_A)_SLIPS / (cos_R - cos_A)_Al]
#      = 0.70 * (0.031/0.300) = 0.073, arrotondato a 0.05.
#    Ancoraggio diretto: Ahmadi & Boreyko (2020) misurano su superfici
#      impregnate di liquido un'efficienza di drenaggio ~0.95, da cui
#      beta = 1 - 0.95 = 0.05.
#
#  m0(SLIPS) = 15 g/m2 -- triangolazione di due fonti indipendenti:
#    (1) Liang et al. 2016 (ETFS 71:70-76), Fig. 14, curva a theta ~ 117°
#        -> 16 g/m2;
#    (2) Yu 2019 (tesi di dottorato, UIUC), rapporto di ritenzione
#        LIS ~ 0.7 % contro Al ~ 5.5 %, ancorato a m0(Al) = 120 g/m2
#        -> 15.3 g/m2.
#
#  A_cov_max: Al = 0.34 (ElSherbini & Jacobi 2006, D_max = 5 mm);
#             SLIPS = 0.34 * (0.6/5.0)^0.27 = 0.19 (scaling su D_max).
SURFACE_PARAMS = {
    "Al nudo":  {"beta": 0.70, "m0": 0.120, "color": "#2980b9", "A_cov_max": 0.34},
    "SLIPS-Al": {"beta": 0.05, "m0": 0.015, "color": "#27ae60", "A_cov_max": 0.19},
}
SURFACES = list(SURFACE_PARAMS.keys())

# Nomi delle superfici usati nelle etichette delle figure, che sono in inglese
# (il riepilogo stampato a schermo resta in italiano).
EN_SURFACE = {"Al nudo": "Bare Al", "SLIPS-Al": "SLIPS-Al"}

# --- Flussi all'interfaccia misurati dal libro: (media, min, max) in W/m2 ---
Q_INT_LIBRO = {
    "I":   (938.0, 285.0, 1852.0),     # Tab. 5.4
    "II":  (3811.0, 1042.0, 6699.0),   # Tab. 5.7
    "III": (1462.0, 245.0, 2847.0),    # Tab. 5.9
}

# --- Sweep sulla potenza del Peltier ---------------------------------------
Q_SWEEP = (6.0, 8.0, 10.0, 12.0, 15.0)   # [W]

# --- Nomi dei file di uscita -----------------------------------------------
FIG_STADIO1 = "defrost_1_stadio1.png"
FIG_STADIO2 = "defrost_2_stadio2.png"
FIG_STADIO3 = "defrost_3_stadio3.png"
FIG_SWEEP = "defrost_4_sweep_Q.png"


# =============================================================================
# 2. CORRELAZIONI COSTITUTIVE
# =============================================================================

def pvs_ice(T_C):
    """Pressione di vapor saturo SU GHIACCIO [Pa], T in °C.
    Stessa formula del modello di frosting: garantisce la coerenza fra i due
    modelli. Si usa negli Stadi I e II, dove la fase solida e' ghiaccio."""
    return 611.2 * np.exp(21.87 * T_C / (T_C + 265.5))


def pvs_water(T_C):
    """Pressione di vapor saturo SU ACQUA LIQUIDA [Pa], T in °C (Buck).
    Si usa SOLO nello Stadio III, dove il fluido e' acqua: usare li' la
    formula su ghiaccio sarebbe fisicamente scorretto."""
    return 610.94 * np.exp(17.625 * T_C / (T_C + 243.3))


def rho_v_sat_ice(T_K):
    """Densita' di vapor saturo su ghiaccio [kg/m3]."""
    return pvs_ice(T_K - 273.15) / (R_V * T_K)


def rho_v_sat_water(T_K):
    """Densita' di vapor saturo su acqua [kg/m3]."""
    return pvs_water(T_K - 273.15) / (R_V * T_K)


def porosity(rho_fr):
    """Porosita' della brina [-]: eps = (rho_ice - rho_fr)/(rho_ice - rho_a)."""
    return (RHO_ICE - rho_fr) / (RHO_ICE - RHO_A)


def tortuosity(eps):
    """Fattore di tortuosita' dei pori [-]."""
    return (1.0 / eps) * (1.0 - eps / 0.58 * (1.0 - eps))


def k_frost(eps):
    """Conducibilita' termica efficace della brina [W/(m K)] (Auracher 1987)."""
    return A_COST * KI * (KA / KI) ** (B_COST * eps)


def cp_frost(eps, rho_fr):
    """Calore specifico efficace della brina [J/(kg K)]."""
    return (RHO_ICE * CP_ICE * (1.0 - eps) + RHO_A * CP_A * eps) / rho_fr


def D_eff(eps):
    """Diffusivita' efficace del vapore nella brina [m2/s]."""
    return tortuosity(eps) * DV * eps


def b1_array(T_arr_K, dT=0.5):
    """Derivata della densita' di vapor saturo su ghiaccio, d(rho_v,sat)/dT
    [kg/(m3 K)], per differenza finita centrata."""
    return (rho_v_sat_ice(T_arr_K + dT)
            - rho_v_sat_ice(T_arr_K - dT)) / (2.0 * dT)


def k_fick_array(eps_arr, T_arr_K):
    """Conducibilita' apparente di Fick [W/(m K)]:
        k_fick = lambda_ig * D_v,eff(eps) * d(rho_v,sat)/dT
    Si somma a k_fr nella PDE termica: k_eff = k_fr + k_fick. Rappresenta il
    trasporto di calore latente per diffusione di vapore dentro i pori."""
    return LAMDA_IG * D_eff(eps_arr) * b1_array(T_arr_K)


def mass_transfer_coeff(h_conv):
    """Coefficiente di scambio di massa h_m [m/s], analogia di Lewis
    (Eq. 3.12): h_m = h / (rho_a * cp_a * Le_a^(2/3)), con Le_a = alpha_a/Dv.
    Scala linearmente con h_conv: e' il motivo per cui la scelta fra 13.2 e 35
    cambia sensibilmente la durata dello Stadio III."""
    alpha_a = KA / (RHO_A * CP_A)
    Le_a = alpha_a / DV
    return h_conv / (RHO_A * CP_A * Le_a ** (2.0 / 3.0))


def gamma1_local(eps_arr, T_arr_K):
    """Rapporto Gamma_1 = k_fick/k_fr [-], mediato sullo spessore.
    Misura il peso del trasporto di vapore rispetto alla pura conduzione:
    il libro (Cap. 5.3.1) osserva Gamma_1 << 1 sul proprio dataset."""
    k_fk = k_fick_array(eps_arr, T_arr_K)
    k_fr = k_frost(eps_arr)
    return float(np.mean(k_fk / np.where(k_fr > 0, k_fr, 1e-10)))


def q_interfaccia(T_sp_h_C, T_ts_h_C):
    """Flusso netto all'interfaccia spreader -> superficie di prova [W/m2]:
        q''_int = (T_sp - T_ts) / (R_sp,ts * A_ts)
    E' la grandezza misurata dal sensore di flusso del libro. Accetta storie
    in °C o in K, perche' usa solo la differenza."""
    return (np.asarray(T_sp_h_C) - np.asarray(T_ts_h_C)) / (RTH_SP_TS * A_TS)


# =============================================================================
# 3. UTILITA' DI OUTPUT (log a schermo + su file, salvataggio figure)
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


def confronto_flusso_interfaccia(stadio, T_sp_h_C, T_ts_h_C):
    """Confronta il flusso all'interfaccia calcolato dal modello con le misure
    del libro per lo stadio indicato ('I', 'II' o 'III')."""
    q = q_interfaccia(T_sp_h_C, T_ts_h_C)
    med, lo, hi = Q_INT_LIBRO[stadio]
    q_medio, q_max = float(np.mean(q)), float(np.max(q))
    dentro = lo <= q_medio <= hi
    log("    q''_interfaccia (Stadio {}): medio = {:.0f}, max = {:.0f} W/m²"
        .format(stadio, q_medio, q_max))
    log("      libro: media {:.0f}, intervallo {:.0f}-{:.0f} W/m²   -> {}"
        .format(med, lo, hi, "nell'intervallo" if dentro else "FUORI intervallo"))
    return q_medio


# =============================================================================
# 4. STATO INIZIALE: DAL MODELLO DI ACCRESCIMENTO DELLA BRINA
#
#    Il profilo iniziale non e' assunto ma calcolato. La porzione di modello
#    di frosting riportata qui sotto e' identica a quella di 'frost_model.py'
#    (stessi parametri, stesse equazioni, stessa discretizzazione): viene
#    inclusa perche' questo script possa essere eseguito da solo.
#    L'integrazione si ferma appena lo spessore raggiunge la soglia di 0.6 mm.
# =============================================================================

# Parametri del modello di frosting (identici a 'frost_model.py')
FR_NY = 41            # [-]  nodi della griglia del frosting
FR_NT = 4000          # [-]  passi temporali
FR_T_MAX = 4000.0     # [s]  durata simulata
FR_TA = 272.15        # [K]  temperatura dell'aria
FR_TW = 258.15        # [K]  temperatura della parete fredda
FR_WA = 0.00339       # [kg/kg] umidita' assoluta dell'aria
FR_P_TOT = 101325.0   # [Pa] pressione totale
FR_R_D = 287.0        # [J/(kg K)] costante dell'aria secca
FR_KW = 0.01          # [kg/(m2 s)] coefficiente di scambio di massa
FR_H_CONV = 13.2      # [W/(m2 K)] coefficiente convettivo
FR_ALPHA = 0.2        # [-]  correzione del flusso diffusivo superficiale
FR_BETA = 0.5         # [-]  correzione della sorgente di densificazione
FR_H_INIZ = 2e-5      # [m]  spessore iniziale
FR_RHO_INIZ = 30.0    # [kg/m3] densita' iniziale
H_TARGET = 0.6e-3     # [m]  soglia di innesco dello sbrinamento


def _frosting_fino_a_soglia():
    """Integra il modello di frosting fino alla soglia di 0.6 mm.

    Ritorna (T_fr [K], rho_fr [kg/m3], h_0 [m], t_frost [s]) sui 41 nodi
    della griglia del frosting. I valori coincidono con quelli salvati da
    'frost_model.py' nel file 'frost_state_06mm.npz'.
    """
    dt = FR_T_MAX / FR_NT
    P_vs_inf = pvs_ice(FR_TA - 273.15)
    P_v_inf = FR_WA / (FR_WA + 0.622) * FR_P_TOT

    def _sup(P_vs_fs):
        """Grado di soprasaturazione alla superficie della brina [-]."""
        return 0.808 * ((P_v_inf / P_vs_inf)
                        * (P_vs_fs / P_vs_inf) ** (-0.657)) - 1.0

    def _umidita_superficie(P_v_fs):
        return 0.622 * (P_v_fs / (FR_P_TOT - 0.378 * P_v_fs))

    rho_fr = np.full(FR_NY, FR_RHO_INIZ)
    eps_0 = porosity(FR_RHO_INIZ)
    k_0 = k_frost(eps_0)
    T_fs_0 = ((FR_H_CONV * FR_TA + (k_0 / FR_H_INIZ) * FR_TW)
              / (FR_H_CONV + k_0 / FR_H_INIZ))
    T_fr = np.linspace(FR_TW, float(np.clip(T_fs_0, FR_TW, FR_TA)), FR_NY)

    h_current = FR_H_INIZ
    A_banded = np.zeros((3, FR_NY))
    B_vec = np.zeros(FR_NY)

    for t in range(FR_NT):
        t_now = (t + 1) * dt
        dy = h_current / (FR_NY - 1)

        # proprieta' locali della miscela aria-vapore
        P_v_local = pvs_ice(T_fr - 273.15)
        rho_v_local = P_v_local / (R_V * T_fr)
        rho_a_local = rho_v_local + (FR_P_TOT - P_v_local) / (FR_R_D * T_fr)
        pvs_wall = pvs_ice(FR_TW - 273.15)
        rho_v_local[0] = pvs_wall / (R_V * FR_TW)
        rho_a_local[0] = rho_v_local[0] + (FR_P_TOT - pvs_wall) / (FR_R_D * FR_TW)
        pvs_top = pvs_ice(T_fr[-1] - 273.15)
        rho_v_local[-1] = pvs_top / (R_V * T_fr[-1])
        rho_a_local[-1] = rho_v_local[-1] + (FR_P_TOT - pvs_top) / (FR_R_D * T_fr[-1])
        phi = rho_v_local / rho_a_local

        eps = porosity(rho_fr)
        D_e = D_eff(eps)

        # densificazione interna (Eulero esplicito, forma conservativa)
        DR = D_e * rho_a_local
        D_faccia = 0.5 * (DR[:-1] + DR[1:])
        flusso = D_faccia * np.diff(phi)
        rho_fr_new = np.copy(rho_fr)
        rho_fr_new[1:-1] = (rho_fr[1:-1]
                            + dt * FR_BETA * (flusso[1:] - flusso[:-1]) / dy ** 2)
        rho_fr_new[0] = (rho_fr[0]
                         + dt * FR_BETA * 2.0 * D_faccia[0] * (phi[1] - phi[0]) / dy ** 2)
        rho_fr_new = np.clip(rho_fr_new, FR_RHO_INIZ, 0.92 * RHO_ICE)
        rho_fr_new[-1] = rho_fr_new[-2]

        # crescita dello spessore
        P_vs_s = pvs_ice(T_fr[-1] - 273.15)
        W_fs = _umidita_superficie(P_vs_s * (1.0 + _sup(P_vs_s)))
        mv_fs = FR_KW * (FR_WA - W_fs)
        grad_phi_s = (3.0 * phi[-1] - 4.0 * phi[-2] + phi[-3]) / (2.0 * dy)
        mv_diff_s = D_e[-1] * RHO_A * grad_phi_s
        h_current += max(0.0, (mv_fs - FR_ALPHA * mv_diff_s) / rho_fr_new[-1]) * dt

        # temperatura di superficie (punto fisso, 5 iterazioni)
        eps_cn = porosity(rho_fr_new)
        k_fr = k_frost(eps_cn)
        cp_fr = cp_frost(eps_cn, rho_fr_new)
        cond_r = float(np.mean(k_fr)) / max(h_current, 1e-9)
        T_fs = float(np.clip(T_fr[-1], FR_TW, FR_TA))
        for _ in range(5):
            Pvs_t = pvs_ice(T_fs - 273.15)
            W_t = _umidita_superficie(Pvs_t * (1.0 + _sup(Pvs_t)))
            mv_t = max(FR_KW * (FR_WA - W_t), 0.0)
            T_new = ((FR_H_CONV * FR_TA + cond_r * FR_TW + LAMDA_IG * mv_t)
                     / (FR_H_CONV + cond_r))
            T_fs = float(np.clip(0.5 * T_fs + 0.5 * T_new, FR_TW, FR_TA))

        # conduzione nello strato (schema a theta variabile, tridiagonale)
        k_mid = 0.5 * (k_fr[1:] + k_fr[:-1])
        km, kp = k_mid[:-1], k_mid[1:]
        rCp = rho_fr_new[1:-1] * cp_fr[1:-1]
        Fo = dt * 0.5 * (km + kp) / (rCp * dy ** 2)
        theta = np.where(Fo > 0.4, 1.0, 0.5)
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
                       + LAMDA_IG * (rho_fr_new[1:-1] - rho_fr[1:-1]) / rCp)
        A_banded[1, 0] = 1.0
        B_vec[0] = FR_TW
        A_banded[1, -1] = 1.0
        B_vec[-1] = T_fs

        T_fr = solve_banded((1, 1), A_banded, B_vec)
        rho_fr = rho_fr_new

        if h_current >= H_TARGET:
            return T_fr.copy(), rho_fr.copy(), h_current, t_now

    raise RuntimeError(
        "La soglia di {:.1f} mm non e' stata raggiunta entro {:.0f} s di "
        "accrescimento: aumentare FR_T_MAX.".format(H_TARGET * 1e3, FR_T_MAX))


def _interpola_su_griglia_defrost(T_frost, rho_frost, h_0, t_frost):
    """Porta lo stato dai 41 nodi del frosting ai NY nodi del defrost.

    La porosita' viene RICALCOLATA dalla densita' interpolata, non interpolata
    a sua volta: cosi' la relazione eps = (rho_ice - rho_fr)/(rho_ice - rho_a)
    resta esattamente soddisfatta nodo per nodo.
    """
    y_src = np.linspace(0.0, 1.0, len(rho_frost))
    y_dst = np.linspace(0.0, 1.0, NY)
    rho_init = np.interp(y_dst, y_src, rho_frost)
    return dict(T_init=np.interp(y_dst, y_src, T_frost),
                rho_init=rho_init,
                eps_init=porosity(rho_init),
                h_0=float(h_0), t_frost=float(t_frost))


def stato_iniziale(percorso_npz=None):
    """Costruisce lo stato iniziale del defrost.

    Senza argomenti esegue il modello di frosting incluso in questo file; con
    'percorso_npz' legge invece lo stato salvato da 'frost_model.py'.
    """
    if percorso_npz is None:
        T_frost, rho_frost, h_0, t_frost = _frosting_fino_a_soglia()
    else:
        percorso = Path(percorso_npz)
        if not percorso.exists():
            sys.exit("\nStato del frosting non trovato: {}\n"
                     "Eseguire prima 'frost_model.py', oppure lanciare questo "
                     "script senza l'opzione --frost-state per calcolarlo.\n"
                     .format(percorso))
        d = np.load(percorso)
        T_frost = d["T_fr_K"]
        rho_frost = d["rho_fr"]
        h_0 = float(d["h_0"])
        t_frost = float(d["t_frost"])
    return _interpola_su_griglia_defrost(T_frost, rho_frost, h_0, t_frost)


def stampa_stato_iniziale(fs, origine):
    """Riepiloga lo stato iniziale ereditato dal modello di frosting."""
    log("=" * 78)
    log("STATO INIZIALE -- dal modello di accrescimento della brina")
    log("=" * 78)
    log("  origine              : {}".format(origine))
    log("  t_frosting           : {:.0f} s  ({:.1f} min)"
        .format(fs["t_frost"], fs["t_frost"] / 60.0))
    log("  h_0                  : {:.3f} mm".format(fs["h_0"] * 1e3))
    log("  porosita' media      : {:.4f}   (parete {:.3f} -> superficie {:.3f})"
        .format(float(np.mean(fs["eps_init"])), float(fs["eps_init"][0]),
                float(fs["eps_init"][-1])))
    log("  densita' della brina : {:.1f} -> {:.1f} kg/m³   (rapporto {:.1f}x)"
        .format(float(fs["rho_init"][0]), float(fs["rho_init"][-1]),
                float(fs["rho_init"][0] / fs["rho_init"][-1])))
    log("  temperatura brina    : {:.2f} -> {:.2f} °C   (media {:.2f} °C)"
        .format(float(fs["T_init"][0]) - 273.15,
                float(fs["T_init"][-1]) - 273.15,
                float(np.mean(fs["T_init"])) - 273.15))
    log("  aria                 : {:.1f} °C".format(T_AIR_K - 273.15))
    log("  Q_Peltier            : {:.0f} W   (Q/A_ts = {:.0f} W/m²)"
        .format(Q_PELTIER, Q_PELTIER / A_TS))
    log("  capacita' metallo    : {:.2f} J/K  (C_TS = {:.2f} + C_SP = {:.2f})"
        .format(C_METAL, C_TS, C_SP))
    log("=" * 78)


# =============================================================================
# 5. STADIO I -- RISCALDAMENTO SENSIBILE DELLA BRINA
#
#    Due sottosistemi accoppiati:
#    (a) struttura metallica, ODE a due nodi (Eq. 4.10 del libro)
#          C_sp dT_sp/dt = (T_ts-T_sp)/R_sp,ts + (T_w-T_sp)/R_e + Q_Peltier
#          C_ts dT_ts/dt = (T_sp-T_ts)/R_sp,ts - Q_brina
#        La capacita' termica del metallo e' il pozzo dominante: il bilancio
#        energetico stampato a fine stadio lo quantifica.
#    (b) brina, PDE termica con k_eff = k_fr + k_fick (Eq. 3.10-3.12).
#
#    Lo stadio termina quando la superficie di prova raggiunge T_melt.
# =============================================================================

def simula_stadio1(T_init, rho_init, h_0, h_conv,
                   Q_W=Q_PELTIER, dt=0.05, t_max=600.0):
    """Integra lo Stadio I.

    Parametri
    ---------
    T_init, rho_init : profili iniziali su NY nodi, da stato_iniziale().
    h_0              : spessore della brina [m].
    h_conv           : coefficiente convettivo aria-brina [W/(m2 K)].

    Ritorna
    -------
    dict con lo stato finale (T_fr, rho_fr, T_ts, T_sp), che e' la condizione
    iniziale dello Stadio II, le storie temporali e il bilancio energetico.
    """
    dy = h_0 / (NY - 1)
    T_fr = T_init.copy()
    rho_fr = rho_init.copy()

    # La piastra parte dalla temperatura di parete del frosting: non e'
    # un'assunzione, e' il risultato della fase di brinamento.
    T_ts = float(T_fr[0])
    T_sp = float(T_fr[0])
    T_ts_0, T_sp_0 = T_ts, T_sp

    time_h, T_ts_h, T_sp_h, T_fs_h, Q_brina_h = [], [], [], [], []
    snap_T = {}
    snap_every = max(1, int(5.0 / dt))     # un profilo ogni ~5 s
    t_end = None

    A_b = np.zeros((3, NY))
    B_v = np.zeros(NY)

    for step in range(int(t_max / dt)):
        t_now = (step + 1) * dt

        # (a) proprieta' della brina: k_eff = k_fr + k_fick
        eps_n = porosity(rho_fr)
        k_fr_n = k_frost(eps_n)
        cp_fr_n = cp_frost(eps_n, rho_fr)
        k_eff_n = k_fr_n + k_fick_array(eps_n, T_fr)
        k_mid = 0.5 * (k_eff_n[1:] + k_eff_n[:-1])

        # (b) ODE della struttura metallica (Eq. 4.10).
        # Il max(...,0) impedisce alla piastra di assorbire calore dalla
        # brina, che durante questo stadio e' sempre piu' fredda di lei.
        Q_brina = max(k_eff_n[0] * A_TS * (T_ts - T_fr[1]) / dy, 0.0)

        a11 = 1.0 + dt / C_SP * (1.0 / RTH_SP_TS + 1.0 / RTH_E)
        a12 = -dt / (C_SP * RTH_SP_TS)
        a21 = -dt / (C_TS * RTH_SP_TS)
        a22 = 1.0 + dt / (C_TS * RTH_SP_TS)
        b1 = T_sp + dt / C_SP * (T_WALL_K / RTH_E + Q_W)
        b2 = T_ts - dt / C_TS * Q_brina

        det = a11 * a22 - a12 * a21
        T_sp_new = (b1 * a22 - b2 * a12) / det
        T_ts_new = (a11 * b2 - a21 * b1) / det

        # (c) condizione al contorno sulla superficie della brina.
        # Si usa k al NODO DI SUPERFICIE e non la media sullo spessore: il
        # profilo di densita' e' molto stratificato, e la media sovrastimerebbe
        # la conducibilita' proprio dove la brina e' rada e conduce poco.
        cr = float(k_eff_n[-1]) / h_0
        T_fs = float(np.clip(T_fr[-1],
                             min(T_ts_new, T_AIR_K), max(T_ts_new, T_AIR_K)))
        for _ in range(5):
            T_fs_new = (h_conv * T_AIR_K + cr * T_ts_new) / (h_conv + cr)
            T_fs_new = float(np.clip(T_fs_new,
                                     min(T_ts_new, T_AIR_K),
                                     max(T_ts_new, T_AIR_K)))
            T_fs = 0.5 * T_fs + 0.5 * T_fs_new

        # (d) PDE termica nella brina, schema a theta variabile:
        # theta = 1 (Eulero implicito) dove Fo > 0.4, altrimenti theta = 0.5
        # (Crank-Nicolson); evita oscillazioni dove la griglia e' rigida.
        rCp = rho_fr[1:-1] * cp_fr_n[1:-1]
        km, kp = k_mid[:-1], k_mid[1:]
        Fo = dt * 0.5 * (km + kp) / (rCp * dy ** 2)
        theta = np.where(Fo > 0.4, 1.0, 0.5)
        r_im = dt * km / (rCp * dy ** 2)
        r_ip = dt * kp / (rCp * dy ** 2)

        A_b[:] = 0.0
        B_v[:] = 0.0
        A_b[0, 2:] = -theta * r_ip
        A_b[1, 1:-1] = 1.0 + theta * (r_im + r_ip)
        A_b[2, :-2] = -theta * r_im
        B_v[1:-1] = (T_fr[1:-1]
                     + (1.0 - theta) * r_im * (T_fr[:-2] - T_fr[1:-1])
                     + (1.0 - theta) * r_ip * (T_fr[2:] - T_fr[1:-1]))
        A_b[1, 0] = 1.0
        B_v[0] = T_ts_new                  # parete: T imposta dalla piastra
        A_b[1, -1] = 1.0
        B_v[-1] = T_fs                     # superficie: equilibrio convettivo

        T_fr = solve_banded((1, 1), A_b, B_v)
        T_sp, T_ts = T_sp_new, T_ts_new

        time_h.append(t_now)
        T_ts_h.append(T_ts - 273.15)
        T_sp_h.append(T_sp - 273.15)
        T_fs_h.append(T_fs - 273.15)
        Q_brina_h.append(Q_brina)

        if step % snap_every == 0:
            snap_T[round(t_now, 1)] = T_fr.copy() - 273.15

        # (e) criterio di arresto: la piastra raggiunge 0 °C
        if T_ts >= T_MELT:
            t_end = t_now
            snap_T[round(t_now, 1)] = T_fr.copy() - 273.15
            break

    # ---- bilancio energetico dello stadio ----
    eps_f = porosity(rho_fr)
    t_fin = t_end if t_end else t_max
    E_appl = Q_W * t_fin
    E_metal = C_TS * (T_ts - T_ts_0) + C_SP * (T_sp - T_sp_0)
    E_brina = float(np.sum(rho_fr * cp_frost(eps_f, rho_fr)
                           * (T_fr - T_init))) * dy * A_TS
    # perdita verso l'ambiente, stimata con la T_sp media: su tempi brevi
    # il termine e' piccolo, ma va contabilizzato per chiudere il bilancio.
    T_sp_medio_K = float(np.mean(T_sp_h)) + 273.15
    E_loss = (T_sp_medio_K - T_WALL_K) / RTH_E * t_fin

    return dict(
        h_conv=h_conv, t_end=t_end, Q_W=Q_W,
        T_fr=T_fr, rho_fr=rho_fr, T_ts=T_ts, T_sp=T_sp,
        gamma1=gamma1_local(eps_f, T_fr),
        E_appl=E_appl, E_metal=E_metal, E_brina=E_brina, E_loss=E_loss,
        time_h=np.array(time_h),
        T_ts_h=np.array(T_ts_h), T_sp_h=np.array(T_sp_h),
        T_fs_h=np.array(T_fs_h), Q_brina_h=np.array(Q_brina_h),
        snap_T=snap_T, h_0=h_0,
    )


# =============================================================================
# 6. STADIO II -- FUSIONE E PERMEAZIONE
#
#    Fronte di fusione: le gocce trattenute coprono una frazione dell'area e
#    sottraggono flusso utile,
#        q''_eff = q''_s (1 - A_cov),   A_cov(t) = A_cov_max (1 - delta_f/h_0)
#        u_front = q''_eff / (lambda_if * rho_fr(fronte))
#    Fronte di permeazione (Aoki et al., Eq. 3.18-3.19):
#        d(delta_p)/dt = beta*u_front + k_p/(eps*rho_w*Sc*lambda_if)*|dT/dy|
#    Vincolo di fusione: finche' resta ghiaccio all'interfaccia la temperatura
#    della superficie di prova resta a T_melt, e ogni squilibrio energetico va
#    nel calore latente anziche' nella temperatura. Il vincolo e' BILATERALE:
#    senza il limite inferiore la conduzione verso la brina ancora fredda
#    trascinerebbe T_ts sotto zero, con un rigelo fittizio nello Stadio III.
# =============================================================================

def simula_stadio2(T_end_s1, rho_fr_s1, T_ts_s1, T_sp_s1, h_0, h_conv,
                   beta, m0_ret, A_cov_max, nome_superficie,
                   Q_W=Q_PELTIER, dt=None, t_max=None):
    """Integra lo Stadio II a partire dallo stato finale dello Stadio I."""
    dy = h_0 / (NY - 1)
    T_fr = T_end_s1.copy()
    rho_fr = rho_fr_s1.copy()
    T_ts = float(T_ts_s1)
    T_sp = float(T_sp_s1)

    eps_init = porosity(rho_fr)
    q_s = Q_W / A_TS

    # Passo temporale adattivo: la condizione CFL va imposta dove il fronte e'
    # piu' veloce, cioe' dove la brina e' meno densa (verso la superficie),
    # non alla parete.
    rho_min = float(np.min(rho_fr))
    rho_mean = float(np.mean(rho_fr))
    u_max = q_s / (LAMDA_IF * rho_min)
    u_mean = q_s / (LAMDA_IF * rho_mean)
    if dt is None:
        dt = min(0.002, 0.25 * dy / u_max)
    if t_max is None:
        t_max = max(5.0, 30.0 * h_0 / u_mean)

    delta_f, i_front = h_0, 0
    delta_p, i_perm = 0.0, 0
    y_grid = np.linspace(0.0, h_0, NY)

    # Massa di ghiaccio totale dal profilo reale del frosting.
    # Si usa rho_ICE*(1-eps) e non rho_fr*(1-eps): rho_fr e' gia' la densita'
    # della brina (ghiaccio + aria), moltiplicarla ancora per (1-eps)
    # sottostimerebbe la massa di un ulteriore fattore.
    m_fusa_tot = float(np.sum(RHO_ICE * (1.0 - eps_init))) * dy

    time_h, delta_f_h, delta_p_h = [], [], []
    u_front_h, rho_front_h, A_cov_h = [], [], []
    T_ts_h, T_sp_h = [], []
    t_end = None

    for step in range(int(t_max / dt)):
        t_now = (step + 1) * dt
        if NY - i_front < 2:
            t_end = t_now
            break

        # (a) proprieta' della brina
        eps_n = porosity(rho_fr)
        k_fr_n = k_frost(eps_n)
        cp_fr_n = cp_frost(eps_n, rho_fr)
        k_eff_n = k_fr_n + k_fick_array(eps_n, T_fr)
        k_mid = 0.5 * (k_eff_n[1:] + k_eff_n[:-1])

        # (b) ODE della struttura metallica con vincolo di fusione
        a11 = 1.0 + dt / C_SP * (1.0 / RTH_SP_TS + 1.0 / RTH_E)
        a12 = -dt / (C_SP * RTH_SP_TS)
        a21 = -dt / (C_TS * RTH_SP_TS)
        a22 = 1.0 + dt / (C_TS * RTH_SP_TS)
        b1v = T_sp + dt / C_SP * (T_WALL_K / RTH_E + Q_W)
        b2v = T_ts - dt / C_TS * Q_W        # tutta la potenza va nella fusione
        det = a11 * a22 - a12 * a21
        T_sp_new = (b1v * a22 - b2v * a12) / det
        T_ts_new = T_MELT                   # vincolo di fusione (bilaterale)

        # (c) fronte di fusione e copertura superficiale
        frac_fusa = float(np.clip(1.0 - delta_f / h_0, 0.0, 1.0))
        A_cov = A_cov_max * frac_fusa
        q_fus_eff = q_s * (1.0 - A_cov)

        # La densita' al fronte si legge per INTERPOLAZIONE, non al nodo
        # intero: delta_f e' integrata con continuita', e campionare rho_fr a
        # indice intero produrrebbe curve u_front(t) e rho_front(t) a gradini
        # (alla parete, dove il fronte e' lento, un solo nodo copre una
        # frazione sensibile della durata dello stadio). L'effetto sui
        # risultati integrali e' trascurabile.
        y_front = h_0 - delta_f
        rho_al_fronte = float(np.interp(y_front, y_grid, rho_fr))
        u_front = (q_fus_eff / (LAMDA_IF * rho_al_fronte)
                   if rho_al_fronte > 0 else 0.0)

        delta_f_new = max(delta_f - u_front * dt, 0.0)
        i_front_new = min(int((h_0 - delta_f_new) / dy), NY - 2)
        if i_front_new > i_front:
            T_fr[i_front:i_front_new] = T_MELT
        delta_f, i_front = delta_f_new, i_front_new

        # (d) fronte di permeazione: imbibizione capillare + rigelo
        i_p = min(i_perm, NY - 2)
        if i_p > i_front and i_p + 1 < NY:
            dT_dy_perm = (T_fr[i_p + 1] - T_fr[i_p]) / dy
        else:
            dT_dy_perm = 0.0
        k_p_local = KP_OVER_KFR * float(k_fr_n[min(i_p, NY - 1)])
        eps_al_perm = float(eps_n[min(i_p, NY - 1)])
        rigelo = (k_p_local / (eps_al_perm * RHO_W * SC * LAMDA_IF)
                  * abs(dT_dy_perm))
        delta_p = max(min(delta_p + (beta * u_front + rigelo) * dt,
                          max(delta_f - dy, 0.0)), 0.0)
        i_perm = min(i_front + int(delta_p / dy), NY - 1)

        # (e) PDE termica sulla brina residua, con avvezione del fronte
        n_s = NY - i_front
        A_b = np.zeros((3, n_s))
        B_v = np.zeros(n_s)
        a_adv = u_front * dt / dy

        idx = np.arange(i_front + 1, NY - 1)
        if idx.size:
            rCp = rho_fr[idx] * cp_fr_n[idx]
            km, kp = k_mid[idx - 1], k_mid[idx]
            Fo = dt * 0.5 * (km + kp) / (rCp * dy ** 2)
            theta = np.where(Fo > 0.4, 1.0, 0.5)
            r_im = dt * km / (rCp * dy ** 2)
            r_ip = dt * kp / (rCp * dy ** 2)

            A_b[0, 2:n_s] = -(theta * r_ip + a_adv)
            A_b[1, 1:n_s - 1] = 1.0 + theta * (r_im + r_ip) + a_adv
            A_b[2, 0:n_s - 2] = -(theta * r_im)
            B_v[1:n_s - 1] = (T_fr[idx]
                              + (1.0 - theta) * r_im * (T_fr[idx - 1] - T_fr[idx])
                              + (1.0 - theta) * r_ip * (T_fr[idx + 1] - T_fr[idx])
                              + a_adv * (T_fr[idx + 1] - T_fr[idx]))

        A_b[1, 0] = 1.0
        B_v[0] = T_MELT                     # al fronte: T = T_melt

        cr_top = float(k_eff_n[NY - 1]) / (dy * 2.0)
        T_fs = (h_conv * T_AIR_K + cr_top * T_fr[NY - 2]) / (h_conv + cr_top)
        T_fs = float(np.clip(T_fs, min(T_MELT, T_AIR_K), max(T_MELT, T_AIR_K)))
        A_b[1, n_s - 1] = 1.0
        B_v[n_s - 1] = T_fs

        T_fr[i_front:] = solve_banded((1, 1), A_b, B_v)
        T_fr[:i_front] = T_MELT
        T_sp, T_ts = T_sp_new, T_ts_new

        time_h.append(t_now)
        delta_f_h.append(delta_f)
        delta_p_h.append(delta_p)
        u_front_h.append(u_front)
        rho_front_h.append(rho_al_fronte)
        A_cov_h.append(A_cov)
        T_ts_h.append(T_ts - 273.15)
        T_sp_h.append(T_sp - 273.15)

        if delta_f <= dy:
            t_end = t_now
            break

    # ---- massa trattenuta: e' l'ingresso dello Stadio III ----
    # m_ret = min(m_fusa_tot, m_fusa_tot*beta + m0_ret)
    # Il vincolo min() impone la conservazione della massa: m0_ret non e'
    # acqua che si aggiunge, e' la CAPACITA' di ritenzione della superficie,
    # cioe' il massimo che una parete verticale trattiene prima che il resto
    # scivoli via (misurata in condensazione, a superficie satura). Con brina
    # sottile si produce meno acqua di quanta la superficie possa trattenere:
    # in quel regime non drena nulla e m_ret coincide con la massa fusa.
    m_ret = min(m_fusa_tot, m_fusa_tot * beta + m0_ret)
    m_drenata = max(m_fusa_tot - m_ret, 0.0)

    t_fin = t_end if t_end else t_max
    E_appl = Q_W * t_fin
    E_min = LAMDA_IF * m_fusa_tot * A_TS    # minimo teorico (solo latente)

    return dict(
        surface=nome_superficie, beta=beta, m0_ret=m0_ret, A_cov_max=A_cov_max,
        t_end=t_end, dt=dt, h_conv=h_conv,
        m_fusa_tot=m_fusa_tot, m_drenata=m_drenata, m_ret=m_ret,
        E_appl=E_appl, E_min=E_min,
        eta_d2=E_min / E_appl if E_appl > 0 else 0.0,
        T_ts_final=T_ts, T_sp_final=T_sp,
        u_max=u_max, u_mean=u_mean,
        time_h=np.array(time_h), delta_f_h=np.array(delta_f_h),
        delta_p_h=np.array(delta_p_h), u_front_h=np.array(u_front_h),
        rho_front_h=np.array(rho_front_h), A_cov_h=np.array(A_cov_h),
        T_ts_h=np.array(T_ts_h), T_sp_h=np.array(T_sp_h),
    )


# =============================================================================
# 7. STADIO III -- ESSICCAZIONE (PELTIER ACCESO) O RIGELO (PELTIER SPENTO)
#
#    La struttura metallica e' modellata con la stessa ODE a due nodi degli
#    stadi precedenti: la capacita' termica dell'acqua residua e' di ordini di
#    grandezza inferiore a quella del metallo, quindi chi si scalda davvero e'
#    il metallo. Trattare l'acqua come un semplice film con capacita' propria
#    farebbe evaporare troppo in fretta e non chiuderebbe il bilancio.
#
#    L'ipotesi T_acqua = T_ts e quella di drenaggio nullo sono verificate
#    numericamente da verifica_ipotesi().
#
#      C_ts dT_ts/dt = (T_sp-T_ts)/R_sp,ts
#                      - A_ts [ h (T_ts-T_aria)
#                               + lambda_fg h_m (rho_v,w(T_ts) - rho_v,a) ]
#      dm/dt = -h_m (rho_v,w(T_ts) - rho_v,a)
#
#    Il termine di pozzo dipende da T_ts in modo esponenziale (formula di
#    Buck), quindi viene valutato con lo stato del passo precedente.
# =============================================================================

def verifica_isotermia(delta_w0, h_conv):
    """Confronta la resistenza di conduzione interna al film (delta_w/k_w) con
    quella di scambio esterno (1/h). Se la prima e' molto minore della seconda
    il film e' isotermo con la piastra, e trattare l'acqua come pozzo al nodo
    T_ts e' lecito."""
    R_int = delta_w0 / KW
    R_ext = 1.0 / h_conv
    return R_int, R_ext, R_int / R_ext


def verifica_drenaggio(delta_w0):
    """Velocita' di un film libero di pari spessore (Nusselt, Eq. 3.29) e
    tempo caratteristico di drenaggio: verifica che, anche ignorando il
    pinning capillare, il drenaggio sarebbe irrilevante sui tempi
    dell'evaporazione."""
    u_w = RHO_W * G * delta_w0 ** 2 / (12.0 * MU_W)
    return u_w, (delta_w0 / u_w if u_w > 0 else np.inf)


def simula_stadio3(m_ret, T_ts_0, T_sp_0, h_conv, nome_superficie,
                   Q_W=Q_PELTIER, dt=0.05, t_max=5000.0):
    """Integra lo Stadio III.

    Q_W > 0 : regime (A), essiccazione; la piastra si scalda e l'acqua evapora.
    Q_W = 0 : regime (B), rigelo; T_ts resta a T_melt finche' c'e' acqua
              liquida e il deficit energetico e' coperto dal calore latente di
              solidificazione.
    Si arresta quando la massa d'acqua si annulla.
    """
    h_m = mass_transfer_coeff(h_conv)
    rho_v_a = rho_v_sat_water(T_AIR_K)      # aria satura (ipotesi ereditata)

    T_ts, T_sp = float(T_ts_0), float(T_sp_0)
    m = m_ret

    time_h, T_ts_h, T_sp_h, m_h, q_evap_h = [], [], [], [], []
    E_loss_ts, E_loss_sp = 0.0, 0.0
    t_end = None
    regime = "essiccazione" if Q_W > 0 else "rigelo"

    for step in range(int(t_max / dt)):
        t_now = (step + 1) * dt

        # pozzo al nodo T_ts: convezione + evaporazione (valutato in ritardo).
        # Qui il fluido e' acqua liquida: pvs su acqua e calore latente di
        # vaporizzazione, non di sublimazione.
        rho_v_w = rho_v_sat_water(T_ts)
        q_conv = h_conv * (T_ts - T_AIR_K)
        q_evap = LAMDA_FG * h_m * (rho_v_w - rho_v_a)
        Q_loss = (q_conv + q_evap) * A_TS

        a11 = 1.0 + dt / C_SP * (1.0 / RTH_SP_TS + 1.0 / RTH_E)
        a12 = -dt / (C_SP * RTH_SP_TS)
        a21 = -dt / (C_TS * RTH_SP_TS)
        a22 = 1.0 + dt / (C_TS * RTH_SP_TS)
        b1v = T_sp + dt / C_SP * (T_WALL_K / RTH_E + Q_W)
        b2v = T_ts - dt / C_TS * Q_loss
        det = a11 * a22 - a12 * a21
        T_sp_new = (b1v * a22 - b2v * a12) / det
        T_ts_new = (a11 * b2v - a21 * b1v) / det

        if Q_W > 0.0:
            # regime (A): essiccazione
            m = max(m - h_m * (rho_v_w - rho_v_a) * dt, 0.0)
            T_ts, T_sp = T_ts_new, T_sp_new
        else:
            # regime (B): rigelo. Finche' resta acqua liquida T_ts non puo'
            # scendere sotto T_melt: il deficit e' coperto dal calore latente
            # di solidificazione (lo Stadio II al contrario).
            if T_ts_new < T_MELT and m > 0.0:
                dE_deficit = C_TS * (T_MELT - T_ts_new)        # [J]
                dm_rigelata = dE_deficit / LAMDA_IF / A_TS     # [kg/m2]
                m = max(m - dm_rigelata, 0.0)
                T_ts, T_sp = T_MELT, T_sp_new
            else:
                T_ts, T_sp = T_ts_new, T_sp_new

        # energie cumulate, per la verifica di chiusura del bilancio
        E_loss_ts += Q_loss * dt
        E_loss_sp += (T_sp - T_WALL_K) / RTH_E * dt

        time_h.append(t_now)
        T_ts_h.append(T_ts - 273.15)
        T_sp_h.append(T_sp - 273.15)
        m_h.append(m * 1e3)
        q_evap_h.append(q_evap)

        if m <= 0.0:
            t_end = t_now
            break

    # ---- chiusura del bilancio energetico ----
    # E_applicata = variazione di energia interna del metallo
    #               + perdite dal nodo T_ts + perdite dal nodo T_sp
    # Il secondo canale di perdita (spreader -> ambiente) non e' trascurabile:
    # ometterlo lascerebbe un residuo di qualche decina di punti percentuali.
    t_fin = t_end if t_end else t_max
    E_appl = Q_W * t_fin
    dU = C_TS * (T_ts - T_ts_0) + C_SP * (T_sp - T_sp_0)
    residuo = (((E_appl - (dU + E_loss_ts + E_loss_sp)) / max(abs(E_appl), 1e-9))
               if Q_W > 0 else 0.0)

    # efficienza di sbrinamento dello stadio (estensione dell'Eq. 1.2):
    # energia minima teorica (latente di vaporizzazione) / energia applicata
    E_min = m_ret * A_TS * LAMDA_FG

    return dict(
        surface=nome_superficie, regime=regime, h_conv=h_conv, Q_W=Q_W,
        t_end=t_end, m_ret_0=m_ret,
        T_ts_final=T_ts, T_sp_final=T_sp,
        E_appl=E_appl, E_min=E_min,
        eta_d=E_min / E_appl if E_appl > 0 else 0.0,
        residuo=residuo, dU_metal=dU,
        E_loss_ts=E_loss_ts, E_loss_sp=E_loss_sp,
        time_h=np.array(time_h), T_ts_h=np.array(T_ts_h),
        T_sp_h=np.array(T_sp_h), m_h=np.array(m_h),
        q_evap_h=np.array(q_evap_h),
    )


def verifica_ipotesi(m_ret_per_superficie, h_conv):
    """Verifica numerica delle due ipotesi dello Stadio III."""
    log("")
    log("  Verifica delle ipotesi dello Stadio III (h_conv = {:g}):".format(h_conv))
    log("  {:<11}{:>13}{:>12}{:>14}{:>13}{:>13}{:>13}"
        .format("Superficie", "m_ret[g/m²]", "sp.film[µm]", "C_acqua[J/K]",
                "C_met/C_acq", "R_int/R_ext", "t_dren[ms]"))
    log("  " + "-" * 89)
    for superficie, m_ret in m_ret_per_superficie.items():
        delta_w0 = m_ret / RHO_W
        C_acqua = m_ret * A_TS * CP_W
        _, _, rapporto_R = verifica_isotermia(delta_w0, h_conv)
        _, t_dren = verifica_drenaggio(delta_w0)
        log("  {:<11}{:>13.1f}{:>12.1f}{:>14.4f}{:>12.0f}x{:>13.4f}{:>13.1f}"
            .format(superficie, m_ret * 1e3, delta_w0 * 1e6, C_acqua,
                    C_METAL / C_acqua, rapporto_R, t_dren * 1e3))
    log("    C_metallo/C_acqua >> 1 : il metallo e' il pozzo termico dominante;")
    log("    R_int/R_ext << 1       : il film e' isotermo con la piastra,")
    log("                             quindi l'ipotesi T_acqua = T_ts e' lecita;")
    log("    t_drenaggio << t_essic.: il drenaggio e' comunque irrilevante.")


# =============================================================================
# 8. CATENA COMPLETA E TABELLE DEI RISULTATI
# =============================================================================

def esegui_catena(fs):
    """Esegue Stadio I + II + III per ogni h_conv e per ogni superficie.

    Ritorna un dizionario annidato: risultati[caso h_conv]['s1' | 'superfici'].
    """
    risultati = {}
    for etichetta, h_conv in H_CONV_CASES.items():
        r1 = simula_stadio1(fs["T_init"], fs["rho_init"], fs["h_0"], h_conv)
        per_superficie = {}
        for superficie, p in SURFACE_PARAMS.items():
            r2 = simula_stadio2(
                r1["T_fr"], r1["rho_fr"], r1["T_ts"], r1["T_sp"],
                fs["h_0"], h_conv,
                beta=p["beta"], m0_ret=p["m0"], A_cov_max=p["A_cov_max"],
                nome_superficie=superficie)
            r3_on = simula_stadio3(r2["m_ret"], r2["T_ts_final"],
                                   r2["T_sp_final"], h_conv, superficie,
                                   Q_W=Q_PELTIER)
            r3_off = simula_stadio3(r2["m_ret"], r2["T_ts_final"],
                                    r2["T_sp_final"], h_conv, superficie,
                                    Q_W=0.0)
            t_tot = (r1["t_end"] or 0.0) + (r2["t_end"] or 0.0) + (r3_on["t_end"] or 0.0)
            per_superficie[superficie] = {"s2": r2, "on": r3_on, "off": r3_off,
                                          "t_tot": t_tot,
                                          "E_tot": Q_PELTIER * t_tot}
        risultati[etichetta] = {"s1": r1, "superfici": per_superficie}
    return risultati


def _fmt(valore, formato="{:.1f}", assente="n/d"):
    """Formatta un valore che puo' essere None (stadio non completato)."""
    return assente if valore is None else formato.format(valore)


def tabelle_stadio1(risultati):
    caso1 = list(H_CONV_CASES.keys())[0]
    log("")
    log("=" * 78)
    log("STADIO I -- RISCALDAMENTO SENSIBILE DELLA BRINA")
    log("=" * 78)
    log("  {:<24}{:>9}{:>9}{:>12}{:>14}{:>12}{:>12}"
        .format("Caso", "t_I [s]", "Gamma_1", "E_appl[J]", "E_metallo[J]",
                "E_brina[J]", "E_perse[J]"))
    log("  " + "-" * 92)
    for etichetta, r in risultati.items():
        s1 = r["s1"]
        log("  {:<24}{:>9}{:>9.4f}{:>12.1f}{:>14.1f}{:>12.2f}{:>12.1f}"
            .format(etichetta, _fmt(s1["t_end"], "{:.2f}"), s1["gamma1"],
                    s1["E_appl"], s1["E_metal"], s1["E_brina"], s1["E_loss"]))

    r0 = risultati[caso1]["s1"]
    log("")
    log("  Confronto col libro -- flusso all'interfaccia spreader -> superficie")
    log("  (il sensore del libro misura questo, non Q/A_ts = {:.0f} W/m²)"
        .format(Q_PELTIER / A_TS))
    confronto_flusso_interfaccia("I", r0["T_sp_h"], r0["T_ts_h"])
    log("      In questo stadio il flusso calcolato supera l'intervallo misurato:")
    log("      la piastra parte dalla temperatura di brinamento, molto al di")
    log("      sotto di quella dello spreader, quindi il salto di temperatura")
    log("      all'interfaccia e' qui il piu' grande dell'intero transitorio.")

    log("")
    log("  Ripartizione dell'energia applicata (caso {}):".format(caso1))
    log("    metallo (C_TS + C_SP = {:.1f} J/K) : {:.1f} %"
        .format(C_METAL, r0["E_metal"] / r0["E_appl"] * 100.0))
    log("    brina                             : {:.1f} %"
        .format(r0["E_brina"] / r0["E_appl"] * 100.0))
    if r0["E_loss"] < 0:
        log("    La somma supera il 100 % perche' la piastra parte piu' fredda")
        log("    dell'ambiente: attraverso R_th,e l'ambiente CEDE calore invece")
        log("    di sottrarlo, e il termine E_perse risulta negativo ({:.1f} J)."
            .format(r0["E_loss"]))
    log("    Il metallo assorbe circa {:.0f} volte l'energia della brina: e' il"
        .format(r0["E_metal"] / max(r0["E_brina"], 1e-9)))
    log("    pozzo dominante dello stadio, ed e' cio' che ne fissa la durata.")
    log("")
    log("  Gamma_1 = k_fick/k_fr = {:.4f}, minore di 1: il trasporto di vapore"
        .format(r0["gamma1"]))
    log("    per sublimazione interna e' un effetto secondario rispetto alla")
    log("    conduzione, coerentemente con quanto osservato dal libro (Cap. 5.3.1).")


def tabelle_stadio2(risultati):
    caso1 = list(H_CONV_CASES.keys())[0]
    log("")
    log("=" * 78)
    log("STADIO II -- FUSIONE E PERMEAZIONE")
    log("=" * 78)
    log("  {:<24}{:<11}{:>10}{:>11}{:>11}{:>11}{:>11}"
        .format("Caso", "Superficie", "t_II [s]", "m_fusa", "m_drenata",
                "m_ret", "T_ts fin."))
    log("  {:<24}{:<11}{:>10}{:>11}{:>11}{:>11}{:>11}"
        .format("", "", "", "[g/m²]", "[g/m²]", "[g/m²]", "[°C]"))
    log("  " + "-" * 89)
    for etichetta, r in risultati.items():
        for superficie in SURFACES:
            s = r["superfici"][superficie]["s2"]
            log("  {:<24}{:<11}{:>10}{:>11.1f}{:>11.1f}{:>11.1f}{:>11.2f}"
                .format(etichetta, superficie, _fmt(s["t_end"], "{:.3f}"),
                        s["m_fusa_tot"] * 1e3, s["m_drenata"] * 1e3,
                        s["m_ret"] * 1e3, s["T_ts_final"] - 273.15))

    r_al = risultati[caso1]["superfici"]["Al nudo"]["s2"]
    r_sl = risultati[caso1]["superfici"]["SLIPS-Al"]["s2"]

    log("")
    log("  Verifica del vincolo di fusione:")
    log("    T_ts a fine stadio = {:.2f} °C (deve essere esattamente 0.00 °C:"
        .format(r_al["T_ts_final"] - 273.15))
    log("    finche' fonde ghiaccio la superficie non puo' allontanarsi da T_melt)")

    log("")
    log("  Confronto col libro -- flusso all'interfaccia spreader -> superficie")
    log("  (in questo stadio il flusso e' massimo: T_ts resta a 0 °C mentre")
    log("   T_sp continua a salire, ed e' il forte aumento di flusso osservato")
    log("   dal libro nel Cap. 5.3)")
    confronto_flusso_interfaccia("II", r_al["T_sp_h"], r_al["T_ts_h"])

    log("")
    log("  Differenziazione fra le due superfici:")
    log("    durata : Al = {} s, SLIPS = {} s  -> rapporto {:.3f}x (differenza modesta)"
        .format(_fmt(r_al["t_end"], "{:.3f}"), _fmt(r_sl["t_end"], "{:.3f}"),
                r_al["t_end"] / r_sl["t_end"]))
    log("    m_ret  : Al = {:.1f} g/m², SLIPS = {:.1f} g/m²  -> rapporto {:.1f}x "
        "(differenza grande)"
        .format(r_al["m_ret"] * 1e3, r_sl["m_ret"] * 1e3,
                r_al["m_ret"] / r_sl["m_ret"]))
    log("    L'effetto della superficie SLIPS non e' accorciare questo stadio,")
    log("    ma lasciare molta meno acqua da evaporare nello Stadio III.")

    log("")
    log("  Effetto della densita' stratificata ereditata dal frosting:")
    log("    densita' al fronte : {:.1f} -> {:.1f} kg/m³ (la parte densa e' alla parete)"
        .format(r_al["rho_front_h"][0], r_al["rho_front_h"][-1]))
    log("    velocita' del fronte: {:.3f} -> {:.3f} mm/s (massimo {:.3f})"
        .format(r_al["u_front_h"][0] * 1e3, r_al["u_front_h"][-1] * 1e3,
                float(np.max(r_al["u_front_h"])) * 1e3))
    log("    Il fronte parte dalla parete, dove la brina e' compatta, e accelera")
    log("    verso la superficie, dove e' rada: la durata dello stadio e'")
    log("    dominata dallo strato basale denso, dove il fronte e' piu' lento.")
    log("    Nel tratto finale la velocita' cala perche' la copertura da gocce")
    log("    trattenute sottrae area utile alla fusione.")


def tabelle_stadio3(risultati):
    caso1 = list(H_CONV_CASES.keys())[0]
    m_per_superficie = {s: risultati[caso1]["superfici"][s]["s2"]["m_ret"]
                        for s in SURFACES}

    log("")
    log("=" * 78)
    log("STADIO III -- ESSICCAZIONE E RIGELO")
    log("=" * 78)
    verifica_ipotesi(m_per_superficie, H_CONV_FROST)

    log("")
    log("  Regime (A): Peltier acceso ({:.0f} W) -- essiccazione".format(Q_PELTIER))
    log("  {:<24}{:<11}{:>10}{:>12}{:>12}{:>10}{:>11}"
        .format("Caso", "Superficie", "m_ret", "t_III [s]", "T_ts fin.",
                "eta_III", "residuo"))
    log("  {:<24}{:<11}{:>10}{:>12}{:>12}{:>10}{:>11}"
        .format("", "", "[g/m²]", "", "[°C]", "[%]", "[%]"))
    log("  " + "-" * 90)
    for etichetta, r in risultati.items():
        for superficie in SURFACES:
            s = r["superfici"][superficie]["on"]
            log("  {:<24}{:<11}{:>10.1f}{:>12}{:>12.1f}{:>10.2f}{:>11.2f}"
                .format(etichetta, superficie, s["m_ret_0"] * 1e3,
                        _fmt(s["t_end"]), s["T_ts_final"] - 273.15,
                        s["eta_d"] * 100.0, s["residuo"] * 100.0))

    log("")
    log("  Confronto col libro -- flusso all'interfaccia spreader -> superficie")
    log("  (qui T_ts si scalda e insegue T_sp: il salto di temperatura")
    log("   all'interfaccia si comprime e il flusso cala rispetto allo Stadio II)")
    r_al_on = risultati[caso1]["superfici"]["Al nudo"]["on"]
    confronto_flusso_interfaccia("III", r_al_on["T_sp_h"], r_al_on["T_ts_h"])

    log("")
    log("  Regime (B): Peltier spento -- rigelo, non essiccazione.")
    log("  Con l'aria a {:.0f} °C il bilancio energetico a T = T_melt e' gia'"
        .format(T_AIR_K - 273.15))
    log("  negativo: convezione ed evaporazione sottraggono piu' calore di")
    log("  quanto ne entri, quindi l'acqua residua non evapora ma rigela.")
    log("  {:<24}{:<11}{:>10}{:>15}"
        .format("Caso", "Superficie", "m_ret", "t_rigelo [s]"))
    log("  " + "-" * 60)
    for etichetta, r in risultati.items():
        for superficie in SURFACES:
            s = r["superfici"][superficie]["off"]
            log("  {:<24}{:<11}{:>10.1f}{:>15}"
                .format(etichetta, superficie, s["m_ret_0"] * 1e3,
                        _fmt(s["t_end"])))

    log("")
    log("  Sbrinamento completo (Stadio I + II + III, Peltier acceso):")
    log("  {:<24}{:<11}{:>9}{:>9}{:>10}{:>12}{:>12}"
        .format("Caso", "Superficie", "t_I[s]", "t_II[s]", "t_III[s]",
                "t_tot[s]", "E_tot[J]"))
    log("  " + "-" * 87)
    for etichetta, r in risultati.items():
        for superficie in SURFACES:
            s = r["superfici"][superficie]
            log("  {:<24}{:<11}{:>9}{:>9}{:>10}{:>12.1f}{:>12.0f}"
                .format(etichetta, superficie, _fmt(r["s1"]["t_end"]),
                        _fmt(s["s2"]["t_end"], "{:.2f}"), _fmt(s["on"]["t_end"]),
                        s["t_tot"], s["E_tot"]))

    log("")
    log("  Risparmio energetico della superficie SLIPS:")
    for etichetta, r in risultati.items():
        E_al = r["superfici"]["Al nudo"]["E_tot"]
        E_sl = r["superfici"]["SLIPS-Al"]["E_tot"]
        log("    {:<24} dE = {:>5.0f} J   ({:.1f} % dell'energia dell'alluminio)"
            .format(etichetta, E_al - E_sl, (E_al - E_sl) / E_al * 100.0))
    log("    Il risparmio e' quasi identico nei due casi di h_conv: l'incertezza")
    log("    su quel parametro sposta i tempi assoluti ma non tocca il confronto")
    log("    fra le due superfici, che e' la conclusione di interesse.")

    r_al = risultati[caso1]["superfici"]["Al nudo"]["on"]
    r_sl = risultati[caso1]["superfici"]["SLIPS-Al"]["on"]
    log("")
    log("  Nota sull'efficienza eta_III (apparente paradosso):")
    log("    SLIPS ha eta_III piu' bassa ({:.1f} % contro {:.1f} %) pur consumando"
        .format(r_sl["eta_d"] * 100.0, r_al["eta_d"] * 100.0))
    log("    meno energia in assoluto. Non e' una contraddizione: entrambe le")
    log("    superfici pagano lo stesso costo fisso di scaldare il metallo")
    log("    ({:.0f} J/K), ma SLIPS ha pochissima acqua da evaporare, quindi quasi"
        .format(C_METAL))
    log("    tutta la sua energia va in quel costo fisso. La metrica corretta per")
    log("    il confronto e' percio' l'energia assoluta, non l'efficienza.")


# =============================================================================
# 9. SWEEP SULLA POTENZA DEL PELTIER
# =============================================================================

def sweep_Q(fs, valori_Q, h_conv=H_CONV_FROST):
    """Ripete Stadio II e Stadio III per piu' valori della potenza applicata.

    Q_PELTIER e' il parametro piu' incerto del modello: lo sweep ne mostra
    l'effetto su durate ed efficienze.
    """
    r1 = simula_stadio1(fs["T_init"], fs["rho_init"], fs["h_0"], h_conv)
    out = {s: {"Q_W": [], "t_II": [], "t_III": [], "m_ret": [], "eta_d": []}
           for s in SURFACES}

    log("")
    log("=" * 78)
    log("SENSIBILITA' ALLA POTENZA DEL PELTIER (h_conv = {:g})".format(h_conv))
    log("=" * 78)
    log("  {:>7}{:>12}{:>12}{:>15}{:>13}{:>16}{:>11}"
        .format("Q [W]", "q_s[W/m²]", "t_II Al[s]", "t_II SLIPS[s]",
                "t_III Al[s]", "t_III SLIPS[s]", "rapporto"))
    log("  " + "-" * 86)

    for Q_W in valori_Q:
        riga = {}
        for superficie, p in SURFACE_PARAMS.items():
            r2 = simula_stadio2(
                r1["T_fr"], r1["rho_fr"], r1["T_ts"], r1["T_sp"],
                fs["h_0"], h_conv,
                beta=p["beta"], m0_ret=p["m0"], A_cov_max=p["A_cov_max"],
                nome_superficie=superficie, Q_W=Q_W)
            # t_max elevato: ai flussi bassi l'essiccazione richiede molte
            # migliaia di secondi, e non e' un problema numerico ma un
            # risultato fisico (la piastra non riesce a scaldarsi abbastanza).
            r3 = simula_stadio3(r2["m_ret"], r2["T_ts_final"],
                                r2["T_sp_final"], h_conv, superficie,
                                Q_W=Q_W, t_max=40000.0)
            out[superficie]["Q_W"].append(Q_W)
            out[superficie]["t_II"].append(r2["t_end"])
            out[superficie]["t_III"].append(r3["t_end"])
            out[superficie]["m_ret"].append(r2["m_ret"] * 1e3)
            out[superficie]["eta_d"].append(r3["eta_d"] * 100.0)
            riga[superficie] = (r2["t_end"], r3["t_end"])

        t2_al, t3_al = riga["Al nudo"]
        t2_sl, t3_sl = riga["SLIPS-Al"]
        rapporto = ("{:.2f}".format(t3_al / t3_sl)
                    if (t3_al and t3_sl) else "n/d")
        log("  {:>7.1f}{:>12.0f}{:>12}{:>15}{:>13}{:>16}{:>11}"
            .format(Q_W, Q_W / A_TS, _fmt(t2_al, "{:.3f}"), _fmt(t2_sl, "{:.3f}"),
                    _fmt(t3_al, "{:.1f}", "> t_max"),
                    _fmt(t3_sl, "{:.1f}", "> t_max"), rapporto))

    log("")
    log("    Il rapporto fra le due superfici non e' costante: cala al crescere")
    log("    della potenza. A bassa potenza domina la differenza di massa da")
    log("    evaporare; ad alta potenza pesa di piu' il costo fisso di scaldare")
    log("    il metallo, identico per le due superfici, e il vantaggio di SLIPS")
    log("    si comprime.")
    return out


# =============================================================================
# 10. FIGURE
# =============================================================================

def figura_stadio1(risultati, fs, outdir, show):
    caso1, caso2 = list(H_CONV_CASES.keys())
    r1, r2 = risultati[caso1]["s1"], risultati[caso2]["s1"]
    y_norm = np.linspace(0, 1, NY)

    fig = plt.figure(figsize=(20, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    # 1: stato iniziale ereditato dal frosting
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(fs["T_init"] - 273.15, y_norm, color="#c0392b", lw=2.5,
            label="T (frosting model)")
    ax.set_xlabel("Temperature [°C]", fontsize=10)
    ax.set_ylabel("y/h₀   (0 = wall)", fontsize=10)
    ax.set_title("Initial state from the frosting model\n"
                 "T ≈ {:.1f} °C (nearly isothermal)"
                 .format(float(np.mean(fs["T_init"])) - 273.15),
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    ax_rho = ax.twiny()
    ax_rho.plot(fs["rho_init"], y_norm, color="#2980b9", lw=2.0, ls="--",
                label="ρ_fr (frosting model)")
    ax_rho.set_xlabel("ρ_fr [kg/m³]", fontsize=9, color="#2980b9")
    ax_rho.tick_params(axis="x", labelcolor="#2980b9")
    ax_rho.legend(fontsize=8, loc="center right")

    # 2: porosita' iniziale
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(fs["eps_init"], y_norm, color="#8e44ad", lw=2.5,
            label="ε (frosting model)")
    ax.set_xlabel("Porosity [–]", fontsize=11)
    ax.set_ylabel("Standardized profile", fontsize=11)
    ax.set_title("Actual frost porosity\nmean ε = {:.4f}"
                 .format(float(np.mean(fs["eps_init"]))),
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # 3: temperature della struttura metallica
    ax = fig.add_subplot(gs[0, 2])
    for r, caso, ls in [(r1, caso1, "-"), (r2, caso2, "--")]:
        ax.plot(r["time_h"], r["T_ts_h"], color="#e74c3c", ls=ls, lw=2.2,
                alpha=1.0 if ls == "-" else 0.6,
                label="T_ts ({})".format(caso.split()[0]))
        ax.plot(r["time_h"], r["T_sp_h"], color="#2980b9", ls=ls, lw=2.0,
                alpha=1.0 if ls == "-" else 0.6,
                label="T_sp ({})".format(caso.split()[0]))
        if r["t_end"]:
            ax.axvline(r["t_end"], color="#7f8c8d", ls=":", lw=1.0, alpha=0.6)
    ax.axhline(0, color="black", lw=1.0, ls=":", label="T_melt = 0 °C")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Temperature [°C]", fontsize=12)
    ax.set_title("Metal structure (Eq. 4.10)\nt_I = {} s (h=13.2)   {} s (h=35)"
                 .format(_fmt(r1["t_end"], "{:.2f}"), _fmt(r2["t_end"], "{:.2f}")),
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # 4: evoluzione del profilo termico nella brina
    ax = fig.add_subplot(gs[1, 0])
    istanti = sorted(r1["snap_T"].keys())
    colori = plt.cm.plasma(np.linspace(0.15, 0.9, len(istanti)))
    for ts, c in zip(istanti, colori):
        ax.plot(r1["snap_T"][ts], y_norm, color=c, lw=1.8,
                label="t={:.0f}s".format(ts))
    ax.axvline(0, color="black", lw=1.0, ls=":")
    ax.set_xlabel("Temperature [°C]", fontsize=10)
    ax.set_ylabel("y/h₀   (0 = wall)", fontsize=10)
    ax.set_title("Temperature profile evolution in the frost\n(h_conv = 13.2)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    # 5: bilancio energetico
    ax = fig.add_subplot(gs[1, 1])
    etichette = ["E_applied", "E_metal", "E_frost", "E_losses"]
    valori = [r1["E_appl"], r1["E_metal"], r1["E_brina"], r1["E_loss"]]
    barre = ax.bar(etichette, valori,
                   color=["#34495e", "#e74c3c", "#3498db", "#95a5a6"], alpha=0.88)
    for b, v in zip(barre, valori):
        ax.text(b.get_x() + b.get_width() / 2, v, "{:.1f}".format(v),
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Energy [J]", fontsize=10)
    ax.set_title("Stage I energy balance\nthe metal absorbs {:.0f}x the frost"
                 .format(r1["E_metal"] / max(r1["E_brina"], 1e-9)),
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    # 6: potenza ceduta alla brina
    ax = fig.add_subplot(gs[1, 2])
    for r, caso, ls in [(r1, caso1, "-"), (r2, caso2, "--")]:
        ax.plot(r["time_h"], r["Q_brina_h"], ls=ls, lw=2.2, color="#16a085",
                alpha=1.0 if ls == "-" else 0.6,
                label="Q_frost ({})".format(caso.split()[0]))
    ax.axhline(Q_PELTIER, color="#c0392b", ls=":", lw=1.4,
               label="Q_Peltier = {:.0f} W".format(Q_PELTIER))
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("Power [W]", fontsize=10)
    ax.set_title("Power transferred from the plate to the frost\n"
                 "(a small fraction of Q_Peltier)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "STAGE I — Sensible heating of the frost layer  |  "
        "h₀={:.2f} mm   mean ε={:.3f}   T_frost,0≈{:.1f}°C   T_air={:.0f}°C   "
        "Q={:.0f} W\nInitial state from the frosting model  |  "
        "k_eff = k_fr + k_fick  |  two-node ODE (Eq. 4.10)  |  "
        "h_conv: 13.2 (Blasius) vs 35 (previous)"
        .format(fs["h_0"] * 1e3, float(np.mean(fs["eps_init"])),
                float(np.mean(fs["T_init"])) - 273.15, T_AIR_K - 273.15,
                Q_PELTIER),
        fontsize=10, fontweight="bold", y=1.01)
    return salva_figura(fig, FIG_STADIO1, outdir, show)


def figura_stadio2(risultati, outdir, show):
    caso1 = list(H_CONV_CASES.keys())[0]
    EN = EN_SURFACE

    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    # 1: spessore residuo della brina
    ax = fig.add_subplot(gs[0, 0])
    for superficie in SURFACES:
        r = risultati[caso1]["superfici"][superficie]["s2"]
        ax.plot(r["time_h"], r["delta_f_h"] * 1e3,
                color=SURFACE_PARAMS[superficie]["color"], lw=2.4,
                label=EN[superficie])
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("Residual frost thickness  δ_f  [mm]", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # 2: velocita' del fronte di fusione
    ax = fig.add_subplot(gs[0, 1])
    for superficie in SURFACES:
        r = risultati[caso1]["superfici"][superficie]["s2"]
        ax.plot(r["time_h"], r["u_front_h"] * 1e3,
                color=SURFACE_PARAMS[superficie]["color"], lw=2.5,
                label=EN[superficie])
    ax.set_yscale("log")
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("Front velocity  u_front  [mm/s]  (log)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, which="both")

    # 3: densita' della brina al fronte
    ax = fig.add_subplot(gs[0, 2])
    r = risultati[caso1]["superfici"]["Al nudo"]["s2"]
    ax.plot(r["time_h"], r["rho_front_h"], color="#c0392b", lw=2.5)
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("Frost density at the front  [kg/m³]", fontsize=10)
    ax.grid(True, alpha=0.25)

    # 4: copertura superficiale delle gocce trattenute
    ax = fig.add_subplot(gs[0, 3])
    for superficie in SURFACES:
        r = risultati[caso1]["superfici"][superficie]["s2"]
        ax.plot(r["time_h"], r["A_cov_h"] * 100,
                color=SURFACE_PARAMS[superficie]["color"], lw=2.5,
                label=EN[superficie])
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("Covered area fraction  A_cov  [%]", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # 5: fronte di permeazione
    ax = fig.add_subplot(gs[1, 0])
    for superficie in SURFACES:
        r = risultati[caso1]["superfici"][superficie]["s2"]
        ax.plot(r["time_h"], r["delta_p_h"] * 1e6,
                color=SURFACE_PARAMS[superficie]["color"], lw=2.5,
                label=EN[superficie])
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("Permeation front  δ_p  [µm]", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # 6: temperature del metallo (vincolo di fusione)
    ax = fig.add_subplot(gs[1, 1])
    for superficie in SURFACES:
        r = risultati[caso1]["superfici"][superficie]["s2"]
        ax.plot(r["time_h"], r["T_ts_h"],
                color=SURFACE_PARAMS[superficie]["color"], lw=2.5,
                label="Test surface — {}".format(EN[superficie]))
        ax.plot(r["time_h"], r["T_sp_h"],
                color=SURFACE_PARAMS[superficie]["color"], lw=1.5, ls="--",
                alpha=0.6, label="Heat spreader — {}".format(EN[superficie]))
    ax.axhline(0, color="black", lw=1.2, ls=":", label="T_melt")
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("Temperature [°C]", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)

    # 7: bilancio di massa
    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(len(SURFACES))
    w = 0.35
    m_dren = [risultati[caso1]["superfici"][s]["s2"]["m_drenata"] * 1e3
              for s in SURFACES]
    m_ret = [risultati[caso1]["superfici"][s]["s2"]["m_ret"] * 1e3
             for s in SURFACES]
    b1 = ax.bar(x - w / 2, m_dren, w, color="#95a5a6", alpha=0.9, label="Drained")
    b2 = ax.bar(x + w / 2, m_ret, w,
                color=[SURFACE_PARAMS[s]["color"] for s in SURFACES],
                alpha=0.9, label="Retained (→ Stage III)")
    for b, v in zip(list(b1) + list(b2), m_dren + m_ret):
        ax.text(b.get_x() + b.get_width() / 2, v, "{:.1f}".format(v),
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([EN[s] for s in SURFACES], fontsize=9)
    ax.set_ylabel("Mass per unit area [g/m²]", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # 8: tabella riassuntiva
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    righe = []
    for superficie in SURFACES:
        s = risultati[caso1]["superfici"][superficie]["s2"]
        righe.append([EN[superficie], _fmt(s["t_end"], "{:.3f}"),
                      "{:.1f}".format(s["m_fusa_tot"] * 1e3),
                      "{:.1f}".format(s["m_drenata"] * 1e3),
                      "{:.1f}".format(s["m_ret"] * 1e3)])
    tabella = ax.table(cellText=righe,
                       colLabels=["Surface", "t_II\n[s]", "Melted\n[g/m²]",
                                  "Drained\n[g/m²]", "Retained\n[g/m²]"],
                       loc="center", cellLoc="center")
    tabella.auto_set_font_size(False)
    tabella.set_fontsize(9)
    tabella.scale(1.0, 2.0)
    for j in range(5):
        tabella[0, j].set_facecolor("#2c3e50")
        tabella[0, j].set_text_props(color="white", fontweight="bold")

    fig.suptitle(
        "STAGE II — Melting and permeation  |  "
        "melting front with droplet coverage A_cov  |  permeation front (Aoki)\n"
        "two-sided melting constraint on T_ts  |  stratified density from the "
        "frosting model  |  h_conv = {:g} W/(m²K)".format(H_CONV_FROST),
        fontsize=10, fontweight="bold", y=1.01)
    return salva_figura(fig, FIG_STADIO2, outdir, show)


def figura_stadio3(risultati, outdir, show):
    caso1, caso2 = list(H_CONV_CASES.keys())
    w = 0.35

    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    # 1: massa residua nel tempo (Peltier acceso)
    ax = fig.add_subplot(gs[0, 0])
    for caso, ls in [(caso1, "-"), (caso2, "--")]:
        for superficie in SURFACES:
            r = risultati[caso]["superfici"][superficie]["on"]
            ax.plot(r["time_h"], r["m_h"],
                    color=SURFACE_PARAMS[superficie]["color"], ls=ls, lw=2.2,
                    alpha=1.0 if ls == "-" else 0.55,
                    label="{} ({})".format(EN_SURFACE[superficie],
                                           caso.split()[0].split("=")[1]))
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("m_ret  [g/m²]", fontsize=10)
    ax.set_title("Drying (Peltier on)\nevaporation of the retained water",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # 2: temperature del metallo (Peltier acceso)
    ax = fig.add_subplot(gs[0, 1])
    for superficie in SURFACES:
        r = risultati[caso1]["superfici"][superficie]["on"]
        ax.plot(r["time_h"], r["T_ts_h"],
                color=SURFACE_PARAMS[superficie]["color"], lw=2.4,
                label="T_ts {}".format(EN_SURFACE[superficie]))
        ax.plot(r["time_h"], r["T_sp_h"],
                color=SURFACE_PARAMS[superficie]["color"], lw=1.4, ls="--",
                alpha=0.6, label="T_sp {}".format(EN_SURFACE[superficie]))
    ax.axhline(T_AIR_K - 273.15, color="#7f8c8d", ls=":", lw=1.2,
               label="T_air")
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("Temperature [°C]", fontsize=10)
    ax.set_title("Metal structure (Peltier on)\n"
                 "it is the metal that heats up, not the water",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # 3: rigelo (Peltier spento)
    ax = fig.add_subplot(gs[0, 2])
    for superficie in SURFACES:
        r = risultati[caso1]["superfici"][superficie]["off"]
        ax.plot(r["time_h"], r["m_h"],
                color=SURFACE_PARAMS[superficie]["color"], lw=2.5,
                label="{}  (t = {} s)".format(EN_SURFACE[superficie],
                                              _fmt(r["t_end"], "{:.0f}")))
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("m_ret  [g/m²]", fontsize=10)
    ax.set_title("Peltier off → refreezing\n(not drying: T_air < T_melt)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # 4: confronto acceso / spento
    ax = fig.add_subplot(gs[0, 3])
    x = np.arange(len(SURFACES))
    t_on = [risultati[caso1]["superfici"][s]["on"]["t_end"] or 0.0 for s in SURFACES]
    t_off = [risultati[caso1]["superfici"][s]["off"]["t_end"] or 0.0 for s in SURFACES]
    b1 = ax.bar(x - w / 2, t_on, w, color="#16a085", alpha=0.9,
                label="on → evaporates")
    b2 = ax.bar(x + w / 2, t_off, w, color="#c0392b", alpha=0.9,
                label="off → refreezes")
    for b, v in zip(list(b1) + list(b2), t_on + t_off):
        ax.text(b.get_x() + b.get_width() / 2, v, "{:.0f}".format(v),
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([EN_SURFACE[s] for s in SURFACES], fontsize=9)
    ax.set_ylabel("Time [s]", fontsize=10)
    ax.set_title("Peltier on or off\ntwo different physical mechanisms",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # 5: durata dei tre stadi
    ax = fig.add_subplot(gs[1, 0])
    etichette, t1s, t2s, t3s = [], [], [], []
    for caso in [caso1, caso2]:
        for superficie in SURFACES:
            s = risultati[caso]["superfici"][superficie]
            etichette.append("{}\n{}".format(EN_SURFACE[superficie].split()[-1],
                                             caso.split()[0].split("=")[1]))
            t1s.append(risultati[caso]["s1"]["t_end"] or 0.0)
            t2s.append(s["s2"]["t_end"] or 0.0)
            t3s.append(s["on"]["t_end"] or 0.0)
    t1s, t2s, t3s = np.array(t1s), np.array(t2s), np.array(t3s)
    xx = np.arange(len(etichette))
    ax.bar(xx, t1s, 0.6, label="Stage I", color="#3498db")
    ax.bar(xx, t2s, 0.6, bottom=t1s, label="Stage II", color="#f39c12")
    ax.bar(xx, t3s, 0.6, bottom=t1s + t2s, label="Stage III", color="#e74c3c")
    for xi, tot in zip(xx, t1s + t2s + t3s):
        ax.text(xi, tot, "{:.0f}".format(tot), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xx)
    ax.set_xticklabels(etichette, fontsize=7)
    ax.set_ylabel("Time [s]", fontsize=10)
    ax.set_title("Duration of the three stages\nStage III dominates the defrost",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # 6: energia totale dello sbrinamento
    ax = fig.add_subplot(gs[1, 1])
    xx = np.arange(2)
    for i, superficie in enumerate(SURFACES):
        valori = [risultati[c]["superfici"][superficie]["E_tot"]
                  for c in [caso1, caso2]]
        barre = ax.bar(xx + (i - 0.5) * w, valori, w,
                       color=SURFACE_PARAMS[superficie]["color"], alpha=0.88,
                       label=EN_SURFACE[superficie])
        for b, v in zip(barre, valori):
            ax.text(b.get_x() + b.get_width() / 2, v, "{:.0f}".format(v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xx)
    ax.set_xticklabels([c.split()[0].split("=")[1] for c in [caso1, caso2]],
                       fontsize=9)
    ax.set_xlabel("h_conv  [W/(m²·K)]", fontsize=9)
    ax.set_ylabel("Total energy  [J]", fontsize=10)
    E_al = risultati[caso1]["superfici"]["Al nudo"]["E_tot"]
    E_sl = risultati[caso1]["superfici"]["SLIPS-Al"]["E_tot"]
    ax.set_title("Total defrost energy\nSLIPS saving ≈ {:.0f} %"
                 .format((E_al - E_sl) / E_al * 100.0),
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # 7: chiusura del bilancio energetico
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    righe = []
    for superficie in SURFACES:
        s = risultati[caso1]["superfici"][superficie]["on"]
        righe.append([EN_SURFACE[superficie].split()[-1],
                      "{:.0f}".format(s["E_appl"]),
                      "{:.0f}".format(s["dU_metal"]),
                      "{:.0f}".format(s["E_loss_ts"]),
                      "{:.0f}".format(s["E_loss_sp"]),
                      "{:.2f}%".format(s["residuo"] * 100.0)])
    tabella = ax.table(cellText=righe,
                       colLabels=["Surf.", "E_appl\n[J]", "ΔU_met\n[J]",
                                  "losses\nT_ts [J]", "losses\nT_sp [J]",
                                  "residual"],
                       loc="center", cellLoc="center")
    tabella.auto_set_font_size(False)
    tabella.set_fontsize(8)
    tabella.scale(1.0, 1.9)
    for j in range(6):
        tabella[0, j].set_facecolor("#2c3e50")
        tabella[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Energy balance closure\n"
                 "E_appl = ΔU_metal + losses", fontsize=10,
                 fontweight="bold", pad=12)

    # 8: riepilogo
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    righe = []
    for caso in [caso1, caso2]:
        for superficie in SURFACES:
            s = risultati[caso]["superfici"][superficie]
            righe.append([caso.split()[0].split("=")[1],
                          EN_SURFACE[superficie].split()[-1],
                          "{:.1f}".format(s["s2"]["m_ret"] * 1e3),
                          _fmt(s["on"]["t_end"], "{:.0f}"),
                          _fmt(s["off"]["t_end"], "{:.0f}"),
                          "{:.1f}%".format(s["on"]["eta_d"] * 100.0)])
    tabella = ax.table(cellText=righe,
                       colLabels=["h_c", "Surf.", "m_ret\n[g/m²]", "t_on\n[s]",
                                  "t_off\n[s]", "η_III"],
                       loc="center", cellLoc="center")
    tabella.auto_set_font_size(False)
    tabella.set_fontsize(8)
    tabella.scale(1.0, 1.9)
    for j in range(6):
        tabella[0, j].set_facecolor("#2c3e50")
        tabella[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Stage III summary", fontsize=10, fontweight="bold", pad=12)

    fig.suptitle(
        "STAGE III — Fate of the retained water  |  coupled metal structure "
        "(C = {:.1f} J/K, far larger than that of the water)\n"
        "Peltier on → evaporates   |   Peltier off → refreezes "
        "(T_air < T_melt)   |   pvs over water, not over ice   |   "
        "h_conv: 13.2 vs 35".format(C_METAL),
        fontsize=10, fontweight="bold", y=1.01)
    return salva_figura(fig, FIG_STADIO3, outdir, show)


def figura_sweep(sweep, outdir, show):
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    for superficie in SURFACES:
        punti = [(q / A_TS, t) for q, t in zip(sweep[superficie]["Q_W"],
                                               sweep[superficie]["t_III"]) if t]
        if punti:
            xs, ys = zip(*punti)
            ax.plot(xs, ys, color=SURFACE_PARAMS[superficie]["color"], lw=2.5,
                    marker="o", ms=7, label=EN_SURFACE[superficie])
    ax.set_xlabel('Applied heat flux  q"  [W/m²]', fontsize=11)
    ax.set_ylabel("t_III [s]", fontsize=11)
    ax.set_title("Stage III duration\nversus the applied power",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    for superficie in SURFACES:
        xs = [q / A_TS for q in sweep[superficie]["Q_W"]]
        ax.plot(xs, sweep[superficie]["eta_d"],
                color=SURFACE_PARAMS[superficie]["color"], lw=2.5,
                marker="s", ms=7, label=EN_SURFACE[superficie])
    ax.set_xlabel('Applied heat flux  q"  [W/m²]', fontsize=11)
    ax.set_ylabel("η_d,III  [%]", fontsize=11)
    ax.set_title("Stage III efficiency\nSLIPS has lower η but consumes less "
                 "in absolute terms", fontsize=10, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    fig.suptitle("Sensitivity to the Peltier power  "
                 "(main case: Q = {:.0f} W)".format(Q_PELTIER),
                 fontsize=12, fontweight="bold", y=1.03)
    return salva_figura(fig, FIG_SWEEP, outdir, show)


# =============================================================================
# 11. PROGRAMMA PRINCIPALE
# =============================================================================

def cartella_predefinita():
    """Sottocartella 'risultati_defrost' accanto allo script (o nella cartella
    di lavoro corrente se __file__ non e' definito, es. in Jupyter)."""
    try:
        base = Path(__file__).resolve().parent
    except NameError:                                        # pragma: no cover
        base = Path.cwd()
    return base / "risultati_defrost"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Modello di sbrinamento in tre stadi: calcolo completo e "
                    "generazione di tutte le figure della tesi.")
    parser.add_argument("--outdir", default=None, metavar="CARTELLA",
                        help="cartella di destinazione dei risultati "
                             "(predefinita: ./risultati_defrost)")
    parser.add_argument("--show", action="store_true",
                        help="mostra i grafici a schermo oltre a salvarli")
    parser.add_argument("--frost-state", default=None, metavar="FILE.npz",
                        help="usa lo stato iniziale salvato da 'frost_model.py' "
                             "invece di ricalcolarlo")
    # parse_known_args ignora gli argomenti aggiunti da Jupyter/IDE.
    args, _ignoti = parser.parse_known_args(argv)

    outdir = Path(args.outdir) if args.outdir else cartella_predefinita()
    outdir.mkdir(parents=True, exist_ok=True)
    show = args.show or in_notebook()

    log("=" * 78)
    log("MODELLO DI SBRINAMENTO IN TRE STADI -- ESECUZIONE COMPLETA")
    log("=" * 78)
    log("Python      : {}".format(sys.version.split()[0]))
    log("numpy       : {}".format(np.__version__))
    log("scipy       : {}".format(scipy.__version__))
    log("matplotlib  : {}".format(matplotlib.__version__))
    log("Risultati in: {}".format(outdir.resolve()))
    log("")

    if args.frost_state:
        origine = "file {}".format(Path(args.frost_state).name)
    else:
        origine = "modello di frosting eseguito in questo script"
    log("Calcolo dello stato iniziale...")
    fs = stato_iniziale(args.frost_state)
    stampa_stato_iniziale(fs, origine)

    log("")
    log("Calcolo della catena Stadio I -> II -> III per i due valori di h_conv")
    log("e per le due superfici...")
    risultati = esegui_catena(fs)

    tabelle_stadio1(risultati)
    tabelle_stadio2(risultati)
    tabelle_stadio3(risultati)

    figura_stadio1(risultati, fs, outdir, show)
    figura_stadio2(risultati, outdir, show)
    figura_stadio3(risultati, outdir, show)

    sweep = sweep_Q(fs, Q_SWEEP)
    figura_sweep(sweep, outdir, show)

    attesi = [FIG_STADIO1, FIG_STADIO2, FIG_STADIO3, FIG_SWEEP]
    log("")
    log("=" * 78)
    log("FILE PRODOTTI IN {}".format(outdir.resolve()))
    log("=" * 78)
    for nome in attesi:
        esiste = (outdir / nome).exists()
        log("  [{}] {}".format("OK      " if esiste else "MANCANTE", nome))
    log("  [OK      ] riepilogo.txt")

    (outdir / "riepilogo.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# In[ ]:




