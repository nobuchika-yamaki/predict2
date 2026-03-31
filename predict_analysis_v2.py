#!/usr/bin/env python3
"""
predict_analysis_v2.py

Corrected unified simulation for:
  "A Sharp Recoverability Boundary in Delayed Adaptive Feedback Systems"

2D model (correctly implemented from paper equations):
  dθ = [ν + K·sin(e(t-τ))] dt + σ_e dW_e
  dν = -γ·ν dt + σ_ν dW_ν          σ_ν = σ·sqrt(2γ)  [fluctuation-dissipation]
  dK = ε[A - cos(e(t-τ)) - μ(K-K₀)] dt
  e  = wrap(θ_ref - θ),  θ_ref = ω·t

Key fixes vs. original code:
  1. Feedback K·sin(e_del) in dθ, not in dν
  2. θ_ref tracked explicitly in 2D
  3. σ_ν = σ√(2γ) keeps var(ν) constant across γ → enables fair γ comparison
  4. Identical sustained success criterion (T_hold) in 1D and 2D
  5. MU = 0.1 consistent everywhere
  6. T_OBS = 40 s → T_total = T_WARM + T_OBS = 60 s (matches paper)
  7. n_trials = 100 (raised from 50)

Outputs to ~/Desktop/:
  fig1_success_rate_ref.pdf       -- sharp transition, reference condition
  fig2_tau_c_vs_sigma.pdf         -- τ_c vs σ for each ε, with analytical bound
  fig3_1d_slices.pdf              -- S(τ) slices for γ = 0.3, 1.65, 3.0
  fig4_heatmaps.pdf               -- S(τ,γ) and <T_rec>(τ,γ) heatmaps
  fig5_tau_c_gamma.pdf            -- τ_c(γ) with 95% bootstrap CI
  results_1d.csv
  results_2d.csv
  tau_c_by_gamma.csv
  analytical_bounds.csv
  run_metadata.json
"""

import math, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTDIR = Path.home() / "Desktop"
OUTDIR.mkdir(exist_ok=True)

# ── shared parameters ─────────────────────────────────────────────────────────
DT      = 0.01
K0      = 1.0
A_COEF  = 0.5
MU      = 0.1
OMEGA   = 2.0 * math.pi
T_WARM  = 20.0
T_OBS   = 40.0        # T_total = 60 s as stated in paper
E_TH    = 0.05 * math.pi
T_HOLD  = 1.0
N_WARM  = int(round(T_WARM / DT))
N_OBS   = int(round(T_OBS  / DT))
N_HOLD  = int(round(T_HOLD / DT))
SEED    = 42

# ── sweep settings ────────────────────────────────────────────────────────────
N_TRIALS_1D = 100
N_TRIALS_2D = 100
N_BOOT      = 1000

TAUS_1D    = np.round(np.arange(0.00, 2.51, 0.05), 8)
TAUS_2D    = np.round(np.arange(0.00, 3.51, 0.10), 8)

EPS_LIST   = [0.02, 0.05, 0.15]
SIGMA_LIST = [0.0,  0.05, 0.10, 0.15]
DPHI_1D    = 0.5 * math.pi

GAMMA_LIST = np.round(np.linspace(0.3, 3.0, 12), 4)
EPS_2D     = 0.05
SIGMA_2D   = 0.05
DPHI_2D    = 0.5 * math.pi


# ── helpers ───────────────────────────────────────────────────────────────────
def wrap(x):
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def analytical_tau_c(eps):
    K_star = K0 + eps * (A_COEF - 1.0) / MU
    return math.pi / (2.0 * K_star) if K_star > 0 else np.nan


def estimate_tau_c(taus, S):
    for i in range(1, len(taus)):
        s0, s1 = S[i - 1], S[i]
        if (s0 >= 0.5 >= s1) or (s0 <= 0.5 <= s1):
            if s1 == s0:
                return float((taus[i - 1] + taus[i]) / 2.0)
            frac = (0.5 - s0) / (s1 - s0)
            return float(taus[i - 1] + frac * (taus[i] - taus[i - 1]))
    return np.nan


def bootstrap_tau_c(taus, succ_mat, n_boot=1000, seed=0):
    rng_b = np.random.default_rng(seed)
    n_tau, n_t = succ_mat.shape
    boot = []
    for _ in range(n_boot):
        idx = rng_b.integers(0, n_t, n_t)
        tc = estimate_tau_c(taus, succ_mat[:, idx].mean(axis=1))
        if not np.isnan(tc):
            boot.append(tc)
    if len(boot) < 10:
        return np.nan, np.nan, np.nan
    tc_pt = estimate_tau_c(taus, succ_mat.mean(axis=1))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return float(tc_pt), float(ci_lo), float(ci_hi)


# ── 1-D simulation ────────────────────────────────────────────────────────────
def sim_1d(tau, eps, sigma, dphi, n_trials, rng):
    N     = n_trials
    delay = max(1, int(round(tau / DT)))
    blen  = delay + 1

    theta = rng.uniform(-math.pi, math.pi, N)
    K     = np.full(N, K0)
    e0    = wrap(OMEGA * 0.0 - theta)
    e_buf = np.tile(e0[:, None], (1, blen)).copy()
    bi    = 0

    nw = rng.standard_normal((N_WARM, N)) * math.sqrt(DT)
    no = rng.standard_normal((N_OBS,  N)) * math.sqrt(DT)

    for i in range(N_WARM):
        t  = (i + 1) * DT
        e  = wrap(OMEGA * t - theta)
        ed = e_buf[:, (bi - delay) % blen]
        theta += (OMEGA + K * np.sin(ed)) * DT + sigma * nw[i]
        K     += (eps * (A_COEF - np.cos(ed)) - MU * (K - K0)) * DT
        bi     = (bi + 1) % blen
        e_buf[:, bi] = e

    below = np.zeros(N, dtype=int)
    succ  = np.zeros(N, dtype=bool)
    Ts    = np.full(N, np.nan)

    for j in range(N_OBS):
        t  = T_WARM + (j + 1) * DT
        e  = wrap(OMEGA * t + dphi - theta)
        ed = e_buf[:, (bi - delay) % blen]
        theta += (OMEGA + K * np.sin(ed)) * DT + sigma * no[j]
        K     += (eps * (A_COEF - np.cos(ed)) - MU * (K - K0)) * DT
        bi     = (bi + 1) % blen
        e_buf[:, bi] = e

        in_zone = np.abs(e) < E_TH
        below   = np.where(in_zone, below + 1, 0)
        new_s   = (~succ) & (below >= N_HOLD)
        Ts      = np.where(new_s, (j + 1 - N_HOLD + 1) * DT, Ts)
        succ   |= new_s

    return succ.astype(float), Ts


# ── 2-D simulation (corrected) ────────────────────────────────────────────────
def sim_2d(tau, gamma, eps, sigma, dphi, n_trials, rng):
    """
    Correct 2D model:
      dθ = [ν + K·sin(e_del)] dt + σ_e dW_e
      dν = -γ(ν - ω) dt + σ_ν dW_ν        ← OU around ω, not 0
      dK = ε[A - cos(e_del) - μ(K-K₀)] dt
      e  = wrap(θ_ref - θ),  θ_ref = ω·t

    ν is an OU process with mean ω (natural frequency).
    In steady state ν ≈ ω → dθ ≈ ω·dt → θ tracks θ_ref. ✓
    γ controls how tightly ν is damped to ω.
    σ_ν = σ·√(2γ) keeps var_ss(ν) = σ² independent of γ.
    """
    N     = n_trials
    delay = max(1, int(round(tau / DT)))
    blen  = delay + 1

    sigma_v = sigma * math.sqrt(2.0 * gamma)

    theta = rng.uniform(-math.pi, math.pi, N)
    v     = np.full(N, OMEGA)           # initialise at natural frequency
    K     = np.full(N, K0)
    e0    = wrap(OMEGA * 0.0 - theta)
    e_buf = np.tile(e0[:, None], (1, blen)).copy()
    bi    = 0

    nw_e = rng.standard_normal((N_WARM, N)) * math.sqrt(DT)
    nw_v = rng.standard_normal((N_WARM, N)) * math.sqrt(DT)
    no_e = rng.standard_normal((N_OBS,  N)) * math.sqrt(DT)
    no_v = rng.standard_normal((N_OBS,  N)) * math.sqrt(DT)

    # warm-up
    for i in range(N_WARM):
        t  = (i + 1) * DT
        e  = wrap(OMEGA * t - theta)
        ed = e_buf[:, (bi - delay) % blen]

        theta += (v + K * np.sin(ed))      * DT + sigma   * nw_e[i]
        v     += (-gamma * (v - OMEGA))    * DT + sigma_v * nw_v[i]
        K     += (eps * (A_COEF - np.cos(ed)) - MU * (K - K0)) * DT

        bi = (bi + 1) % blen
        e_buf[:, bi] = e

    # observation with perturbation
    below = np.zeros(N, dtype=int)
    succ  = np.zeros(N, dtype=bool)
    Ts    = np.full(N, np.nan)

    for j in range(N_OBS):
        t  = T_WARM + (j + 1) * DT
        e  = wrap(OMEGA * t + dphi - theta)
        ed = e_buf[:, (bi - delay) % blen]

        theta += (v + K * np.sin(ed))      * DT + sigma   * no_e[j]
        v     += (-gamma * (v - OMEGA))    * DT + sigma_v * no_v[j]
        K     += (eps * (A_COEF - np.cos(ed)) - MU * (K - K0)) * DT

        bi = (bi + 1) % blen
        e_buf[:, bi] = e

        in_zone = np.abs(e) < E_TH
        below   = np.where(in_zone, below + 1, 0)
        new_s   = (~succ) & (below >= N_HOLD)
        Ts      = np.where(new_s, (j + 1 - N_HOLD + 1) * DT, Ts)
        succ   |= new_s

    return succ.astype(float), Ts


# ── sweeps ────────────────────────────────────────────────────────────────────
def run_1d_sweep():
    rng  = np.random.default_rng(SEED)
    rows = []
    total = len(EPS_LIST) * len(SIGMA_LIST) * len(TAUS_1D)
    done, t0 = 0, time.time()

    for eps in EPS_LIST:
        for sigma in SIGMA_LIST:
            succ_mat = np.zeros((len(TAUS_1D), N_TRIALS_1D))
            ts_mat   = np.full((len(TAUS_1D), N_TRIALS_1D), np.nan)

            for ti, tau in enumerate(TAUS_1D):
                s, ts = sim_1d(tau, eps, sigma, DPHI_1D, N_TRIALS_1D, rng)
                succ_mat[ti] = s
                ts_mat[ti]   = ts
                done += 1
                if done % 50 == 0:
                    print(f"  1D {done}/{total}  ({time.time()-t0:.0f}s)")

            S     = succ_mat.mean(axis=1)
            tau_c = estimate_tau_c(TAUS_1D, S)

            for ti, tau in enumerate(TAUS_1D):
                vt = ts_mat[ti][~np.isnan(ts_mat[ti])]
                rows.append(dict(
                    eps=eps, sigma=sigma, tau=float(tau),
                    S=float(S[ti]),
                    Ts_mean=float(vt.mean())       if vt.size     else np.nan,
                    Ts_std =float(vt.std(ddof=1))  if vt.size > 1 else np.nan,
                    tau_c        =float(tau_c),
                    tau_c_linear =analytical_tau_c(eps),
                ))
    return pd.DataFrame(rows)


def run_2d_sweep():
    rng  = np.random.default_rng(SEED + 1)
    rows, tc_rows = [], []
    total = len(GAMMA_LIST) * len(TAUS_2D)
    done, t0 = 0, time.time()

    for gamma in GAMMA_LIST:
        succ_mat = np.zeros((len(TAUS_2D), N_TRIALS_2D))
        ts_mat   = np.full((len(TAUS_2D), N_TRIALS_2D), np.nan)

        for ti, tau in enumerate(TAUS_2D):
            s, ts = sim_2d(tau, gamma, EPS_2D, SIGMA_2D, DPHI_2D, N_TRIALS_2D, rng)
            succ_mat[ti] = s
            ts_mat[ti]   = ts
            done += 1
            if done % 20 == 0:
                print(f"  2D {done}/{total}  ({time.time()-t0:.0f}s)")

        S                   = succ_mat.mean(axis=1)
        tc_pt, ci_lo, ci_hi = bootstrap_tau_c(TAUS_2D, succ_mat, N_BOOT)
        tc_rows.append(dict(gamma=float(gamma),
                            tau_c=tc_pt, ci_lo=ci_lo, ci_hi=ci_hi))

        for ti, tau in enumerate(TAUS_2D):
            vt = ts_mat[ti][~np.isnan(ts_mat[ti])]
            rows.append(dict(
                gamma=float(gamma), tau=float(tau),
                S=float(S[ti]),
                Ts_mean=float(vt.mean())       if vt.size     else np.nan,
                Ts_std =float(vt.std(ddof=1))  if vt.size > 1 else np.nan,
            ))

    return pd.DataFrame(rows), pd.DataFrame(tc_rows)


# ── figures ───────────────────────────────────────────────────────────────────
RC = {'figure.dpi': 150, 'font.size': 11,
      'axes.spines.top': False, 'axes.spines.right': False}


def fig1(df1d):
    """Sharp transition – reference condition (ε=0.05, σ=0.05)."""
    sub = df1d[(df1d.eps == 0.05) & (df1d.sigma == 0.05)].sort_values('tau')
    tc  = sub.tau_c.iloc[0]
    tcl = sub.tau_c_linear.iloc[0]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(sub.tau, sub.S, 'k-o', ms=4, lw=1.5)
        ax.axhline(0.5,  ls='--', color='gray',      lw=1,   label='S = 0.5')
        ax.axvline(tc,   ls=':',  color='steelblue',  lw=1.8, label=f'τ_c = {tc:.2f} s')
        ax.axvline(tcl,  ls='--', color='firebrick',  lw=1.2,
                   label=f'τ_c^linear = {tcl:.2f} s  (π/2K*)')
        ax.set_xlabel('Feedback delay τ (s)')
        ax.set_ylabel('Success rate S')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title('Sharp recoverability transition\n(ε=0.05, σ=0.05, Δφ=0.5π)')
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(OUTDIR / 'fig1_success_rate_ref.pdf')
        plt.close(fig)
    print('  fig1 saved')


def fig2(df1d):
    """τ_c vs σ for each ε, with analytical upper bound."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for eps, col in zip(EPS_LIST, colors):
            sub = (df1d[df1d.eps == eps]
                   .groupby('sigma')[['tau_c', 'tau_c_linear']]
                   .first().reset_index())
            ax.plot(sub.sigma, sub.tau_c, 'o-', color=col, ms=6, lw=1.5,
                    label=f'ε = {eps}')
            tcl = sub.tau_c_linear.iloc[0]
            ax.axhline(tcl, ls='--', color=col, lw=0.9, alpha=0.7)
        ax.set_xlabel('Noise intensity σ')
        ax.set_ylabel('Critical delay τ_c (s)')
        ax.set_title('τ_c vs noise\n(dashed: analytical upper bound π/2K*)')
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUTDIR / 'fig2_tau_c_vs_sigma.pdf')
        plt.close(fig)
    print('  fig2 saved')


def fig3(df2d):
    """S(τ) slices for 3 representative γ values."""
    all_g  = sorted(df2d.gamma.unique())
    targets= [0.3, 1.65, 3.0]
    sel    = [min(all_g, key=lambda g: abs(g - t)) for t in targets]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for g, col in zip(sel, colors):
            sub = df2d[df2d.gamma == g].sort_values('tau')
            tc  = estimate_tau_c(sub.tau.values, sub.S.values)
            ax.plot(sub.tau, sub.S, 'o-', color=col, ms=4, lw=1.5,
                    label=f'γ = {g:.2f}')
            if not np.isnan(tc):
                ax.axvline(tc, ls=':', color=col, lw=0.9)
        ax.axhline(0.5, ls='--', color='gray', lw=1)
        ax.set_xlabel('Feedback delay τ (s)')
        ax.set_ylabel('Success rate S')
        ax.set_title('S(τ) for representative γ  (2D model)\nΔφ=0.5π')
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUTDIR / 'fig3_1d_slices.pdf')
        plt.close(fig)
    print('  fig3 saved')


def fig4(df2d):
    """Heatmaps: S(τ,γ) and <T_rec>(τ,γ)."""
    taus   = sorted(df2d.tau.unique())
    gammas = sorted(df2d.gamma.unique())
    ng, nt = len(gammas), len(taus)
    S_mat  = np.full((ng, nt), np.nan)
    T_mat  = np.full((ng, nt), np.nan)

    for gi, g in enumerate(gammas):
        for ti, t in enumerate(taus):
            row = df2d[(df2d.gamma == g) & (df2d.tau == t)]
            if len(row):
                S_mat[gi, ti] = row.S.values[0]
                T_mat[gi, ti] = row.Ts_mean.values[0]

    ext = [taus[0] - 0.05, taus[-1] + 0.05,
           gammas[0] - 0.1, gammas[-1] + 0.1]

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        im0 = axes[0].imshow(S_mat, aspect='auto', origin='lower',
                              extent=ext, vmin=0, vmax=1, cmap='viridis')
        axes[0].set_xlabel('Feedback delay τ (s)')
        axes[0].set_ylabel('Damping rate γ')
        axes[0].set_title('(a)  Success rate  S(τ, γ)')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(np.ma.masked_invalid(T_mat), aspect='auto',
                              origin='lower', extent=ext, cmap='plasma')
        axes[1].set_xlabel('Feedback delay τ (s)')
        axes[1].set_ylabel('Damping rate γ')
        axes[1].set_title('(b)  Mean recovery time  ⟨T_rec⟩ (s)')
        plt.colorbar(im1, ax=axes[1])

        fig.tight_layout()
        fig.savefig(OUTDIR / 'fig4_heatmaps.pdf')
        plt.close(fig)
    print('  fig4 saved')


def fig5(df_tc):
    """τ_c(γ) with bootstrap 95% CI."""
    valid   = df_tc.dropna()
    mean_tc = valid.tau_c.mean() if len(valid) else np.nan
    span    = valid.tau_c.max() - valid.tau_c.min() if len(valid) else np.nan

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        if len(valid):
            ax.errorbar(
                valid.gamma, valid.tau_c,
                yerr=[valid.tau_c - valid.ci_lo, valid.ci_hi - valid.tau_c],
                fmt='ko', ms=6, lw=1.5, capsize=4, elinewidth=1)
            ax.axhline(mean_tc, ls='--', color='red', lw=1.5,
                       label=f'Mean τ_c = {mean_tc:.3f} s')
            ax.text(0.97, 0.05, f'Δτ_c = {span:.3f} s',
                    ha='right', va='bottom', transform=ax.transAxes,
                    fontsize=10, color='gray')
        ax.set_xlabel('Damping rate γ')
        ax.set_ylabel('Critical delay τ_c (s)')
        ax.set_title('τ_c(γ) with 95% bootstrap CI\n(σ_ν = σ√2γ, equal steady-state noise)')
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUTDIR / 'fig5_tau_c_gamma.pdf')
        plt.close(fig)
    print('  fig5 saved')


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    print('=== 1D sweep ===')
    df1d = run_1d_sweep()
    df1d.to_csv(OUTDIR / 'results_1d.csv', index=False)
    print(f'  {len(df1d)} rows → results_1d.csv')

    print('=== 2D sweep ===')
    df2d, df_tc = run_2d_sweep()
    df2d.to_csv(OUTDIR / 'results_2d.csv', index=False)
    df_tc.to_csv(OUTDIR / 'tau_c_by_gamma.csv', index=False)
    print(f'  {len(df2d)} rows → results_2d.csv')
    print(f'  {len(df_tc)} rows → tau_c_by_gamma.csv')

    print('=== Figures ===')
    fig1(df1d)
    fig2(df1d)
    fig3(df2d)
    fig4(df2d)
    fig5(df_tc)

    bounds = pd.DataFrame([dict(
        eps=e,
        K_star=round(K0 + e * (A_COEF - 1.0) / MU, 4),
        tau_c_linear=round(analytical_tau_c(e), 4),
    ) for e in EPS_LIST])
    bounds.to_csv(OUTDIR / 'analytical_bounds.csv', index=False)

    meta = dict(
        DT=DT, K0=K0, A_COEF=A_COEF, MU=MU, OMEGA=OMEGA,
        T_WARM=T_WARM, T_OBS=T_OBS, T_total=T_WARM+T_OBS,
        E_TH=E_TH, T_HOLD=T_HOLD,
        N_TRIALS_1D=N_TRIALS_1D, N_TRIALS_2D=N_TRIALS_2D,
        N_BOOT=N_BOOT, SEED=SEED,
        EPS_LIST=EPS_LIST, SIGMA_LIST=SIGMA_LIST,
        GAMMA_LIST=GAMMA_LIST.tolist(),
        TAUS_1D=TAUS_1D.tolist(), TAUS_2D=TAUS_2D.tolist(),
        EPS_2D=EPS_2D, SIGMA_2D=SIGMA_2D,
        sigma_v_scaling='sigma * sqrt(2*gamma)',
        note_2d='Feedback K*sin(e_del) applied to dtheta; sigma_v scaled for constant var_ss(v)',
    )
    with open(OUTDIR / 'run_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print('\n=== Analytical bounds (τ_c^linear = π/2K*) ===')
    print(bounds.to_string(index=False))
    print('\n=== 2D bootstrap τ_c(γ) ===')
    print(df_tc.to_string(index=False))
    print(f'\nTotal: {time.time()-t0:.0f}s  |  All outputs → {OUTDIR}')


if __name__ == '__main__':
    main()
