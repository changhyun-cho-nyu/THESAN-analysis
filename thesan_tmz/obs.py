"""
Observational and semi-empirical relations to compare the run against.

Each function returns the relation evaluated on an array you supply, so it can
be over-plotted directly.  Every fit is quoted with its source equation number
so it can be checked against the paper; where a relation has an intrinsic
scatter the function also returns it (in dex) via the ``scatter`` attribute
attached to the module-level constant, or as a second return value.

IMPORTANT about h and IMF: all of these are stated in physical Msun with the
paper's own h and IMF (Chabrier or Kroupa for the stellar masses, which is what
AREPO's GFM star particles approximate).  We do *not* apply h corrections here
-- the analysis code converts the simulation to physical Msun first.

References
----------
Moster, Naab & White (2013), MNRAS 428, 3121        -- abundance-matched SMHM
Behroozi, Wechsler & Conroy (2013), ApJ 770, 57     -- abundance-matched SMHM
Kormendy & Ho (2013), ARA&A 51, 511                 -- M_BH-M_bulge, M_BH-sigma
Haering & Rix (2004), ApJ 604, L89                  -- M_BH-M_bulge
Reines & Volonteri (2015), ApJ 813, 82              -- M_BH-M_star (local + AGN)
Speagle et al. (2014), ApJS 214, 15                 -- star-forming main sequence
Madau & Dickinson (2014), ARA&A 52, 415             -- cosmic SFH
Tremonti et al. (2004), ApJ 613, 898                -- mass-metallicity (strong line)
Andrews & Martini (2013), ApJ 765, 140              -- mass-metallicity (direct T_e)
McGaugh (2012), AJ 143, 40                          -- baryonic Tully-Fisher
Bryan & Norman (1998), ApJ 495, 80                  -- virial overdensity
Maiolino et al. (2024), Nature 627, 59              -- JWST over-massive BHs (indicative)
Baggen et al. (2023), ApJL 955, L12                 -- compact massive candidates z ~ 7-9 (sizes)
Baggen et al. (2024), ApJL 977, L13                 -- Balmer-break compact galaxies z = 6.7-8.4
Tacchella et al. (2023), arXiv:2302.07234           -- GN-z11 size and mass
Messa et al. (2025), arXiv:2507.18705               -- Cosmic Gems star clusters, z = 9.625
Hopkins et al. (2010), MNRAS 401, L19               -- maximum stellar surface density
Curti et al. (2024), A&A 684, A75                   -- JADES gas MZR, z = 3-10
Nakajima et al. (2023), ApJS 269, 33                -- JWST gas MZR, z = 4-10
Chemerynska et al. (2024), arXiv:2407.17110         -- low-mass gas MZR, z = 6-8
Kirby et al. (2013), ApJ 779, 102                   -- Local Group dwarfs, stellar <[Fe/H]>-M*
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Stellar mass -- halo mass
# ---------------------------------------------------------------------------

#: Moster, Naab & White (2013), Table 1, free-parameter best fit.
MOSTER13 = dict(M10=11.590, M11=1.195, N10=0.0351, N11=-0.0247, beta10=1.376, beta11=-0.826, gamma10=0.608, gamma11=0.329)


#: redshifts over which the SMHM fits are constrained by data (checked 2026-11-12):
#: Moster+13 z = 0-4 (abstract, Sec. 2.2, 3.2); Behroozi+13 z = 0-8 (abstract, Sec. V; beyond is
#: extrapolation, App. B).  Neither paper states a halo-mass limit.
MOSTER13_Z_RANGE = (0.0, 4.0)
BEHROOZI13_Z_RANGE = (0.0, 8.0)


def moster13_smhm(Mhalo, z=0.0, params=None):
    """Stellar mass from halo mass, Moster+2013 Eq. 2 with Eqs. 11-14.

    ``Mhalo`` in Msun (M_200c), returns M_star in Msun.  Calibrated over
    0 <= z <= 4; extrapolating past that is common but should be labelled.
    """
    p = dict(MOSTER13, **(params or {}))
    M = np.asarray(Mhalo, dtype=np.float64)
    zz = np.asarray(z, dtype=np.float64)
    f = zz / (zz + 1.0)

    logM1 = p["M10"] + p["M11"] * f
    N = p["N10"] + p["N11"] * f
    beta = p["beta10"] + p["beta11"] * f
    gamma = p["gamma10"] + p["gamma11"] * f

    M1 = 10.0 ** logM1
    x = M / M1
    return M * 2.0 * N / (x ** (-beta) + x ** gamma)


#: Behroozi, Wechsler & Conroy (2013), Section 5 best fit.
BEHROOZI13 = dict(
    M1_0=11.514,
    M1_a=-1.793,
    M1_z=-0.251,
    eps_0=-1.777,
    eps_a=-0.006,
    eps_z=0.000,
    eps_a2=-0.119,
    alpha_0=-1.412,
    alpha_a=0.731,
    delta_0=3.508,
    delta_a=2.608,
    delta_z=-0.043,
    gamma_0=0.316,
    gamma_a=1.319,
    gamma_z=0.279,
)


def behroozi13_smhm(Mhalo, z=0.0, params=None):
    """Stellar mass from halo mass, Behroozi+2013 Eqs. 3-4.

    ``Mhalo`` in Msun (M_200c), returns M_star in Msun.  Calibrated to z ~ 8,
    which is what makes it the useful comparison for a THESAN-Zoom run.
    """
    p = dict(BEHROOZI13, **(params or {}))
    zz = np.asarray(z, dtype=np.float64)
    a = 1.0 / (1.0 + zz)
    nu = np.exp(-4.0 * a * a)

    logM1 = p["M1_0"] + (p["M1_a"] * (a - 1.0) + p["M1_z"] * zz) * nu
    logeps = p["eps_0"] + (p["eps_a"] * (a - 1.0) + p["eps_z"] * zz) * nu + p["eps_a2"] * (a - 1.0)
    alpha = p["alpha_0"] + (p["alpha_a"] * (a - 1.0)) * nu
    delta = p["delta_0"] + (p["delta_a"] * (a - 1.0) + p["delta_z"] * zz) * nu
    gamma = p["gamma_0"] + (p["gamma_a"] * (a - 1.0) + p["gamma_z"] * zz) * nu

    def f(x):
        # Behroozi+2013 Eq. 4.  The exp(10^-x) term saturates for x << 0, so we
        # clip the exponent to keep the low-mass end finite instead of inf.
        p1 = -np.log10(10.0 ** (alpha * x) + 1.0)
        p2 = delta * np.power(np.log10(1.0 + np.exp(np.clip(x, -700, 700))), gamma)
        p3 = 1.0 + np.exp(np.clip(10.0 ** (-x), -700, 700))
        return p1 + p2 / p3

    x = np.log10(np.asarray(Mhalo, dtype=np.float64)) - logM1
    logMstar = logeps + logM1 + f(x) - f(0.0)
    return 10.0 ** logMstar


# ---------------------------------------------------------------------------
# Black hole scaling relations
# ---------------------------------------------------------------------------

KORMENDY_HO_13_SCATTER = 0.29  # dex, M_BH-M_bulge
REINES_VOLONTERI_15_SCATTER = 0.55  # dex, AGN relation


def kormendy_ho13_mbh_mbulge(Mbulge):
    """Kormendy & Ho (2013) Eq. 10:

        M_BH / 1e9 Msun = 0.49 (M_bulge / 1e11 Msun)^1.17

    Intrinsic scatter 0.29 dex.  Calibrated on *classical bulges and
    ellipticals* only -- pseudobulges and late types lie well below it, so
    comparing a gas-rich high-z disc against this line is an upper envelope,
    not an expectation.
    """
    M = np.asarray(Mbulge, dtype=np.float64)
    return 0.49e9 * (M / 1.0e11) ** 1.17


def haering_rix04_mbh_mbulge(Mbulge):
    """Haering & Rix (2004): log M_BH = 8.20 + 1.12 log(M_bulge / 1e11)."""
    M = np.asarray(Mbulge, dtype=np.float64)
    return 10.0 ** (8.20 + 1.12 * np.log10(M / 1.0e11))


def kormendy_ho13_mbh_sigma(sigma_kms):
    """Kormendy & Ho (2013) Eq. 7:

        M_BH / 1e9 Msun = 0.309 (sigma / 200 km/s)^4.38

    Intrinsic scatter 0.29 dex.
    """
    s = np.asarray(sigma_kms, dtype=np.float64)
    return 0.309e9 * (s / 200.0) ** 4.38


def reines_volonteri15_local(Mstar):
    """Reines & Volonteri (2015) Eq. 4, local inactive E/classical-bulge sample:

        log M_BH = 8.95 + 1.40 log(M_star / 1e11)

    Scatter 0.47 dex.
    """
    M = np.asarray(Mstar, dtype=np.float64)
    return 10.0 ** (8.95 + 1.40 * np.log10(M / 1.0e11))


def reines_volonteri15_agn(Mstar):
    """Reines & Volonteri (2015) Eq. 5, broad-line AGN sample:

        log M_BH = 7.45 + 1.05 log(M_star / 1e11)

    Scatter 0.55 dex.  This is the more appropriate comparison for a
    growing, actively accreting BH in a star-forming galaxy -- it sits about
    1.5 dex below the inactive-elliptical relation at fixed M_star.
    """
    M = np.asarray(Mstar, dtype=np.float64)
    return 10.0 ** (7.45 + 1.05 * np.log10(M / 1.0e11))


def jwst_overmassive_band(Mstar, lo_dex=1.0, hi_dex=2.5):
    """Indicative band for the JWST z > 4 'over-massive' BH population.

    Maiolino et al. (2024) and companion papers find little red dots and
    broad-line AGN sitting 1-2.5 dex above the local M_BH-M_star relation at
    fixed stellar mass.  Individual masses are uncertain (single-epoch virial,
    uncertain host decomposition), so this is deliberately a *band* anchored on
    the local AGN relation rather than a fit.  Returns (lo, hi) in Msun.
    """
    base = reines_volonteri15_agn(Mstar)
    return base * 10.0 ** lo_dex, base * 10.0 ** hi_dex


def bh_mass_fraction_local():
    """Typical local M_BH / M_bulge, for a horizontal reference line.

    Kormendy & Ho (2013) quote ~0.5% for classical bulges/ellipticals.
    """
    return 5.0e-3


# ---------------------------------------------------------------------------
# Star formation
# ---------------------------------------------------------------------------


def speagle14_main_sequence(Mstar, age_gyr):
    """Speagle et al. (2014) Eq. 28 -- the star-forming main sequence:

        log SFR = (0.84 - 0.026 t) log M_star - (6.51 - 0.11 t)

    with ``t`` the age of the universe in Gyr.  Returns SFR in Msun/yr.
    Calibrated over 0 < z < 6 (t >~ 0.9 Gyr); below that it is extrapolation.
    """
    logM = np.log10(np.asarray(Mstar, dtype=np.float64))
    t = np.asarray(age_gyr, dtype=np.float64)
    return 10.0 ** ((0.84 - 0.026 * t) * logM - (6.51 - 0.11 * t))


def madau_dickinson14_csfh(z):
    """Madau & Dickinson (2014) Eq. 15 -- cosmic SFR density:

        psi(z) = 0.015 (1+z)^2.7 / [1 + ((1+z)/2.9)^5.6]   Msun/yr/Mpc^3

    Salpeter IMF as published; multiply by ~0.63 for Chabrier.
    """
    zz = np.asarray(z, dtype=np.float64)
    return 0.015 * (1.0 + zz) ** 2.7 / (1.0 + ((1.0 + zz) / 2.9) ** 5.6)


def specific_sfr_from_cosmology(z, Omega0=0.3089, OmegaLambda=0.6911, h=0.6774):
    """Baryonic-accretion-limited sSFR, ~ the halo specific growth rate.

    A crude but useful upper envelope: a galaxy cannot sustain a specific SFR
    much above the specific mass growth rate of its halo.  Uses the
    Dekel et al. (2009) fit sSFR ~ 0.03 (1+z)^2.5 / Gyr.
    """
    zz = np.asarray(z, dtype=np.float64)
    return 0.03 * (1.0 + zz) ** 2.5  # per Gyr


# ---------------------------------------------------------------------------
# Metallicity
# ---------------------------------------------------------------------------


def tremonti04_mzr(Mstar):
    """Tremonti et al. (2004) Eq. 3 -- SDSS strong-line MZR:

        12 + log(O/H) = -1.492 + 1.847 log M* - 0.08026 (log M*)^2

    Valid 8.5 < log M* < 11.5; the quadratic turns over unphysically outside.
    """
    x = np.log10(np.asarray(Mstar, dtype=np.float64))
    return -1.492 + 1.847 * x - 0.08026 * x * x


def andrews_martini13_mzr(Mstar):
    """Andrews & Martini (2013) direct-T_e MZR:

        12 + log(O/H) = 8.798 - log10(1 + (10^8.901 / M*)^0.640)

    Sits ~0.3 dex below Tremonti+2004 at high mass; the offset between the two
    is a fair proxy for the systematic uncertainty in any MZR comparison.
    """
    M = np.asarray(Mstar, dtype=np.float64)
    return 8.798 - np.log10(1.0 + (10.0 ** 8.901 / M) ** 0.640)


#: Solar oxygen abundance, Asplund et al. (2009).
OH_SOLAR = 8.69
#: Solar metallicity mass fraction used by GFM (src/GFM/, All.SolarMetallicity).
Z_SOLAR = 0.0127


def logOH_from_Zmass(Z_mass_fraction, Z_solar=Z_SOLAR, OH_solar=OH_SOLAR):
    """Convert a GFM mass-fraction metallicity to 12+log(O/H), scaled-solar.

    This assumes the O/Z ratio is solar, which is wrong at the ~0.1-0.2 dex
    level for alpha-enhanced young systems.  If the snapshot carries
    GFM_Metals, use the actual oxygen mass fraction instead.
    """
    Z = np.asarray(Z_mass_fraction, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return OH_solar + np.log10(np.where(Z > 0, Z / Z_solar, np.nan))


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


def mcgaugh12_btfr(V_kms):
    """Baryonic Tully-Fisher, McGaugh (2012): M_bar = 47 V_flat^4 Msun."""
    V = np.asarray(V_kms, dtype=np.float64)
    return 47.0 * V ** 4


def bryan_norman98_delta_c(z, Omega0=0.3089, OmegaLambda=0.6911):
    """Virial overdensity relative to the critical density."""
    zz = np.asarray(z, dtype=np.float64)
    a = 1.0 / (1.0 + zz)
    Om_z = Omega0 / a ** 3 / (Omega0 / a ** 3 + OmegaLambda)
    x = Om_z - 1.0
    return 18.0 * np.pi ** 2 + 82.0 * x - 39.0 * x ** 2


# ---------------------------------------------------------------------------
# Convenience: shaded scatter bands
# ---------------------------------------------------------------------------


def scatter_band(y, dex):
    """(lower, upper) envelopes for a relation with ``dex`` of log scatter."""
    y = np.asarray(y, dtype=np.float64)
    return y * 10.0 ** (-dex), y * 10.0 ** (+dex)


# ---------------------------------------------------------------------------
# Compact high-redshift systems (JWST), for the stellar density profile (fig 23)
# ---------------------------------------------------------------------------
# Values checked against each paper's table on 2026-11-11 (arXiv HTML versions;
# see the work log).  Sizes are as published plus b/a, so the caller can
# circularise: r_e,c = r_e sqrt(b/a).

#: Salpeter -> Chabrier stellar mass (dex), as used for Baggen+23 (masses from Labbe+23, Salpeter)
SALPETER_TO_CHABRIER_DEX = -0.24

#: Baggen+23 Table 1 (arXiv:2305.17162): ID, z_phot, log M* (Salpeter, from Labbe+23),
#: adopted major-axis r_e [pc] (+err, -err), b/a (WebbPSF fit).  F200W.
BAGGEN23 = [
    (7274, 7.77, 9.87, 273, 18, 22, 0.5), (11184, 7.32, 10.18, 216, 18, 32, 0.7),
    (14924, 8.83, 10.02, 108, 26, 10, 0.7), (16624, 8.52, 9.30, 311, 85, 54, 0.7),
    (21834, 8.54, 9.61, 137, 51, 60, 0.1), (25666, 7.93, 9.52, 86, 20, 12, 0.8),
    (28984, 7.54, 9.57, 111, 12, 17, 0.5), (37888, 6.51, 9.23, 88, 40, 27, 0.3),
    (38094, 7.48, 10.89, 82, 8, 15, 0.6),
]

#: Baggen+24 Table 1 (arXiv:2408.07745): RUBIES ID, z_spec, log M* medium / min (max AGN) /
#: max (no AGN) (Chabrier, from Wang et al.), major-axis r_e [pc], b/a.  F200W.
BAGGEN24 = [
    (49140, 6.68, 9.93, 9.50, 11.2, 123, 0.58),
    (55604, 6.98, 9.79, 8.99, 11.1, 54, 0.78),
    (966323, 8.35, 9.84, 8.72, 10.6, 86, 0.73),
]

#: GN-z11, Tacchella+23 (arXiv:2302.07234): z = 10.60; point source + extended size
#: 64 +- 20 pc (the extended part alone is ~200 pc); Prospector mass 10^9.1 (+0.3, -0.4),
#: Chabrier, no AGN light.  (Size and mass read from the paper's later sections via a summary.)
GNZ11 = dict(z=10.60, r_e=64.0, r_e_err=20.0, logm=9.1, logm_hi=0.3, logm_lo=0.4)

#: Messa+25 Table 1 (arXiv:2507.18705), Cosmic Gems clusters, z = 9.625, glafic lens model,
#: Kroupa IMF: ID, log M*, R_eff [pc], R_eff is an upper limit
MESSA25 = [
    ("A.1", 6.79, 1.1, False), ("B.1", 6.35, 1.1, False), ("C.1", 6.02, 0.9, True), ("D.1", 5.89, 0.7, False),
    ("E.1", 5.97, 0.4, False), ("A.2", 6.49, 1.7, False), ("B.2", 6.47, 1.3, True), ("C.2+D.2", 6.60, 1.9, False),
]

#: Hopkins+10 (MNRAS 401, L19) abstract: Sigma_max ~ 1e11 Msun/kpc^2 = 1e5 Msun/pc^2
HOPKINS10_SIGMA_MAX_PC2 = 1.0e5


# ---------------------------------------------------------------------------
# High-redshift gas-phase MZR (12 + log O/H) and the Local Group iron-mass
# relation, for figs 05 (MZR), 24, 25.  Checked against the papers on
# 2026-11-11 (see the work log); entries marked "summary" were read from a
# summary of the paper's later sections, not verbatim text.
# ---------------------------------------------------------------------------

#: Curti+24 (arXiv:2304.08516) Table 3 (summary): 12+log(O/H) = Zm8 + slope log(M*/1e8),
#: Chabrier; the 3-10 row is also in the abstract (verbatim).  Key: redshift bin.
CURTI24_FITS = {(3, 10): (7.72, 0.17), (3, 6): (7.75, 0.18), (6, 10): (7.65, 0.11)}
#: Curti+24 Table 2 (summary), z = 6-10 medians: (log M*, 12+log O/H)
CURTI24_MEDIANS_Z6_10 = [(7.35, 7.64), (8.10, 7.67), (8.76, 7.73)]
#: mass range of the Curti+24 sample (log M*)
CURTI24_LOGM_RANGE = (6.0, 10.0)

#: Nakajima+23 (arXiv:2301.12825) eq. 1 (summary): 12+log(O/H) = 8.24 + 0.25 log(M*/1e10), z = 4-10,
#: Chabrier; mass range log M* 7.5-9 (verbatim intro; 9.5 only in the summary)
NAKAJIMA23 = (8.24, 0.25, 10.0)
NAKAJIMA23_LOGM_RANGE = (7.5, 9.0)

#: Chemerynska+24 (arXiv:2407.17110) abstract (verbatim), z = 6-8, M* = 1e6-1e8:
#: 12+log(O/H) = a x^2 + b x + c, x = log M*
CHEMERYNSKA24 = (-0.076, 1.61, -0.26)
CHEMERYNSKA24_LOGM_RANGE = (6.0, 8.0)

#: Kirby+13 (arXiv:1310.0814) eq. 4 (verbatim): <[Fe/H]> = -1.69 + 0.30 log(M*/1e6), rms 0.17;
#: <[Fe/H]> is the error-weighted mean of individual stars; fitted from log M* = 3.68 (Coma Ber)
#: to 5e8 Msun (NGC 205), z = 0 Local Group dwarfs; solar log eps(Fe) = 7.52 in the paper
KIRBY13 = (-1.69, 0.30, 6.0, 0.17)
KIRBY13_LOGM_RANGE = (3.68, np.log10(5e8))


def curti24_mzr(logm, zbin=(6, 10)):
    zm8, slope = CURTI24_FITS[zbin]
    return zm8 + slope * (np.asarray(logm) - 8.0)


def nakajima23_mzr(logm):
    z10, slope, pivot = NAKAJIMA23
    return z10 + slope * (np.asarray(logm) - pivot)


def chemerynska24_mzr(logm):
    a, b, c = CHEMERYNSKA24
    x = np.asarray(logm)
    return a * x ** 2 + b * x + c


def kirby13_feh(logm):
    a, b, pivot, _ = KIRBY13
    return a + b * (np.asarray(logm) - pivot)
