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
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Stellar mass -- halo mass
# ---------------------------------------------------------------------------

#: Moster, Naab & White (2013), Table 1, free-parameter best fit.
MOSTER13 = dict(M10=11.590, M11=1.195, N10=0.0351, N11=-0.0247, beta10=1.376, beta11=-0.826, gamma10=0.608, gamma11=0.329)


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
