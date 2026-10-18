"""
Unit system and cosmology helpers for AREPO2 output.

AREPO stores everything in *code units*, defined by three numbers in the
parameter file:

    UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s

and, for a comoving run, additionally divided by the Hubble parameter h and
scaled by the expansion factor a.  The exact powers of a and h are the single
most common source of silent factor-of-a-few errors in analysis scripts, so
they are all collected here once and never repeated elsewhere in the package.

Convention used by AREPO for a cosmological run (ComovingIntegrationOn 1):

    quantity          stored as                       physical value
    ----------------  -----------------------------   -------------------------
    Coordinates       comoving, in Unit / h           x * a / h
    Masses            in Unit / h                     m / h
    Density           comoving, Unit h^2 / a^3        rho * h^2 / a^3
    Velocities        sqrt(a) * peculiar velocity     v * sqrt(a)  (peculiar)
    InternalEnergy    physical (no a or h)            u
    SmoothingLength   comoving, Unit / h              h_sml * a / h
    BH_Mdot           code mass / code time / h?      see note below
    Potential         (a) km^2/s^2                    phi / a

Note on BH_Mdot: AREPO's own blackholes.txt conversion (blackhole.cc:215) is

    mdot_msun_per_yr = mdot * (UnitMass_in_g / SOLAR_MASS)
                            / (UnitTime_in_s / SEC_PER_YEAR)

i.e. *no* factor of h and *no* factor of a.  The h cancels because both the
mass and the time carry one power of 1/h in the standard THESAN/Illustris unit
choice (kpc/h, 1e10 Msun/h, km/s -> UnitTime = 0.978 Gyr/h).  We reproduce that
exact expression in `mdot_to_msun_per_yr` so the Python numbers agree with the
code's own log to machine precision.

References
----------
Springel (2010), MNRAS 401, 791 -- AREPO method paper.
Planck Collaboration VI (2020), A&A 641, A6 -- default cosmology.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants, in cgs, matching src/main/constants.h so that anything we
# compute here can be compared against the code's own numbers exactly.
# ---------------------------------------------------------------------------
GRAVITY = 6.6738e-8  # cm^3 g^-1 s^-2
SOLAR_MASS = 1.989e33  # g
SOLAR_LUM = 3.826e33  # erg s^-1
PROTONMASS = 1.67262178e-24  # g
ELECTRONMASS = 9.1093829e-28  # g
BOLTZMANN = 1.38065e-16  # erg K^-1
CLIGHT = 2.99792458e10  # cm s^-1
THOMPSON = 6.65245873e-25  # cm^2
SEC_PER_MEGAYEAR = 3.15576e13
SEC_PER_YEAR = 3.15576e7
HUBBLE = 3.2407789e-18  # h / s
KPC_IN_CM = 3.085678e21
MPC_IN_CM = 3.085678e24
GAMMA = 5.0 / 3.0
GAMMA_MINUS1 = GAMMA - 1.0
HYDROGEN_MASSFRAC = 0.76

# Eddington accretion rate coefficient, mirroring blackhole.cc:216.
#   Mdot_Edd = 4 pi G M m_p / (eps_r c sigma_T)
# The helper below takes the radiative efficiency explicitly because this
# branch supports BH_VARIABLE_RADIATIVE_EFFICIENCY.


class Units:
    """Code-unit <-> physical-unit conversions for one AREPO run.

    Build one with :meth:`from_snapshot_file` (preferred -- it reads the
    ``/Header`` and ``/Parameters`` groups that AREPO writes into every HDF5
    snapshot) or by passing the three unit numbers directly.

    All ``*_to_*`` attributes are plain floats you multiply by.  Anything that
    depends on the expansion factor is a *method* taking ``a``, so that it is
    impossible to accidentally use a snapshot's factor on another snapshot.
    """

    def __init__(
        self,
        UnitLength_in_cm=KPC_IN_CM,
        UnitMass_in_g=1.989e43,
        UnitVelocity_in_cm_per_s=1.0e5,
        HubbleParam=0.6774,
        Omega0=0.3089,
        OmegaLambda=0.6911,
        OmegaBaryon=0.0486,
        ComovingIntegrationOn=True,
        BoxSize=None,
    ):
        self.UnitLength_in_cm = float(UnitLength_in_cm)
        self.UnitMass_in_g = float(UnitMass_in_g)
        self.UnitVelocity_in_cm_per_s = float(UnitVelocity_in_cm_per_s)
        self.h = float(HubbleParam)
        self.Omega0 = float(Omega0)
        self.OmegaLambda = float(OmegaLambda)
        self.OmegaBaryon = float(OmegaBaryon)
        self.comoving = bool(ComovingIntegrationOn)
        self.BoxSize = None if BoxSize is None else float(BoxSize)

        # Derived cgs scales (identical to All.Unit* in the code).
        self.UnitTime_in_s = self.UnitLength_in_cm / self.UnitVelocity_in_cm_per_s
        self.UnitDensity_in_cgs = self.UnitMass_in_g / self.UnitLength_in_cm ** 3
        self.UnitEnergy_in_cgs = self.UnitMass_in_g * self.UnitLength_in_cm ** 2 / self.UnitTime_in_s ** 2
        self.UnitPressure_in_cgs = self.UnitMass_in_g / self.UnitLength_in_cm / self.UnitTime_in_s ** 2
        self.UnitTime_in_Myr = self.UnitTime_in_s / SEC_PER_MEGAYEAR
        self.UnitTime_in_Gyr = self.UnitTime_in_Myr / 1.0e3

        # Gravitational constant in code units (All.G).
        self.G = GRAVITY / self.UnitLength_in_cm ** 3 * self.UnitMass_in_g * self.UnitTime_in_s ** 2
        # Hubble constant in code units (All.Hubble); h is carried separately.
        self.Hubble = HUBBLE * self.UnitTime_in_s

    # -- construction -------------------------------------------------------

    @classmethod
    def from_snapshot_file(cls, path):
        """Read units + cosmology straight out of an HDF5 snapshot.

        Uses ``/Parameters`` when present (AREPO2 writes it, io.cc:969) and
        falls back to ``/Header`` for older files, which only carry the
        cosmology and not the unit scales.
        """
        import h5py

        with h5py.File(path, "r") as f:
            hdr = dict(f["Header"].attrs)
            par = dict(f["Parameters"].attrs) if "Parameters" in f else {}

        def pick(key, default):
            if key in par:
                return par[key]
            if key in hdr:
                return hdr[key]
            return default

        return cls(
            UnitLength_in_cm=pick("UnitLength_in_cm", KPC_IN_CM),
            UnitMass_in_g=pick("UnitMass_in_g", 1.989e43),
            UnitVelocity_in_cm_per_s=pick("UnitVelocity_in_cm_per_s", 1.0e5),
            HubbleParam=pick("HubbleParam", 0.6774),
            Omega0=pick("Omega0", 0.3089),
            OmegaLambda=pick("OmegaLambda", 0.6911),
            OmegaBaryon=pick("OmegaBaryon", 0.0486),
            ComovingIntegrationOn=bool(pick("ComovingIntegrationOn", 1)),
            BoxSize=pick("BoxSize", None),
        )

    @classmethod
    def from_parameterfile(cls, path):
        """Read units + cosmology from a plain-text parameter file.

        Works both on the file you handed to AREPO and on the
        ``<param>-usedvalues`` file it writes back out (parameters.cc:68).
        """
        vals = {}
        with open(path) as fh:
            for line in fh:
                line = line.split("%")[0].split("#")[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    vals[parts[0]] = parts[1]

        def num(key, default):
            try:
                return float(vals[key])
            except (KeyError, ValueError):
                return default

        return cls(
            UnitLength_in_cm=num("UnitLength_in_cm", KPC_IN_CM),
            UnitMass_in_g=num("UnitMass_in_g", 1.989e43),
            UnitVelocity_in_cm_per_s=num("UnitVelocity_in_cm_per_s", 1.0e5),
            HubbleParam=num("HubbleParam", 0.6774),
            Omega0=num("Omega0", 0.3089),
            OmegaLambda=num("OmegaLambda", 0.6911),
            OmegaBaryon=num("OmegaBaryon", 0.0486),
            ComovingIntegrationOn=bool(num("ComovingIntegrationOn", 1)),
            BoxSize=num("BoxSize", None) or None,
        )

    # -- a/h-independent conversions ---------------------------------------

    @property
    def mass_to_msun(self):
        """Code mass -> Msun.  Includes the 1/h of a comoving run."""
        f = self.UnitMass_in_g / SOLAR_MASS
        return f / self.h if self.comoving else f

    @property
    def energy_to_erg(self):
        f = self.UnitEnergy_in_cgs
        return f / self.h if self.comoving else f

    @property
    def specific_energy_to_cgs(self):
        """Code specific internal energy -> erg/g.  No a, no h (physical)."""
        return self.UnitVelocity_in_cm_per_s ** 2

    @property
    def mdot_to_msun_per_yr(self):
        """Code accretion rate -> Msun/yr.

        Deliberately reproduces blackhole.cc:215 exactly, including the
        absence of an explicit 1/h, so that values derived here match
        column 5 of ``blackholes.txt`` bit for bit.
        """
        return (self.UnitMass_in_g / SOLAR_MASS) / (self.UnitTime_in_s / SEC_PER_YEAR)

    @property
    def velocity_to_kms(self):
        return self.UnitVelocity_in_cm_per_s / 1.0e5

    # -- a-dependent conversions -------------------------------------------

    def length_to_kpc(self, a=1.0):
        """Code (comoving) length -> physical kpc."""
        f = self.UnitLength_in_cm / KPC_IN_CM
        if self.comoving:
            f *= a / self.h
        return f

    def length_to_ckpc_per_h(self, a=1.0):
        """Code length -> comoving kpc/h (i.e. leave it alone, just rescale)."""
        return self.UnitLength_in_cm / KPC_IN_CM

    def density_to_cgs(self, a=1.0):
        """Code density -> g/cm^3 (physical)."""
        f = self.UnitDensity_in_cgs
        if self.comoving:
            f *= self.h ** 2 / a ** 3
        return f

    def density_to_nH_cgs(self, a=1.0, XH=HYDROGEN_MASSFRAC):
        """Code density -> hydrogen number density in cm^-3 (physical)."""
        return self.density_to_cgs(a) * XH / PROTONMASS

    def peculiar_velocity_to_kms(self, a=1.0):
        """Snapshot Velocities -> physical peculiar velocity in km/s.

        AREPO stores ``v_code = v_pec / sqrt(a)``, so the conversion carries
        ``sqrt(a)``.  For a non-comoving run this is the identity.
        """
        f = self.UnitVelocity_in_cm_per_s / 1.0e5
        if self.comoving:
            f *= np.sqrt(a)
        return f

    # -- derived physics ----------------------------------------------------

    def temperature(self, u_code, xe=None, XH=HYDROGEN_MASSFRAC):
        """Gas temperature in K from InternalEnergy and ElectronAbundance.

        mu = 4 / (1 + 3 XH + 4 XH xe) * m_p is the standard primordial mean
        molecular weight.  If ``xe`` is None a fully ionised primordial gas is
        assumed (xe = 1.16), which is fine for the hot phase but wrong by up to
        a factor ~2.4 for neutral gas -- always pass ElectronAbundance if the
        snapshot has it.
        """
        u = np.asarray(u_code, dtype=np.float64) * self.specific_energy_to_cgs
        if xe is None:
            xe = 1.0 + 2.0 * (1.0 - XH) / (4.0 * XH)  # fully ionised H + He
        mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * np.asarray(xe)) * PROTONMASS
        return GAMMA_MINUS1 * u * mu / BOLTZMANN

    def entropy_kev_cm2(self, u_code, rho_code, a=1.0, xe=None, XH=HYDROGEN_MASSFRAC):
        """X-ray-style entropy K = kT / n_e^(2/3), in keV cm^2."""
        T = self.temperature(u_code, xe=xe, XH=XH)
        rho = np.asarray(rho_code, dtype=np.float64) * self.density_to_cgs(a)
        if xe is None:
            xe = 1.0 + 2.0 * (1.0 - XH) / (4.0 * XH)
        ne = rho * XH / PROTONMASS * np.asarray(xe)
        kT_keV = BOLTZMANN * T / 1.602176634e-9
        return kT_keV / np.power(np.maximum(ne, 1e-30), 2.0 / 3.0)

    def eddington_rate_msun_per_yr(self, M_bh_msun, radiative_efficiency=0.1):
        """Mdot_Edd = 4 pi G M m_p / (eps_r c sigma_T), in Msun/yr."""
        M = np.asarray(M_bh_msun, dtype=np.float64) * SOLAR_MASS
        mdot_cgs = 4.0 * np.pi * GRAVITY * M * PROTONMASS / (radiative_efficiency * CLIGHT * THOMPSON)
        return mdot_cgs / SOLAR_MASS * SEC_PER_YEAR

    def eddington_luminosity_erg_per_s(self, M_bh_msun):
        M = np.asarray(M_bh_msun, dtype=np.float64) * SOLAR_MASS
        return 4.0 * np.pi * GRAVITY * M * PROTONMASS * CLIGHT / THOMPSON

    # -- cosmology ----------------------------------------------------------

    def redshift(self, a):
        """Expansion factor -> redshift.  Identity-safe for a non-comoving run."""
        if not self.comoving:
            return np.zeros_like(np.asarray(a, dtype=np.float64))
        return 1.0 / np.asarray(a, dtype=np.float64) - 1.0

    def scalefactor(self, z):
        return 1.0 / (1.0 + np.asarray(z, dtype=np.float64))

    def age_gyr(self, a):
        """Age of the universe at expansion factor ``a``, in Gyr.

        Exact for a flat matter + Lambda cosmology:

            t(a) = 2 / (3 H0 sqrt(OmegaL)) * asinh( sqrt(OmegaL/Om) a^{3/2} )

        Radiation is neglected, which is a <0.1% error for z < 20 and grows to
        ~1% by z = 100 -- irrelevant for anything a zoom simulation plots.
        """
        a = np.asarray(a, dtype=np.float64)
        if not self.comoving:
            return a * self.UnitTime_in_Gyr
        H0 = self.h * 100.0 * 1.0e5 / MPC_IN_CM  # s^-1
        pre = 2.0 / (3.0 * H0 * np.sqrt(self.OmegaLambda))
        t_s = pre * np.arcsinh(np.sqrt(self.OmegaLambda / self.Omega0) * a ** 1.5)
        return t_s / SEC_PER_YEAR / 1.0e9

    def lookback_gyr(self, a):
        return self.age_gyr(1.0) - self.age_gyr(a)

    def hubble_z(self, a):
        """H(a) in km/s/Mpc (matter + Lambda + curvature)."""
        a = np.asarray(a, dtype=np.float64)
        Ok = 1.0 - self.Omega0 - self.OmegaLambda
        return self.h * 100.0 * np.sqrt(self.Omega0 / a ** 3 + Ok / a ** 2 + self.OmegaLambda)

    def rho_crit_cgs(self, a=1.0):
        Hz = self.hubble_z(a) * 1.0e5 / MPC_IN_CM  # s^-1
        return 3.0 * Hz ** 2 / (8.0 * np.pi * GRAVITY)

    def delta_vir(self, a=1.0):
        """Bryan & Norman (1998) virial overdensity w.r.t. the *critical* density."""
        a = np.asarray(a, dtype=np.float64)
        Om_z = self.Omega0 / a ** 3 / (self.Omega0 / a ** 3 + self.OmegaLambda)
        x = Om_z - 1.0
        return 18.0 * np.pi ** 2 + 82.0 * x - 39.0 * x ** 2

    def __repr__(self):
        return (
            "Units(UnitLength_in_cm={:.6g}, UnitMass_in_g={:.6g}, "
            "UnitVelocity_in_cm_per_s={:.6g}, h={:.4f}, Om0={:.4f}, OmL={:.4f}, "
            "comoving={})".format(
                self.UnitLength_in_cm,
                self.UnitMass_in_g,
                self.UnitVelocity_in_cm_per_s,
                self.h,
                self.Omega0,
                self.OmegaLambda,
                self.comoving,
            )
        )


#: The THESAN / IllustrisTNG standard unit choice, used as the fallback when
#: no snapshot or parameter file is available.
THESAN_UNITS = Units(
    UnitLength_in_cm=KPC_IN_CM,
    UnitMass_in_g=1.989e43,
    UnitVelocity_in_cm_per_s=1.0e5,
    HubbleParam=0.6774,
    Omega0=0.3089,
    OmegaLambda=0.6911,
    OmegaBaryon=0.0486,
    ComovingIntegrationOn=True,
)
