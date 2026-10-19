"""
The local Voronoi mesh around each black hole, reconstructed from a snapshot.

With LOCAL_INJECTION_BLACK_HOLES the BH deposits its feedback through the
faces of its own Voronoi cell in a local mesh (voronoi_local.cc).  That mesh is
never written out, but a Voronoi tessellation is fixed by its generating
points, so it can be rebuilt from ``PartType0/Coordinates``.  For every BH this
module reconstructs

  * the injection point -- the BH position displaced by
    1e-12 * max(BoxSize) * (0.6, 0.48, 0.64), as fill_bh_data() does
    (local_injection.cc), so it never coincides with a gas generating point;
  * the BH's cell: faces, areas, area-weighted centroids, solid angles
    (Van Oosterom-Strackee), faces below MinSolidAngle dropped;
  * the Hopkins et al. (2018) face weights: WeightScalar_j, the share of the
    isotropic thermal energy each neighbour receives;
  * the collimated shares: the profile |cos alpha|^N about the jet axis, with
    N = -ln 2 / ln cos(theta) for theta = BHBipolarInj{Thermal,Kinetic}Theta,
    times WeightScalar_j, normalised over the BH's faces
    (bh_local_angular_profile / local_source_precompute<bh_data>);
  * the jet axis as blackhole_bipolar_inject() phase 0 builds it: the
    m^2-weighted angular momentum sum_j m_j^2 dr_j x (v_j - v_BH) of the gas
    within BH_Hsml, using the cold gas (u < BHBipolarInjColdTemp, fully ionised
    mu) when it holds more than BHBipolarInjColdFraction of that mass.

The geometry is the same as ``arepo/examples/BHInABox_3d/mesh_geometry.py``
(checked there against an exact rational construction to 5e-6); the
neighbourhood is chosen from the local cell size instead of a lattice spacing.
What a snapshot cannot give is a *measured* share: it holds many steps of
deposits mixed with hydro, so these are predictions of where the next deposit
goes.

    python3 -m thesan_tmz.local_mesh OUTPUT_DIR [-s SNAP] [--bh-id ID ...] [--max-bhs 10] [--html out.html]
"""

from __future__ import annotations

import numpy as np

from . import arepo_io as io
from .units import BOLTZMANN, GAMMA_MINUS1, HYDROGEN_MASSFRAC, PROTONMASS

MIN_SOLID_ANGLE = 4e-10 * np.pi             # MinSolidAngle, voronoi_local.cc
SOURCE_OFFSET_FRAC = 1.0e-12                # fill_bh_data(), local_injection.cc
SOURCE_OFFSET_DIR = np.array([0.6, 0.48, 0.64])


# ---- geometry (as examples/BHInABox_3d/mesh_geometry.py) ------------------
def _triangle_solid_angle(A, B, C):
    na, nb, nc = (np.linalg.norm(v) for v in (A, B, C))
    if min(na, nb, nc) <= 0.0:
        return 0.0
    num = float(np.dot(A, np.cross(B, C)))
    den = na * nb * nc + float(np.dot(A, B)) * nc + float(np.dot(B, C)) * na + float(np.dot(C, A)) * nb
    return 2.0 * np.arctan2(abs(num), den)


def _order_polygon(verts, normal):
    c = verts.mean(axis=0)
    n = normal / (np.linalg.norm(normal) or 1.0)
    a = verts[0] - c
    a = a - np.dot(a, n) * n
    if np.linalg.norm(a) == 0:
        return np.arange(len(verts))
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    d = verts - c
    return np.argsort(np.arctan2(d @ b, d @ a))


def _area_centroid(v):
    cent, tot = np.zeros(3), 0.0
    for i in range(1, len(v) - 1):
        ar = 0.5 * np.linalg.norm(np.cross(v[i] - v[0], v[i + 1] - v[0]))
        cent += ar * (v[0] + v[i] + v[i + 1]) / 3.0
        tot += ar
    return tot, (cent / tot if tot > 0 else v.mean(axis=0))


def _solid_angle(v):
    c = v.mean(axis=0)
    return sum(_triangle_solid_angle(c, v[i], v[(i + 1) % len(v)]) for i in range(len(v)))


def voronoi_cell(others, scale):
    """Voronoi cell of a point at the origin among ``others`` (positions relative
    to it), from the half-spaces n_j.x <= |d_j|/2.  Returns faces as dicts with
    ``nb`` (index into ``others``), ``verts``, ``area``, ``centroid``, ``omega``."""
    from scipy.optimize import linprog
    from scipy.spatial import HalfspaceIntersection

    dist = np.linalg.norm(others, axis=1)
    idx = np.where(dist > 0)[0]
    nrm = others[idx] / dist[idx, None]
    b = 0.5 * dist[idx]
    res = linprog(c=[0, 0, 0, -1], A_ub=np.hstack([nrm, np.ones((len(nrm), 1))]), b_ub=b,
                  bounds=[(None, None)] * 3 + [(0, None)])
    if not res.success or res.x[3] <= 0:
        raise RuntimeError("no interior point")
    hsi = HalfspaceIntersection(np.hstack([nrm, -b[:, None]]), res.x[:3])
    on = {}
    for vi, planes in enumerate(hsi.dual_facets):
        for j in planes:
            on.setdefault(j, []).append(vi)
    faces = []
    for j, vis in on.items():
        v = hsi.intersections[vis]
        v = v[np.unique(np.round(v / scale, 13), axis=0, return_index=True)[1]]
        if len(v) < 3:
            continue
        v = v[_order_polygon(v, nrm[j])]
        area, cent = _area_centroid(v)
        om = _solid_angle(v)
        if area > 0 and om >= MIN_SOLID_ANGLE:
            faces.append(dict(nb=int(idx[j]), verts=v, area=area, centroid=cent, omega=om))
    return faces


def hopkins_weights(faces):
    """Hopkins et al. (2018) Eqs. (9)-(11) with the generating-point fallback.
    Sets ``w`` (normalised vector) and ``wscalar`` (= |w|, the isotropic share)."""
    def split(vec):
        u = vec / (np.linalg.norm(vec) or 1.0)
        return np.where(u > 0, u, 0.0), np.where(u > 0, 0.0, u)

    for f in faces:
        f["xp"], f["xm"] = split(f["centroid"])
    sp = np.sum([f["omega"] * np.abs(f["xp"]) for f in faces], axis=0)
    sm = np.sum([f["omega"] * np.abs(f["xm"]) for f in faces], axis=0)
    if np.any(sp == 0) or np.any(sm == 0):
        for f in faces:
            f["xp"], f["xm"] = split(f["offset"])
        sp = np.sum([f["omega"] * np.abs(f["xp"]) for f in faces], axis=0)
        sm = np.sum([f["omega"] * np.abs(f["xm"]) for f in faces], axis=0)
    safe = lambda s: np.where(s ** 2 > 0, s ** 2, 1.0)
    fp = np.where(sp ** 2 > 0, np.sqrt(0.5 * (1 + sm ** 2 / safe(sp))), 1.0)
    fm = np.where(sm ** 2 > 0, np.sqrt(0.5 * (1 + sp ** 2 / safe(sm))), 1.0)
    for f in faces:
        f["w"] = f["omega"] * (f["xp"] * fp + f["xm"] * fm)
    tot = sum(np.linalg.norm(f["w"]) for f in faces)
    for f in faces:
        f["w"] = f["w"] / tot
        f["wscalar"] = float(np.linalg.norm(f["w"]))


def profile_exponent(theta_deg):
    """N with |cos theta|^N = 1/2 (bh_local_profile_exponent); 0 = isotropic."""
    c = np.cos(np.radians(theta_deg))
    return 0.0 if not (0.0 < c < 1.0) else -np.log(2.0) / np.log(c)


def collimated_shares(faces, axis, theta_deg):
    """Share of a collimated channel per face, as local_source_precompute<bh_data>:
    |cos alpha_j|^N * WeightScalar_j / sum_k(...), alpha_j the angle between
    the face's weight vector and the axis."""
    if axis is None:
        return None
    n = profile_exponent(theta_deg)
    mu = np.clip([abs(np.dot(f["w"], axis)) / (np.linalg.norm(f["w"]) or 1.0) for f in faces], 0.0, 1.0)
    prof = np.where(mu > 0, mu ** n, 0.0)                    # bh_local_angular_profile: mu <= 0 gives 0
    s = prof * np.array([f["wscalar"] for f in faces])
    return s / s.sum() if s.sum() > 0 else None


def injection_point(bhpos, box):
    x = np.asarray(bhpos, dtype=np.float64) + SOURCE_OFFSET_FRAC * box * SOURCE_OFFSET_DIR
    return np.where(x >= box, x - box, x)


# ---- per-BH reconstruction --------------------------------------------------
def bh_axis(gas, d_gas, bh_vel, hsml, params, u_to_cgs):
    """The jet axis as blackhole_bipolar_inject() phase 0 builds it (see module
    docstring).  ``d_gas`` are gas offsets from the BH (signed, periodic).
    Returns (unit axis or None, used_cold_disk)."""
    r = np.linalg.norm(d_gas, axis=1)
    sel = (r < hsml) & (r > 0)
    if sel.sum() < 2:
        return None, False
    m = gas["Masses"][sel]
    dv = gas["Velocities"][sel] - bh_vel           # a common velocity factor does not change the direction
    J = (m ** 2)[:, None] * np.cross(d_gas[sel], dv)
    cold_disk = False
    if "BHBipolarInjColdTemp" in params and "InternalEnergy" in gas:
        mu = 4.0 / (8.0 - 5.0 * (1.0 - HYDROGEN_MASSFRAC))          # fully ionised, as bipolar_u_to_temp_fac()
        T = gas["InternalEnergy"][sel] * u_to_cgs * mu * PROTONMASS / BOLTZMANN * GAMMA_MINUS1
        cold = T < float(params["BHBipolarInjColdTemp"])
        if m.sum() > 0 and m[cold].sum() / m.sum() > float(params.get("BHBipolarInjColdFraction", np.inf)):
            cold_disk, J = True, J[cold]           # bipolar_finalise_axis(): J_cold even if it is zero
    Jt = J.sum(axis=0)
    nJ = np.linalg.norm(Jt)
    return (Jt / nJ if np.isfinite(nJ) and nJ > 0 else None), cold_disk


def complete_cell(d_all, r_all, scale, max_tries=12):
    """The Voronoi cell of the origin among the points d_all (|d_all| = r_all), built from
    as few points as possible but provably complete.  A point farther than twice the cell's
    farthest vertex cannot cut the cell (its bisector plane lies beyond that vertex), so the
    search radius is widened until it reaches 2 x the vertex radius.  Closing to 4 pi is NOT
    a completeness test: any bounded cell does, and one built from too few points is too
    big (a BH at the edge of a galaxy got a cell several times too large that way).
    Returns (faces, indices of the candidate points) or (None, None)."""
    radius = 3.0 * scale
    for _ in range(max_tries):
        cand = np.where(r_all < radius)[0]
        try:
            faces = voronoi_cell(d_all[cand], scale)
        except Exception:                                       # unbounded: too few points
            radius *= 2.0
            continue
        r_vert = max(float(np.max(np.linalg.norm(f["verts"], axis=1))) for f in faces)
        if 2.0 * r_vert <= radius or cand.size == r_all.size:
            if abs(sum(f["omega"] for f in faces) / (4 * np.pi) - 1.0) > 1e-3:
                return None, None
            return faces, cand
        radius = 2.05 * r_vert
    return None, None


def bh_local_mesh(snap, k, gas=None, params=None):
    """Reconstruct the local cell of BH number ``k`` (index into PartType5).

    Returns a dict: ``id``, ``src`` (code), ``faces`` (each with ``nb_global``,
    ``offset``, ``wscalar``, ``w``, ``omega``, ``area``, ``verts``), ``host``
    (gas index of the nearest cell), ``scale`` (local cell size, code),
    ``omega_tot``, ``axis``, ``cold_disk``, ``coll_therm``, ``coll_kin`` (or None).
    """
    params = params if params is not None else snap.parameters
    if gas is None:
        gas = snap.part(io.PT_GAS, ["Coordinates", "Masses", "Velocities", "InternalEnergy"])
    bh = snap.part(io.PT_BH, ["Coordinates", "ParticleIDs", "Velocities", "BH_Hsml"])
    box = snap.boxsize or 0.0
    src = injection_point(bh["Coordinates"][k], box) if box else np.asarray(bh["Coordinates"][k], dtype=np.float64)
    d_all = io.periodic_delta(gas["Coordinates"], src, box)
    r_all = np.linalg.norm(d_all, axis=1)
    order = np.argsort(r_all)
    scale = float(r_all[order[min(15, order.size - 1)]])        # ~ the 16th nearest cell
    faces, near = complete_cell(d_all, r_all, scale)
    if faces is None:
        raise RuntimeError("the cell of BH %d could not be built completely" % bh["ParticleIDs"][k])
    omega_tot = sum(f["omega"] for f in faces)
    for f in faces:
        g = int(near[f["nb"]])
        f["nb_global"], f["offset"] = g, d_all[g]
    hopkins_weights(faces)
    faces.sort(key=lambda f: -f["wscalar"])

    hsml = float(bh["BH_Hsml"][k]) if "BH_Hsml" in bh else 3.0 * scale
    axis, cold_disk = bh_axis(gas, io.periodic_delta(gas["Coordinates"], bh["Coordinates"][k], box),
                              bh["Velocities"][k] if "Velocities" in bh else np.zeros(3), hsml, params,
                              snap.units.specific_energy_to_cgs)
    # the BH's surroundings: nearest gas cell, and the mean density of the 16 nearest
    n16 = order[:16]
    rho16 = float(np.sum(gas["Masses"][n16])) / (4.0 / 3.0 * np.pi * scale ** 3) if scale > 0 else np.nan
    out = dict(id=int(bh["ParticleIDs"][k]), src=src, faces=faces, host=int(order[0]), scale=scale,
               omega_tot=omega_tot, axis=axis, cold_disk=cold_disk, coll_therm=None, coll_kin=None,
               r_nearest=float(r_all[order[0]]), rho16=rho16)
    cfg = snap.config
    if "BH_BIPOLAR_INJECTION_THERMAL" in cfg and "BHBipolarInjThermalTheta" in params:
        out["coll_therm"] = collimated_shares(faces, axis, float(params["BHBipolarInjThermalTheta"]))
    if "BH_BIPOLAR_INJECTION_KINETIC" in cfg and "BHBipolarInjKineticTheta" in params:
        out["coll_kin"] = collimated_shares(faces, axis, float(params["BHBipolarInjKineticTheta"]))
    return out


def all_bh_meshes(snap, bh_ids=None, max_bhs=10):
    """Reconstruct the local cell of the requested BHs (default: the ``max_bhs``
    most massive).  Returns a list of :func:`bh_local_mesh` dicts; a BH whose
    cell cannot be built is reported and skipped."""
    bh = snap.part(io.PT_BH, ["ParticleIDs", "BH_Mass", "Masses"])
    if not bh.get("count"):
        return []
    mass = np.asarray(bh.get("BH_Mass", bh.get("Masses")), dtype=np.float64)
    if bh_ids:
        ks = [int(np.where(bh["ParticleIDs"] == i)[0][0]) for i in bh_ids if np.any(bh["ParticleIDs"] == i)]
    else:
        ks = list(np.argsort(-mass)[:max_bhs])
    gas = snap.part(io.PT_GAS, ["Coordinates", "Masses", "Velocities", "InternalEnergy"])
    out = []
    for k in ks:
        try:
            out.append(bh_local_mesh(snap, int(k), gas=gas))
        except Exception as exc:
            print("  [local_mesh] BH %d skipped: %s" % (bh["ParticleIDs"][k], exc))
    return out


# ---- the 3D page ---------------------------------------------------------------
def _figure(go, snap, mesh, gas):
    """One interactive 3D scene for one BH.  Coordinates: physical kpc from the BH."""
    L = snap.units.length_to_kpc(snap.a)
    faces = mesh["faces"]
    shares = {"isotropic thermal": np.array([f["wscalar"] for f in faces])}
    if mesh["coll_therm"] is not None:
        shares["collimated thermal"] = mesh["coll_therm"]
    if mesh["coll_kin"] is not None:
        shares["kinetic"] = mesh["coll_kin"]
    names = list(shares)
    cmax = {nm: float(v.max()) or 1.0 for nm, v in shares.items()}
    rcell = max(float(np.max(np.linalg.norm(f["verts"], axis=1))) for f in faces) * L   # the cell's extent
    fig = go.Figure()
    for i, f in enumerate(faces):
        v = f["verts"] * L
        nv = len(v)
        hover = "<br>".join("%s share %.4f" % (nm, shares[nm][i]) for nm in names)
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=[0] * (nv - 2), j=list(range(1, nv - 1)),
                                k=list(range(2, nv)), intensity=np.full(nv, shares[names[0]][i]), cmin=0, cmax=cmax[names[0]],
                                colorscale="Viridis", opacity=0.8, flatshading=True, showscale=(i == 0),
                                colorbar=dict(title="share", len=0.6), name="BH cell", legendgroup="cell",
                                showlegend=(i == 0),
                                hovertemplate="<b>face %d</b> (gas cell %d)<br>%s<br>solid angle %.3f sr<extra></extra>"
                                              % (i, f["nb_global"], hover, f["omega"])))
    n_face_traces = len(faces)
    xs, ys, zs = [], [], []
    for f in faces:
        loop = np.vstack([f["verts"], f["verts"][:1]]) * L
        xs += list(loop[:, 0]) + [None]; ys += list(loop[:, 1]) + [None]; zs += list(loop[:, 2]) + [None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="black", width=3),
                               name="cell edges", hoverinfo="skip"))
    # surrounding gas: the BH cell's neighbours and the next shell, coloured by temperature
    d = io.periodic_delta(gas["Coordinates"], mesh["src"], snap.boxsize) * L
    r = np.linalg.norm(d, axis=1)
    sel = np.where(r < 2.2 * rcell)[0]            # every face neighbour lies within 2 rcell
    T = snap.units.temperature(gas["InternalEnergy"][sel])
    fig.add_trace(go.Scatter3d(x=d[sel, 0], y=d[sel, 1], z=d[sel, 2], mode="markers",
                               marker=dict(size=4, color=np.log10(np.maximum(T, 1)), colorscale="Inferno", opacity=0.6,
                                           colorbar=dict(title="log T", x=1.12, len=0.6)),
                               name="gas cells (log T)", text=["T = %.3g K" % x for x in T],
                               hovertemplate="%{text}<extra></extra>"))
    # gas velocity relative to the BH, peculiar + Hubble flow, physical km/s
    bh = snap.part(io.PT_BH, ["ParticleIDs", "Velocities"])
    kb = int(np.where(bh["ParticleIDs"] == mesh["id"])[0][0])
    f_v = snap.units.peculiar_velocity_to_kms(snap.a)
    dv = (gas["Velocities"][sel] - bh["Velocities"][kb]) * f_v + snap.units.hubble_kms_per_kpc(snap.a) * d[sel]
    vmax = float(np.max(np.linalg.norm(dv, axis=1))) if sel.size else 1.0
    arrow = 1.2 * rcell
    vv = dv * (arrow / max(vmax, 1e-30))
    fig.add_trace(go.Cone(x=d[sel, 0], y=d[sel, 1], z=d[sel, 2], u=vv[:, 0], v=vv[:, 1], w=vv[:, 2], anchor="tail",
                          sizemode="raw", sizeref=1.0, colorscale="Blues", showscale=False, visible="legendonly",
                          name="gas velocity rel. to BH (max %.3g km/s)" % vmax, showlegend=True,
                          hovertext=["|v| = %.3g km/s" % x for x in np.linalg.norm(dv, axis=1)], hoverinfo="text"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="BH",
                               marker=dict(size=8, color="red", symbol="diamond", line=dict(color="black", width=1))))
    if mesh["axis"] is not None:
        ax_ = mesh["axis"] * 2.0 * rcell
        fig.add_trace(go.Scatter3d(x=[-ax_[0], ax_[0]], y=[-ax_[1], ax_[1]], z=[-ax_[2], ax_[2]], mode="lines",
                                   line=dict(color="red", width=5, dash="dash"),
                                   name="jet axis%s" % (" (cold disk)" if mesh["cold_disk"] else "")))
    # buttons: colour the faces by each channel's share
    buttons = [dict(label=nm, method="restyle",
                    args=[{"intensity": [np.full(len(f["verts"]), shares[nm][i]) for i, f in enumerate(faces)],
                           "cmax": [cmax[nm]] * n_face_traces},
                          list(range(n_face_traces))]) for nm in names]
    fig.update_layout(
        title="BH %d: %d faces, total solid angle %.6f x 4π; cell radius %.3g kpc"
              % (mesh["id"], len(faces), mesh["omega_tot"] / (4 * np.pi), rcell),
        height=720, margin=dict(l=0, r=0, t=90, b=0), legend=dict(orientation="h", y=-0.02),
        scene=dict(xaxis_title="x [kpc]", yaxis_title="y [kpc]", zaxis_title="z [kpc]", aspectmode="data"),
        updatemenus=[dict(type="buttons", direction="right", x=0.0, y=1.0, xanchor="left", yanchor="bottom",
                          pad=dict(b=4), buttons=buttons)]
        if len(names) > 1 else [])
    return fig


def _deposit_note(config):
    if "LOCAL_INJECTION_BLACK_HOLES" in config:
        return ""
    return ("<p><b>This run does not deposit BH feedback through this cell</b>: LOCAL_INJECTION_BLACK_HOLES "
            "is off, so the BH uses the kernel (or the cone) around BH_Hsml. The shares below are what local "
            "injection would give for the same gas.</p>")


def write_page(snap, meshes, path, offline=False):
    """One HTML page: a 3D scene and a face table per BH."""
    import plotly.io as pio
    import plotly.graph_objects as go

    gas = snap.part(io.PT_GAS, ["Coordinates", "Masses", "Velocities", "InternalEnergy"])
    parts = []
    for n, m in enumerate(meshes):
        rows = "".join("<tr><td>%d</td><td>%d</td><td>%.4f</td><td>%.4f</td>%s%s</tr>" % (
            i, f["nb_global"], f["omega"], f["wscalar"],
            "<td>%.4f</td>" % m["coll_therm"][i] if m["coll_therm"] is not None else "",
            "<td>%.4f</td>" % m["coll_kin"][i] if m["coll_kin"] is not None else "") for i, f in enumerate(m["faces"]))
        head = ("<tr><th>face</th><th>gas cell</th><th>solid angle [sr]</th><th>isotropic share</th>%s%s</tr>"
                % ("<th>collimated thermal</th>" if m["coll_therm"] is not None else "",
                   "<th>kinetic</th>" if m["coll_kin"] is not None else ""))
        wsum = np.linalg.norm(np.sum([f["w"] for f in m["faces"]], axis=0))
        L = snap.units.length_to_kpc(snap.a)
        nH = m["rho16"] * snap.units.density_to_nH_cgs(snap.a)
        env = ("nearest gas cell %.3g kpc away; mean gas density of the 16 nearest cells n<sub>H</sub> &asymp; "
               "%.2g cm<sup>-3</sup>" % (m["r_nearest"] * L, nH))
        if nH < 1e-4:
            env += ("<br><b>This BH sits in very low-density gas</b> (around the intergalactic or outer-halo "
                    "density), so its cell is large. It is probably not in the centre of a galaxy: check its "
                    "offset and host in bh_table.txt / fig 16.")
        parts.append("<h2>BH %d</h2><p>host gas cell %d; |Σ w<sub>j</sub>| = %.2g (0 = no net direction); "
                     "jet axis %s<br>%s</p>" % (m["id"], m["host"], wsum,
                                          "[%.3f, %.3f, %.3f]%s" % (tuple(m["axis"]) + (" from the cold disk" if m["cold_disk"] else "",))
                                          if m["axis"] is not None else "undefined", env)
                     + pio.to_html(_figure(go, snap, m, gas), full_html=False,
                                   include_plotlyjs=((True if offline else "cdn") if n == 0 else False))
                     + "<details><summary>face table</summary><table>%s%s</table></details>" % (head, rows))
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>BH local meshes</title><style>"
            "body{font-family:system-ui,sans-serif;max-width:1200px;margin:auto;padding:1em}"
            "td,th{padding:2px 8px;text-align:right;font-size:13px}p{color:#444}</style></head><body>"
            "<h1>Local Voronoi mesh around each black hole</h1>"
            "<p>%s, z = %.3f. Reconstructed from the gas generating points: the BH's cell in its local mesh, "
            "faces coloured by the predicted share of the next deposit (buttons switch channel). A snapshot "
            "cannot show measured shares; these are predictions. Coordinates: physical kpc from the BH.</p>%s"
            % (snap.files[0], snap.z, _deposit_note(snap.config)) + "".join(parts) + "</body></html>")
    with open(path, "w") as fh:
        fh.write(html)
    print("  [local_mesh] wrote %s (%d black holes)" % (path, len(meshes)))
    return path


def missing_dependencies():
    """The optional packages this module needs and that are not installed:
    scipy (the half-space intersection) and plotly (the page)."""
    import importlib.util

    return [m for m in ("scipy", "plotly") if importlib.util.find_spec(m) is None]


def local_mesh_page(snap, path, max_bhs=10, bh_ids=None, offline=False):
    """Reconstruct the local cell of the ``max_bhs`` most massive BHs (or
    ``bh_ids``) and write the 3D page.  Returns the path, or None when there is
    nothing to show or scipy/plotly are not installed."""
    missing = missing_dependencies()
    if missing:
        print("  [local_mesh] %s not installed -- skipping the 3D page" % " and ".join(missing))
        return None
    meshes = all_bh_meshes(snap, bh_ids=bh_ids, max_bhs=max_bhs)
    if not meshes:
        print("  [local_mesh] no black hole cell could be built")
        return None
    return write_page(snap, meshes, path, offline=offline)


def main(argv=None):
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Local Voronoi mesh around each BH, as an interactive 3D page.")
    ap.add_argument("outdir")
    ap.add_argument("-s", "--snap", type=int, default=None, help="snapshot number (default: the last)")
    ap.add_argument("--bh-id", type=int, nargs="*", default=None)
    ap.add_argument("--max-bhs", type=int, default=10)
    ap.add_argument("--html", default=None)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)
    nums = io.available_snapshots(a.outdir)
    snap = io.Snapshot(a.outdir, a.snap if a.snap is not None else nums[-1])
    meshes = all_bh_meshes(snap, bh_ids=a.bh_id, max_bhs=a.max_bhs)
    if not meshes:
        print("  [local_mesh] no black hole cell could be built")
        return None
    L = snap.units.length_to_kpc(snap.a)
    for m in meshes:
        print("  BH %d: %d faces, solid angle %.8f x 4pi, max isotropic share %.3f, axis %s; nearest gas %.3g kpc, "
              "n_H(16 nearest) %.2g cm^-3"
              % (m["id"], len(m["faces"]), m["omega_tot"] / (4 * np.pi), m["faces"][0]["wscalar"],
                 None if m["axis"] is None else np.round(m["axis"], 3), m["r_nearest"] * L,
                 m["rho16"] * snap.units.density_to_nH_cgs(snap.a)))
    out = a.html or os.path.join(os.path.dirname(os.path.abspath(a.outdir)), "bh_local_mesh_%03d.html" % snap.num)
    return write_page(snap, meshes, out, offline=a.offline)


if __name__ == "__main__":
    main()
