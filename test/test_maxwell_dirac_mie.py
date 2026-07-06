r"""Mie validation for the Helsing-Rosen 3D Maxwell-Dirac operator.

Bohren-Huffman Mie reference for a homogeneous sphere, using the same HR field
scaling as the operator.

Notation:

* ``k_int`` and ``k_ext`` are the interior and exterior wavenumbers, with
  :math:`k = \omega\sqrt{\epsilon\mu}` in each region.
* ``k_hat = k_int/k_ext`` is the relative refractive index, written ``m``
  in the Bohren-Huffman formulas and function arguments below;
  ``mu_hat = mu_int/mu_ext`` is the permeability ratio, and
  ``x = k_ext*a`` is the size parameter.
* Magnetic fields are in HR units:
  :math:`\widetilde B = B/\sqrt{\epsilon\mu}`.
* Incident field:
  :math:`E_0 = e_1\exp(i k_{\rm ext} x_3)`,
  :math:`\widetilde B_0 = e_2\exp(i k_{\rm ext} x_3)`.
* The exterior representation returns
  :math:`E_{\rm sca}, \widetilde B_{\rm sca}`.  Add the incident field before
  comparing with the total Mie field.
"""
from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from arraycontext import flatten

from pytential import GeometryCollection, bind, sym


# {{{ Bohren & Huffman Lorenz-Mie reference

def _n_terms(m, x, *, guard=1):
    r"""Series truncation

    .. math::

        n_{\max} = \left\lceil x_s + 4.05\, x_s^{1/3} + 2 \right\rceil,
        \qquad x_s = \max(x, |m|x).

    ``guard`` adds margin for near-surface evaluation, which converges
    more slowly than far-field sums.
    """
    x_stop = max(x, abs(m) * x)
    return int(np.ceil(x_stop + 4.05 * x_stop ** (1 / 3) + 2)) + guard


def _sph(n_max, z):
    """Return ``j_n(z)``, ``y_n(z)``, ``j_n'(z)``, ``y_n'(z)`` up to ``n_max``."""
    n = np.arange(n_max + 1)
    jn = spherical_jn(n, z)
    yn = spherical_yn(n, z)
    djn = spherical_jn(n, z, derivative=True)
    dyn = spherical_yn(n, z, derivative=True)
    return jn, yn, djn, dyn


def _riccati(n_max, x, m):
    r"""Riccati-Bessel functions used in the coefficient formulas.

    .. math::

        \psi_n(z) = z j_n(z), \qquad
        \xi_n(z) = z h_n^{(1)}(z) = z(j_n(z) + i y_n(z)).
    """
    jx, yx, djx, dyx = _sph(n_max, x)
    jmx, _, djmx, _ = _sph(n_max, m * x)
    psi_x = x * jx
    dpsi_x = jx + x * djx
    hx = jx + 1j * yx
    dhx = djx + 1j * dyx
    xi_x = x * hx
    dxi_x = hx + x * dhx
    psi_mx = (m * x) * jmx
    dpsi_mx = jmx + (m * x) * djmx
    return psi_x, dpsi_x, xi_x, dxi_x, psi_mx, dpsi_mx


def mie_ab(m, x, n_max, *, mu_hat=1.0):
    r"""Scattered Mie coefficients for ``n = 1 .. n_max``.

    .. math::

        a_n =
        \frac{m\psi_n(mx)\psi_n'(x)
              - \hat\mu\,\psi_n(x)\psi_n'(mx)}
             {m\psi_n(mx)\xi_n'(x)
              - \hat\mu\,\xi_n(x)\psi_n'(mx)},

        b_n =
        \frac{\hat\mu\,\psi_n(mx)\psi_n'(x)
              - m\psi_n(x)\psi_n'(mx)}
             {\hat\mu\,\psi_n(mx)\xi_n'(x)
              - m\xi_n(x)\psi_n'(mx)}.
    """
    psi_x, dpsi_x, xi_x, dxi_x, psi_mx, dpsi_mx = _riccati(n_max, x, m)
    n = np.arange(1, n_max + 1)
    a = ((m * psi_mx[n] * dpsi_x[n] - mu_hat * psi_x[n] * dpsi_mx[n])
         / (m * psi_mx[n] * dxi_x[n] - mu_hat * xi_x[n] * dpsi_mx[n]))
    b = ((mu_hat * psi_mx[n] * dpsi_x[n] - m * psi_x[n] * dpsi_mx[n])
         / (mu_hat * psi_mx[n] * dxi_x[n] - m * xi_x[n] * dpsi_mx[n]))
    return a, b


def mie_cd(m, x, n_max, *, mu_hat=1.0):
    r"""Interior Mie coefficients for ``n = 1 .. n_max``.

    .. math::

        c_n =
        \frac{m\hat\mu\,(\psi_n(x)\xi_n'(x)
              - \xi_n(x)\psi_n'(x))}
             {\hat\mu\,\psi_n(mx)\xi_n'(x)
              - m\xi_n(x)\psi_n'(mx)},

        d_n =
        \frac{m\hat\mu\,(\psi_n(x)\xi_n'(x)
              - \xi_n(x)\psi_n'(x))}
             {m\psi_n(mx)\xi_n'(x)
              - \hat\mu\,\xi_n(x)\psi_n'(mx)}.
    """
    psi_x, dpsi_x, xi_x, dxi_x, psi_mx, dpsi_mx = _riccati(n_max, x, m)
    n = np.arange(1, n_max + 1)
    num = m * mu_hat * (psi_x[n] * dxi_x[n] - xi_x[n] * dpsi_x[n])
    c = num / (mu_hat * psi_mx[n] * dxi_x[n] - m * xi_x[n] * dpsi_mx[n])
    d = num / (m * psi_mx[n] * dxi_x[n] - mu_hat * xi_x[n] * dpsi_mx[n])
    return c, d


def _pi_tau(n_max, cos_theta):
    r"""Angular Mie functions from the upward recurrence.

    With ``mu = cos(theta)``:

    .. math::

        \pi_n =
        \frac{2n - 1}{n - 1}\mu\pi_{n-1}
        - \frac{n}{n - 1}\pi_{n-2},
        \qquad
        \tau_n = n\mu\pi_n - (n + 1)\pi_{n-1}.

    Seeded by ``pi_0 = 0`` and ``pi_1 = 1``.
    """
    pi_n = np.zeros(n_max + 1)
    tau_n = np.zeros(n_max + 1)
    if n_max >= 1:
        pi_n[1] = 1.0
        tau_n[1] = cos_theta
    for nn in range(2, n_max + 1):
        pi_n[nn] = (((2 * nn - 1) / (nn - 1)) * cos_theta * pi_n[nn - 1]
                    - (nn / (nn - 1)) * pi_n[nn - 2])
        tau_n[nn] = nn * cos_theta * pi_n[nn] - (nn + 1) * pi_n[nn - 1]
    return pi_n, tau_n


def _vsh_partial_wave(k, c_m, c_n, points, *, radial):
    r"""Evaluate the ``m = 1`` vector-spherical-harmonic sum.

    .. math::

        E = \sum_n e_n (c_m[n] M_{o1n} - i c_n[n] N_{e1n}),

        \widetilde B =
        \frac{1}{i}\sum_n e_n(c_m[n]N_{o1n} - i c_n[n]M_{e1n}),

    where ``e_n = i^n (2n + 1)/(n(n + 1))``.  ``radial="reg"`` uses
    ``j_n(k r)``; ``radial="out"`` uses ``h_n^(1)(k r)``.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    n_max = len(c_m)
    n = np.arange(1, n_max + 1)
    e_n = (1j ** n) * (2 * n + 1) / (n * (n + 1))

    e_xyz = np.zeros((len(points), 3), dtype=complex)
    b_xyz = np.zeros((len(points), 3), dtype=complex)

    for idx, (px, py, pz) in enumerate(points):
        r = np.sqrt(px * px + py * py + pz * pz)
        if r < 1.0e-1:
            raise ValueError(
                    "_vsh_partial_wave needs an accurate small r implementation")

        theta = np.arccos(np.clip(pz / r, -1.0, 1.0))
        phi = np.arctan2(py, px)

        cth, sth = np.cos(theta), np.sin(theta)
        cph, sph = np.cos(phi), np.sin(phi)

        rho = k * r

        jn, yn, djn, dyn = _sph(n_max, rho)

        if radial == "out":
            zn, dzn = (jn + 1j * yn)[n], (djn + 1j * dyn)[n]
        else:
            zn, dzn = jn[n], djn[n]

        d_n = zn / rho + dzn

        pi_n, tau_n = _pi_tau(n_max, cth)
        pi_n, tau_n = pi_n[n], tau_n[n]

        radial_term = n * (n + 1) * sth * pi_n * zn / rho

        # Bohren & Huffman eq. 4.50
        zero = np.zeros_like(zn)
        m_o1n = np.stack([zero, cph * pi_n * zn, -sph * tau_n * zn])
        m_e1n = np.stack([zero, -sph * pi_n * zn, -cph * tau_n * zn])
        n_e1n = np.stack([cph * radial_term, cph * tau_n * d_n,
                          -sph * pi_n * d_n])
        n_o1n = np.stack([sph * radial_term, sph * tau_n * d_n,
                          cph * pi_n * d_n])

        # E = sum_n e_n (c_m M_o1n - i c_n N_e1n).
        e_sph = (np.sum(e_n * c_m * m_o1n, axis=1)
                  - 1j * np.sum(e_n * c_n * n_e1n, axis=1))

        # B~ = (1/i) sum_n e_n (c_m N_o1n - i c_n M_e1n).
        b_sph = (np.sum(e_n * c_m * n_o1n, axis=1)
                  - 1j * np.sum(e_n * c_n * m_e1n, axis=1)) / 1j

        # convert spherical to Cartesian basis
        r_hat = np.array([sth * cph, sth * sph, cth])
        theta_hat = np.array([cth * cph, cth * sph, -sth])
        phi_hat = np.array([-sph, cph, 0.0])
        sph_to_xyz = np.column_stack([r_hat, theta_hat, phi_hat])

        e_xyz[idx] = sph_to_xyz @ e_sph
        b_xyz[idx] = sph_to_xyz @ b_sph
    return e_xyz, b_xyz


def lorenz_mie_scattered_field(k_ext, m, radius, points, n_max=None, *,
                               mu_hat=1.0):
    """Mie scattered field."""
    x = k_ext * radius
    if n_max is None:
        n_max = _n_terms(m, x, guard=6)
    a, b = mie_ab(m, x, n_max, mu_hat=mu_hat)
    return _vsh_partial_wave(k_ext, -b, -a, points, radial="out")


def _incident_fields(k_ext, points, *, d=(0.0, 0.0, 1.0), p=(1.0, 0.0, 0.0)):
    r"""Incident plane wave, with the magnetic field in HR units:

    .. math::

        E_0 = p\exp(i k_{\rm ext} d\cdot x), \qquad
        \widetilde B_0 = (d\times p)\exp(i k_{\rm ext} d\cdot x).

    The second relation holds in any homogeneous medium: from
    :math:`\nabla\times E = ik\widetilde B`, the material factors cancel
    in HR units.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    d, p = np.asarray(d, dtype=float), np.asarray(p, dtype=float)
    phase = np.exp(1j * k_ext * (points @ d))[:, None]
    return p * phase, np.cross(d, p) * phase


# }}}


# {{{ reference self-checks

def test_incident_plane_wave():
    k_ext = 1.7
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.0, (8, 3)) + np.array([0.0, 0.0, 2.0])
    # truncation must cover k r at the farthest evaluation point
    r_max = np.max(np.linalg.norm(pts, axis=1))
    n_max = _n_terms(1.0, k_ext * r_max, guard=8)

    ones = np.ones(n_max)
    e_inc, b_inc = _vsh_partial_wave(k_ext, ones, ones, pts, radial="reg")
    pw_e, pw_b = _incident_fields(k_ext, pts)
    assert np.max(np.abs(e_inc - pw_e)) < 1e-8
    assert np.max(np.abs(b_inc - pw_b)) < 1e-8


# }}}


# {{{ HR transmission solve vs Mie

def _make_sphere(actx, nref, *, qbx_order, target_order):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
        InterpolatoryQuadratureGroupFactory,
    )
    from meshmode.mesh import TensorProductElementGroup
    from meshmode.mesh.generation import generate_sphere

    from pytential.qbx import QBXLayerPotentialSource

    mesh = generate_sphere(
        1.0, target_order, uniform_refinement_rounds=nref,
        group_cls=TensorProductElementGroup)
    pre_discr = Discretization(
        actx, mesh, InterpolatoryQuadratureGroupFactory(target_order))
    qbx = QBXLayerPotentialSource(
        pre_discr, fine_order=4 * qbx_order, qbx_order=qbx_order,
        fmm_order=qbx_order + 5, target_association_tolerance=0.1)
    places = GeometryCollection(qbx, auto_where="qbx")
    h_max = actx.to_numpy(bind(places, sym.h_max(qbx.ambient_dim))(actx))
    return places, float(h_max)


def run_dirac_maxwell_sphere_mie(actx, eps_in, *,
                                 nrefs=(2, 3), qbx_order=4):
    r"""Compare the HR Maxwell-Dirac sphere solve against Lorenz-Mie.

    Unit sphere with ``eps_hat = eps_in`` and ``mu_hat = 1``.  Solve

    .. math::

        (I + P E_{k_{\rm int}}N' - N E_{k_{\rm ext}}P')h = 2Nf_0,

    recover the exterior scattered trace, add the incident field, and compare
    against the Lorenz-Mie total field.
    """
    from pytools import obj_array
    from pytools.convergence import EOCRecorder

    from pytential.linalg.gmres import gmres
    from pytential.symbolic.pde.maxwell import MaxwellDomain
    from pytential.symbolic.pde.maxwell.dirac import DiracMaxwellTransmissionOperator

    omega = 1.5
    op = DiracMaxwellTransmissionOperator.from_domains(
        omega,
        [MaxwellDomain(epsilon=1.0, mu=1.0),        # exterior
         MaxwellDomain(epsilon=eps_in, mu=1.0)])    # interior
    k_ext = float(np.real(op.k_ext))

    e_eoc = EOCRecorder()
    b_eoc = EOCRecorder()
    print(f"\neps_in={eps_in!r}", flush=True)
    print(f"{'nref':>4} {'h_max':>12} {'ndofs':>8} {'gmres':>7} "
          f"{'E_err':>12} {'B_err':>12}", flush=True)
    for nref in nrefs:
        print(f"running nref={nref} ...", flush=True)
        places, h_max = _make_sphere(
            actx, nref, qbx_order=qbx_order, target_order=qbx_order+1)
        discr = places.get_discretization("qbx")

        unk = op.make_unknown("sigma")
        bound_op = bind(places, op.operator(unk))

        nodes = sym.nodes(3, dofdesc="qbx").as_vector()
        phase = sym.exp(1j * k_ext * nodes[2])
        zero = 0 * nodes[0]
        rhs = bind(places, op.rhs([phase, zero, zero],
                                  [zero, phase, zero],
                                  dofdesc="qbx"))(actx)

        zero_dof = discr.zeros(actx, dtype=np.complex128)
        rhs = obj_array.new_1d([
            c + zero_dof if np.isscalar(c) else c
            for c in rhs])

        result = gmres(
            bound_op.scipy_op(actx, "sigma", np.complex128),
            rhs, tol=1e-12)
        sigma = result.solution

        field = bind(places, op.scattered_volume_field(
            unk, side="exterior", qbx_forced_limit=+1,
            source="qbx", target="qbx"))(actx, sigma=sigma)
        e_sca, b_sca = field[:3], field[3:]

        points = np.stack([
            actx.to_numpy(flatten(c, actx))
            for c in actx.thaw(discr.nodes())]).T
        e_sca_mie, b_sca_mie = lorenz_mie_scattered_field(
            k_ext, op.k_hat, 1.0, points)

        e_inc, b_inc = _incident_fields(k_ext, points)

        e_rec = np.stack([
            actx.to_numpy(flatten(c, actx)) for c in e_sca]).T + e_inc
        b_rec = np.stack([
            actx.to_numpy(flatten(c, actx)) for c in b_sca]).T + b_inc

        e_ref = e_sca_mie + e_inc
        b_ref = b_sca_mie + b_inc
        e_err = np.max(np.abs(e_rec - e_ref))
        b_err = np.max(np.abs(b_rec - b_ref))
        print(f"{nref:4d} {h_max:12.5e} {discr.ndofs:8d} "
              f"{result.iteration_count:7d} {e_err:12.5e} {b_err:12.5e}",
              flush=True)
        e_eoc.add_data_point(h_max, e_err)
        b_eoc.add_data_point(h_max, b_err)

    print("\nE-field EOC", flush=True)
    print(e_eoc, flush=True)
    print("\nB-field EOC", flush=True)
    print(b_eoc, flush=True)
    return e_eoc, b_eoc

# }}}


if __name__ == "__main__":
    import pyopencl as cl

    from pytential.array_context import PyOpenCLArrayContext

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    test_incident_plane_wave()

    actx = PyOpenCLArrayContext(queue)
    for eps_in in [2.25, 2.0 + 0.5j, -18.0 + 0.5j]:
        run_dirac_maxwell_sphere_mie(
            actx, eps_in, nrefs=(0, 1, 2, 3), qbx_order=4)
