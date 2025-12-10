"""
Stokes flow on the unit square using mixed finite elements.

- Velocity space: continuous vector P2 (quadratic) elements
- Pressure space: continuous P1 (linear) elements
- Discretisation: symmetric gradient form with a saddle-point matrix
- Boundary conditions: Dirichlet velocity on the boundary, pressure fixed at one node

The script can be imported as a module (for programmatic use) or run
as a standalone program to solve the problem on a given mesh resolution.
"""

from math import pi, sin, cos
from argparse import ArgumentParser

import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, bmat
from scipy.sparse.linalg import splu

from fe_utils import UnitSquareMesh, FunctionSpace, Function, LagrangeElement
from fe_utils.finite_elements import VectorFiniteElement
from fe_utils.quadrature import gauss_quadrature
from fe_utils.utils import errornorm


def analytical_velocity(x: np.ndarray) -> np.ndarray:
    """
    Analytical velocity field u = (u_x, u_y) used to manufacture an exact solution.

    Parameters
    ----------
    x : ndarray of shape (2,)
        Spatial point (x, y) in the unit square.

    Returns
    -------
    ndarray of shape (2,)
        Velocity vector at x.
    """
    u_x = 2.0 * pi * sin(2.0 * pi * x[1]) * (1.0 - cos(2.0 * pi * x[0]))
    u_y = -2.0 * pi * sin(2.0 * pi * x[0]) * (1.0 - cos(2.0 * pi * x[1]))
    return np.array([u_x, u_y])


def analytical_pressure(x: np.ndarray) -> float:
    """
    Analytical pressure field.

    Here we take p ≡ 0, so that the pressure is constant and the
    main focus is on the velocity error.

    Parameters
    ----------
    x : ndarray
        Spatial point (x, y) in the unit square.

    Returns
    -------
    float
        Pressure value (always 0.0).
    """
    return 0.0


def forcing_term(x: np.ndarray) -> np.ndarray:
    """
    Forcing term f(x) on the right-hand side of the Stokes equations.

    The forcing is chosen such that (analytical_velocity, analytical_pressure)
    is an exact solution of the continuous PDE.

    Parameters
    ----------
    x : ndarray of shape (2,)
        Spatial point (x, y) in the unit square.

    Returns
    -------
    ndarray of shape (2,)
        Forcing vector f(x).
    """
    f_x = -4.0 * pi**3 * sin(2.0 * pi * x[1]) * (2.0 * cos(2.0 * pi * x[0]) - 1.0)
    f_y = -4.0 * pi**3 * sin(2.0 * pi * x[0]) * (1.0 - 2.0 * cos(2.0 * pi * x[1]))
    return np.array([f_x, f_y])


def boundary_velocity_nodes(V: FunctionSpace) -> list[int]:
    """
    Return the indices of velocity nodes lying on the boundary of the unit square.

    This uses a simple marker function evaluated at the degrees of freedom
    to detect points on the boundary (x = 0, 1 or y = 0, 1).

    Parameters
    ----------
    V : FunctionSpace
        Velocity function space.

    Returns
    -------
    list of int
        Indices of boundary nodes in the global velocity vector.
    """
    eps = 1e-10
    marker = Function(V)

    def on_boundary(x: np.ndarray) -> tuple[float, float]:
        if x[0] < eps or x[0] > 1.0 - eps or x[1] < eps or x[1] > 1.0 - eps:
            # Non-zero indicates a boundary node for each component
            return 1.0, 1.0
        return 0.0, 0.0

    marker.interpolate(on_boundary)
    return [i for i, val in enumerate(marker.values) if np.any(val != 0.0)]


def impose_boundary_conditions(
    A, B, Z, F, V: FunctionSpace, P: FunctionSpace
):
    """
    Apply Dirichlet boundary conditions to the velocity and fix pressure at one node.

    The Stokes system has a nullspace in the pressure (p -> p + constant).
    We remove this nullspace by fixing the pressure at a single degree of freedom.

    Parameters
    ----------
    A : lil_matrix
        Velocity-velocity block.
    B : lil_matrix
        Pressure-velocity coupling block.
    Z : lil_matrix
        Pressure-pressure block (zero for Stokes).
    F : ndarray
        Global right-hand side vector [F_u; F_p].
    V : FunctionSpace
        Velocity function space.
    P : FunctionSpace
        Pressure function space.

    Returns
    -------
    (A, B, Z, F) : tuple
        Modified blocks and right-hand side after applying boundary conditions.
    """
    n = V.node_count
    bnodes = boundary_velocity_nodes(V)

    # Dirichlet BC: u = 0 on the boundary.
    A[bnodes, :] = 0.0
    A[bnodes, bnodes] = 1.0
    F[bnodes] = 0.0

    # Fix pressure at a single node to eliminate nullspace.
    B[0, :] = 0.0
    Z[0, 0] = 1.0
    F[n] = 0.0

    return A, B, Z, F


def assemble_stokes_system(
    V: FunctionSpace, P: FunctionSpace, f: Function
):
    """
    Assemble the global finite element system for the Stokes problem.

    We assemble the 2x2 block system:
        [ A  B^T ] [u] = [F_u]
        [ B   0 ] [p]   [F_p]

    where A is the viscosity block (involving the symmetric gradient),
    B is the discrete divergence, and F contains the body force contributions.

    Parameters
    ----------
    V : FunctionSpace
        Velocity space (vector P2).
    P : FunctionSpace
        Pressure space (scalar P1).
    f : Function
        Forcing term interpolated into V.

    Returns
    -------
    A, B, Z, F : (lil_matrix, lil_matrix, lil_matrix, ndarray)
        Assembled block matrices and right-hand side vector.
    """
    n, m = V.node_count, P.node_count
    A = lil_matrix((n, n), dtype=float)
    B = lil_matrix((m, n), dtype=float)
    Z = lil_matrix((m, m), dtype=float)
    F = np.zeros(n + m, dtype=float)

    mesh = V.mesh
    quad = gauss_quadrature(V.element.cell, 2 * V.element.degree)

    # psi_grad: gradients of vector basis functions on reference cell
    psi_grad = V.element.tabulate(quad.points, grad=True)
    # psi_vals: basis function values on reference cell
    psi_vals = V.element.tabulate(quad.points, grad=False)
    # phi_p: scalar pressure basis functions on reference cell
    phi_p = P.element.tabulate(quad.points, grad=False)

    for c in range(mesh.entity_counts[-1]):
        J = mesh.jacobian(c)
        detJ = abs(np.linalg.det(J))
        J_inv = np.linalg.inv(J)

        u_nodes = V.cell_nodes[c]
        p_nodes = P.cell_nodes[c]

        # Transform gradients to physical coordinates:
        # grad_psi_phys[q, j, a, b] ~ ∂_b psi_j^a at quadrature point q
        grad_psi = np.einsum("ca,pjcb->pjab", J_inv, psi_grad)

        # Symmetric gradient (strain tensor) ε(u) = (∇u + (∇u)^T) / 2
        strain = 0.5 * (grad_psi + np.einsum("pjab->pjba", grad_psi))

        # Local A_ij = ∫_K 2 ε(ψ_i) : ε(ψ_j) dx (up to constant scaling)
        local_A = np.einsum(
            "piab,pjab,p->ij", strain, strain, quad.weights
        ) * detJ
        A[np.ix_(u_nodes, u_nodes)] += local_A

        # Divergence of vector basis functions: div ψ_j = tr(∇ψ_j)
        div_psi = np.trace(grad_psi, axis1=2, axis2=3)

        # Local B_ij = ∫_K (div ψ_j) φ_i dx
        local_B = np.einsum(
            "pj,pi,p->ji", phi_p, div_psi, quad.weights
        ) * detJ
        B[np.ix_(p_nodes, u_nodes)] += local_B

        # Local right-hand side F_u: ∫_K f · ψ_j dx
        # f.values[u_nodes] has shape (n_local, dim)
        # psi_vals has shape (n_quad, n_local, dim)
        f_local = f.values[u_nodes]
        # Contract over components and quadrature points
        local_F = np.einsum(
            "kd,qjd,qjd,q->j", f_local, psi_vals, psi_vals, quad.weights
        ) * detJ
        F[u_nodes] += local_F

    return A, B, Z, F


def solve_stokes(resolution: int, return_error: bool = False):
    """
    Solve the manufactured Stokes problem on a uniform triangulation
    of the unit square.

    Parameters
    ----------
    resolution : int
        Number of cells in each coordinate direction (mesh resolution).
    return_error : bool, optional
        If True, the returned velocity and pressure are replaced by
        their error (numerical - analytical). Default is False.

    Returns
    -------
    (u, p), error : tuple
        u : Function
            Approximate velocity solution in V.
        p : Function
            Approximate pressure solution in P.
        error : float
            L² norm of the combined (velocity, pressure) error.
    """
    # Mesh and function spaces
    mesh = UnitSquareMesh(resolution, resolution)
    V = FunctionSpace(mesh, VectorFiniteElement(LagrangeElement(mesh.cell, 2)))
    P = FunctionSpace(mesh, LagrangeElement(mesh.cell, 1))

    n, m = V.node_count, P.node_count

    # Forcing term interpolated into V
    f = Function(V)
    f.interpolate(forcing_term)

    # Assemble block system
    A, B, Z, F = assemble_stokes_system(V, P, f)

    # Apply boundary conditions
    A, B, Z, F = impose_boundary_conditions(A, B, Z, F, V, P)

    # Build global saddle-point system:
    # [ A  B^T ] [u] = [F_u]
    # [ B   0 ] [p]   [F_p]
    M = bmat([[A, B.T], [B, Z]], format="lil")
    solver = splu(csc_matrix(M))
    solution = solver.solve(F)

    # Split solution into velocity and pressure components
    u = Function(V)
    p = Function(P)
    u.values[:] = solution[:n]
    p.values[:] = solution[n:]

    # Build analytical solutions
    u_exact = Function(V)
    p_exact = Function(P)
    u_exact.interpolate(analytical_velocity)
    p_exact.interpolate(analytical_pressure)

    # Compute L² error norm (velocity + pressure combined)
    err_u = errornorm(u_exact, u)
    err_p = errornorm(p_exact, p)
    error = np.sqrt(err_u**2 + err_p**2)

    if return_error:
        u.values -= u_exact.values
        p.values -= p_exact.values

    return (u, p), error


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Stokes flow on the unit square using mixed finite elements."
    )
    parser.add_argument(
        "resolution",
        type=int,
        nargs=1,
        help="Number of cells in each direction on the mesh.",
    )
    parser.add_argument(
        "--error",
        action="store_true",
        help="Print the L² error instead of just solving.",
    )

    args = parser.parse_args()
    resolution = args.resolution[0]

    (u, p), error = solve_stokes(resolution, return_error=False)

    if args.error:
        print(f"L² norm of combined (u, p) error: {error:.6e}")

    # Optionally, you can visualise u and p using the plotting
    # utilities in the fe_utils.Function class:
    # u.plot()
    # p.plot()
