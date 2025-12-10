from .quadrature import gauss_quadrature
import numpy as np
from .finite_elements import VectorFiniteElement

def errornorm(f1, f2):
    """Calculate the L^2 norm of the difference between f1 and f2."""

    fs1 = f1.function_space
    fs2 = f2.function_space

    fe1 = fs1.element
    fe2 = fs2.element
    mesh = fs1.mesh

    # Create a quadrature rule
    Q = gauss_quadrature(fe1.cell, 16 * max(fe1.degree, fe2.degree))

    # Evaluate the local basis functions at the quadrature points
    phi = fe1.tabulate(Q.points)
    psi = fe2.tabulate(Q.points)

    norm = 0.0

    for c in range(mesh.entity_counts[-1]):
        nodes1 = fs1.cell_nodes[c, :]
        nodes2 = fs2.cell_nodes[c, :]
        J = mesh.jacobian(c)
        detJ = abs(np.linalg.det(J))

        # Handle vector-valued finite elements (e.g., for VectorFiniteElement)
        if phi.ndim == 3:
            diff = np.sum(np.dot(f1.values[nodes1], phi.T) - np.dot(f2.values[nodes2], psi.T), axis=0)
            norm += np.dot(diff**2, Q.weights) * detJ
        else:
            norm += np.dot(
                (np.dot(f1.values[nodes1], phi.T) - np.dot(f2.values[nodes2], psi.T))**2,
                Q.weights,
            ) * detJ

    return norm**0.5