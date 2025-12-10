import numpy as np
import sympy as sp
from sympy import symbols, lambdify
from scipy.special import comb
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.
    """
    points = []

    def generate_barycentric_tuples(current, degree, dim, result):
        """Recursive function to help to generate integer partitions,
         so  that sum to `degree`
        across `dim + 1` barycentric coordinates (all positive)."""
        if len(current) == dim:
            if sum(current) < degree:
                current.append(degree - sum(current))
                result.append(current.copy())
                current.pop()
            return
        for j in range(1, degree - sum(current)):
            current.append(j)
            generate_barycentric_tuples(current, degree, dim, result)
            current.pop()

    # Loop over all entity dimensions (vertices, edges, faces)
    for d in range(cell.dim + 1):
        result = []
        generate_barycentric_tuples([], degree, d, result)
        bary_coords = np.array(result) / degree

        for entity_index, vertices in cell.topology[d].items():
            vertex_coords = np.array([cell.vertices[i] for i in vertices])
            entity_points = [np.dot(b, vertex_coords) for b in bary_coords]

            if d == 1:
                # Ensure edge nodes are ordered from vertex 0 to 1
                v0, v1 = vertex_coords
                direction = v1 - v0
                projections = [np.dot(p - v0, direction)
                               for p in entity_points]
                entity_points = [pt for _, pt in
                                 sorted(zip(projections, entity_points))]

            points.extend(entity_points)

    return np.array(points)


def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`
    """
    dim = cell.dim
    x = symbols(f"x0:{dim}")  # Create symbolic variables
    monomial_basis = []

    # Recursively generate monomials in lexicographic order
    def _create_monomials(total_degree, dimension, term):
        if total_degree == 0:
            monomial_basis.append(term)
            return
        if dimension == dim:
            return
        for j in reversed(range(total_degree + 1)):
            _create_monomials(
                total_degree - j, dimension + 1, term * x[dimension] ** j
                )

    for i in range(degree + 1):
        _create_monomials(i, 0, sp.Integer(1))

    points = np.asarray(points)

    if grad:
        # Build gradient basis functions: shape (num_basis, dim)
        gradient_basis = [
            [lambdify(x, m.diff(x[i]), modules='numpy') for i in range(dim)]
            for m in monomial_basis
        ]

        # Evaluate: shape (num_points, num_basis, dim)
        V = np.empty((len(points), len(monomial_basis), dim), dtype=np.float64)
        for pt_index, pt in enumerate(points):
            for m_index, partials in enumerate(gradient_basis):
                V[pt_index, m_index, :] = [f(*pt) for f in partials]

        return V

    else:
        # Evaluate monomial basis at all points: shape (num_points, num_basis)
        eval_basis = [lambdify(x, m, modules='numpy') for m in monomial_basis]
        V = np.empty((len(points), len(eval_basis)), dtype=np.float64)
        for i, f in enumerate(eval_basis):
            V[:, i] = [f(*pt) for pt in points]

        return V


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            point evaluation node locations on the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with
            entity `(d, i)` of dimension `d` and index `i`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array(
                [len(entity_nodes[d][0]) for d in range(cell.dim + 1)]
            )
        # self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.
        V = vandermonde_matrix(cell, degree, nodes)

        if isinstance(self, VectorFiniteElement):
            self.basis_coefs = None
        else:
            self.basis_coefs = np.linalg.inv(V)

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``false`` and (points, nodes, dim) if ``grad``
            is ``True``.
        """

        V = vandermonde_matrix(self.cell, self.degree, points, grad=grad)
        return np.einsum(
            'ib...,bj->ij...', V, self.basis_coefs, optimize=True
            )

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """
        nodes = np.asarray(self.nodes)
        return np.array([fn(node) for node in nodes])

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """

        # Construct the list of node coordinates
        nodes = lagrange_points(cell, degree)

        # Initialize the entity_nodes dictionary
        entity_nodes = {d: {} for d in range(cell.dim + 1)}

        node_index = 0  # Global node counter

        # Assign nodes to each topological entity
        for d in range(cell.dim + 1):
            # Number of nodes per entity of this dimension
            num_nodes_per_entity = int(comb(degree - 1, d))

            for entity in cell.topology[d]:
                entity_nodes[d][entity] = list(
                    range(node_index, node_index + num_nodes_per_entity)
                )
                node_index += num_nodes_per_entity

        FiniteElement.__init__(self, cell, degree, nodes, entity_nodes)


class VectorFiniteElement(FiniteElement):
    def __init__(self, scalar_element):
        """A vector finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param element: the scalar finite element defining the basis
            functions of the vector element.

        The implementation of this class is left as an :ref:`exercise
        <ex-vector-element>`.
        """
        self.scalar_element = scalar_element
        self.cell = scalar_element.cell
        self.degree = scalar_element.degree
        self.dim = self.cell.dim

        # Repeat each scalar node `dim` times
        self.nodes = np.repeat(scalar_element.nodes, self.dim, axis=0)

        # Build entity_nodes structure for the vector element
        self.entity_nodes = {
            d: {
                i: [self.dim * n + j for n in nodes for j in range(self.dim)]
                for i, nodes in scalar_element.entity_nodes[d].items()
            }
            for d in scalar_element.entity_nodes
        }

        # Number of nodes per entity is scaled by the dimension
        self.nodes_per_entity = scalar_element.nodes_per_entity * self.dim

        # Basis function direction indicators (weights)
        self.node_weights = np.array([
            np.eye(self.dim)[i % self.dim]
            for i in range(len(self.nodes))
        ])

        FiniteElement.__init__(
            self, self.cell, self.degree, self.nodes, self.entity_nodes
            )

    def tabulate(self, points, grad=False):
        """Tabulate the basis functions of this vector finite element
        at given points.
        :param points: A list of coordinate tuples.
        :param grad: Whether to return the gradient tabulation.
        :return: The tabulated basis functions with appropriate rank.
        """
        scalar_tabulation = self.scalar_element.tabulate(points, grad=grad)
        num_points, num_nodes = scalar_tabulation.shape[:2]

        if grad:
            # Rank 4: (num_points, num_nodes * self.dim, self.dim, self.dim)
            tabulated = np.zeros(
                (num_points, num_nodes * self.dim, self.dim, self.dim)
                )
            for i in range(num_nodes * self.dim):
                scalar_index = i // self.dim
                vector_index = i % self.dim
                tabulated[:, i, :, vector_index] = (
                    scalar_tabulation[:, scalar_index, :]
                )
        else:
            # Rank 3: (num_points, num_nodes * self.dim, dim)
            tabulated = np.zeros((num_points, num_nodes * self.dim, self.dim))
            for i in range(num_nodes * self.dim):
                scalar_index = i // self.dim
                vector_index = i % self.dim
                tabulated[:, i, vector_index] = (
                    scalar_tabulation[:, scalar_index]
                )

        return tabulated
