import numpy as np
import mesh_manager as mema
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import NullFormatter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import random
import copy

def count_dof (mesh):
    """
    Calculate total number of degrees of freedom of the solver
    (size of the linear system)

    Args:
        mesh (ddr.mesh2D): mesh

    Returns:
        list (int): number of dofs for pressure, for velocity and linear system size

    Raises:
        ErrorType: Description of when this error might be raised.
    """
    no_elems = len(mesh.elem2node)
    no_edges = mesh.no_edges

    return [no_elems, 2*(no_elems +no_edges), no_elems*3 + no_edges*2]


def loc_dof_description (mesh, iel, loc_dof):
    """
    Returns description of the local dof (p/V, x/y, T/F, face_number)

    Args:
        mesh (ddr.mesh2D): mesh
        iel              : index of elem
        loc_dof     (int): local degree of freedom

    Returns:
        list of str: ["p"/"V", "-"/x"/"y", "T"/"F", face_number]

    Raises:
        ...
    """
    if (loc_dof==0):
        return ["p", "-", "T", -1]
    elif (loc_dof==1):
        return ["V", "x", "T", -1]
    elif (loc_dof==2):
        return ["V", "y", "T", -1]
    elif ((loc_dof-3)%2==0):
        return ["V", "x", "F",  int((loc_dof-3)/2)]
    else:
        return ["V", "y", "F",  int((loc_dof-3)/2)]



def dof_loc2glob (mesh, iel, loc_dof):
    """
    Maps local dof index to global dof index

    Args:
        mesh (ddr.mesh2D): mesh
        iel       (int): element index
        loc_dof   (int): local degree of freedom

    Returns:
        int: global dof index

    Raises:
        ErrorType: Description of when this error might be raised.
    """
    no_elems = len(mesh.elem2node)
    no_edges = mesh.no_edges
    [phys_type, coord, mesh_type, idx_edge] = loc_dof_description(mesh, iel, loc_dof)
    if (phys_type=="p"):
        return iel
    else:
        if (mesh_type=="T"):
            if (coord=="x"):
                return no_elems + 2*iel
            else:
                return no_elems + 2*iel + 1
        else:
            # retrieve global index of the edge
            edge_glob_idx = mesh.elem2edge[iel][idx_edge]
            if (coord=="x"):
                return 3*no_elems + 2*edge_glob_idx
            else:
                return 3*no_elems + 2*edge_glob_idx + 1

def get_edge_dofs(mesh, iel, ie):
    """
    Description: returns 2 velocity edge dofs given element index and local edge index

    Args:
        mesh (ddr.mesh2D): mesh
        iel         (int): element index
        ie          (int): local index of edge

    Returns:
        list        (int): dof_x, dof_y
    """
    dof_x = dof_loc2glob(mesh, iel, 3+ 2*ie)
    dof_y = dof_loc2glob(mesh, iel, 3+ 2*ie + 1)

    return [dof_x, dof_y]


def get_local_polynomials(mesh, iel, dof):
    """
    Provides local p0 polynomials associated to a velocity dof,
    with the convention [v_T, {v_F}_F].
    v_T is p0(T)^2, v_F is p0(F)^2 for each F

    Args:
        mesh         (ddr.Mesh): mesh
        iel               (int): element index
        dof               (int): local degree of freedom

    Returns:
        list (np.array(2)) len=(1+num_edge)): local p0 polynomials (R^2 constants)

    Raises:
           ...
    """
    # get description of dof
    [phys_type, coord, mesh_type, idx_edge] = loc_dof_description(mesh, iel, dof)
    # reject if pressure dof
    if (phys_type=="p"):
        raise ValueError("Operation not available for pressure dof")
    # get number of edges and instantiate list with 0 everywhere
    no_edges = len(mesh.elem2edge[iel])
    polynomials = [np.array([.0, .0])   for _ in range(1+no_edges)] # syntax to have different independent arrays
    # correct list with 1 where necessary
    if (mesh_type=="T"):
        list_pos = 0
    else:
        list_pos = 1 + idx_edge
    if (coord == "x"):
        array_pos = 0
    else:
        array_pos = 1
    polynomials[list_pos][array_pos] = 1.0

    return polynomials

def get_local_velocities (mesh, iel, v_h):
    """
    Provides local velocities associated to element
    w.r.t. a global velocity unknown v_h
    with the convention [v_T, {v_E}_E].
    v_T is p0(T)^2, v_E is p0(E)^2 for each E

    Args:
        mesh         (ddr.Mesh): mesh
        iel               (int): element index
        v_h               (np_array): local degree of freedom

    Returns:
        list (np.array(2)) len=(1+num_edge)): local velocities [v_T, {v_E}]
    """
    elem_velocities = []
    # get and append v_T
    v_T_x = v_h[2*iel]
    v_T_y = v_h[2*iel + 1]
    elem_velocities.append(np.array([v_T_x, v_T_y]))
    # get and append v_E for each E
    elem = mesh.elem2node[iel]
    no_elem_dofs = len(mesh.elem2node)*2
    for ie in range(len(elem)):
        glob_ie = mesh.elem2edge[iel][ie]
        v_E_x = v_h[no_elem_dofs + 2*glob_ie]
        v_E_y = v_h[no_elem_dofs + 2*glob_ie + 1]
        elem_velocities.append(np.array([v_E_x, v_E_y]))
    return elem_velocities


def get_local_interpolation (mesh, iel, ref_v):
    """
    Provides local p0 polynomials associated to a velocity dof,
    with the convention [v_T, {v_F}_F].
    v_T is p0(T)^2, v_F is p0(F)^2 for each F

    Args:
        mesh         (ddr.Mesh): mesh
    iel               (int): element index
    ref_v          (lambda): H1 velocity to interpolate

    Returns:
        np.array  (2*(1+num_edge)): local p0 polynomials (R^2 constants) (underline{v}_T)

    Raises:
           ...
    """
    # initialize interpolation array
    no_edges = len(mesh.elem2edge[iel])
    interpolation  = np.zeros((2, 1+no_edges))

    # assign interpolation onto element
    xT = mesh.element_barycenter[iel]
    vT = ref_v(xT[0], xT[1])
    interpolation [0, 0] = vT[0]
    interpolation [1, 0] = vT[1]

    # assign interpolation onto edge
    for ie in range(no_edges):
        glob_ie = mesh.elem2edge[iel][ie]
        xE = mesh.edge_xE[glob_ie]
        vE = ref_v(xE[0], xE[1])
        interpolation [0, 1+ ie] = vE[0]
        interpolation [1, 1+ ie] = vE[1]

    return interpolation


def get_local_projections(mesh, iel, ref_v):
    """
    Given a reference velocity, computes the
    projection over the local basis functions
    and packs them into an array (usual convention for ordering).
    Combining get_local_polynomials and get_local_projections
    should be the same as get_local_interpolation

    Args:
        mesh         (ddr.Mesh): mesh
        iel               (int): element index
        dof               (int): local degree of freedom

    Returns:
        np.array  (no_loc_dofs): local projections

    Raises:
           ...
    """
    interpolation = get_local_interpolation (mesh, iel, ref_v)
    return interpolation.flatten(order='F')


def get_v_rec_from_localV(mesh, iel, localV):
    """


    Args:
        mesh                   (Mesh2D): mesh
        iel                         (int): element index
        localV   list(np.array) : local function (v_T, {v_E}_E)

    Returns:
        np.array       (2*2): A (grad r^{k+1}_T)
        np.array       (2*1): b (1/mod_T*int_T r^{k+1}_T : element average)

    Raises:
        ...
    """

    # average coincides with vT
    b = localV[0]

    # initialize grad{ r^{k+1}_T}
    A = np.zeros((2,2))

    # loop over edges
    no_edges = len(mesh.elem2edge[iel])
    mod_T = mesh.element_surface [iel]   # NB: mod_T "=" h_T**2

    for ie in range(no_edges):

        nTE = mesh.edge_normal[iel][ie]
        vE = localV[1+ie]

        outer_product = np.array([[vE[0]*nTE[0], vE[0]*nTE[1]],[vE[1]*nTE[0], vE[1]*nTE[1]]])

        A = A + 1/mod_T*outer_product

    return [A, b]



def get_v_rec(mesh, iel, dof): # version that uses get_v_rec_from_localV
    """
    Provides coefficients of r^{k+1}_T for the local basis function
    associated to the local degree of freedom w.r.t. the canonical basis.
    i.e. returns [A, b] if r^{k+1}_T = A*(x-xT)+ b (x, xT, b are in R^2)
    In particular, A is the gradient to use in aT, and b is the average

    Args:
        mesh      (ddr.Mesh): mesh
        iel            (int): element index
        dof            (int): local degree of freedom

    Returns:
        np.array       (2*2): A (grad r^{k+1}_T)
        np.array       (2*1): b

    Raises:
        ...
    """
    # get \underline{phi}_T
    polynomials = get_local_polynomials (mesh, iel, dof)

    return get_v_rec_from_localV(mesh, iel, polynomials)



def get_disc_div_from_localV(mesh, iel, localV):  # MAYBE CHANGE TO get_dof_disc_div
    """
    Discrete divergence

    Args:
        mesh (ddr.Mesh): mesh
        iel       (int): element index
        localV   (np.array, 2*1+no_edges): local function (v_T, {v_E}_E)

    Returns:
        float          : dicrete divergence (p0(T): constant)

    Raises:
        ...
    """

    # initialize D^k_T
    DkT = 0
    # loop over edges
    no_edges = len(mesh.elem2edge[iel])
    mod_T = mesh.element_surface [iel]   # NB: mod_T "=" h_T**2

    for ie in range(no_edges):

        nTE = mesh.edge_normal[iel][ie]
        vE = localV[ 1+ie]

        DkT = DkT + 1/mod_T*np.dot(vE, nTE)

    return DkT

def get_disc_div (mesh, iel, dof): # version that uses get_disc_div_from_localV
    """
    Provides coefficients of r^{k+1}_T for the local basis function
    associated to the local degree of freedom w.r.t. the canonical basis.
    i.e. returns [A, b] if r^{k+1}_T = A*(x-xT)+ b (x, xT, b are in R^2)
    In particular, A is the gradient to use in aT, and b is the average

    Args:
        mesh      (ddr.Mesh): mesh
        iel            (int): element index
        dof            (int): local degree of freedom

    Returns:
        np.array       (2*2): A (grad r^{k+1}_T)
        np.array       (2*1): b

    Raises:
        ...
    """
    # get \underline{phi}_T
    polynomials = get_local_polynomials (mesh, iel, dof)

    return get_disc_div_from_localV(mesh, iel, polynomials)


def local_contribution_aT(mesh, v_rec, iel, i, j, nu):
    """
    Calculate local contribution aT (grad:grad)

    Args:
        mesh (Mesh2D): mesh
        v_rec (list): list of velocity reconstrunction format [A,b]
        iel  (int): element index
        i    (int): local dof 1
        j    (int): local dof 2
        nu (float): element viscosity

    Returns:
        real: local contribution

    Raises:
        ...
    """
    # get coefficients of velocity reconstruction
    [A_i, b_i] = v_rec[iel][i - 1]
    [A_j, b_j] = v_rec[iel][j - 1]
    # take out gradient
    grad_rec_i = A_i
    grad_rec_j = A_j
    # take only symmetric part
    sym_grad_rec_i = 0.5*(A_i+ np.transpose(A_i))
    sym_grad_rec_j = 0.5*(A_j+ np.transpose(A_j))
    # get element surface
    mod_T = mesh.element_surface [iel]
    return nu*mod_T*np.sum(2 * sym_grad_rec_i * sym_grad_rec_j) # D(u):D(v)
    #return mod_T*np.sum( grad_rec_i * grad_rec_j) # D(u):D(v)



def local_contribution_sT(mesh, v_rec, iel, i, j):
    """
    Calculate local contribution sT (stabilization)

    Args:
        mesh (ddr.Mesh): mesh
        v_rec (list): list of velocity reconstrunction format [A,b]
        iel  (int): element index
        i    (int): local dof 1
        j    (int): local dof 2

    Returns:
        real: local contribution

    Raises:
        ...
    """
    contribution = 0
    # get velocity reconstruction (proj^0 is just b)
    [A_i, b_i] = v_rec[iel][i - 1]
    [A_j, b_j] = v_rec[iel][j - 1]
    # get projection (value at xT (that is b)) of r^{k+1}_T
    proj_i_T = b_i
    proj_j_T = b_j
    # get element p0 polynomials associated to dof
    v_i_T = get_local_polynomials (mesh, iel, i)[0]
    v_j_T = get_local_polynomials (mesh, iel, j)[0]
    # add element contribution
    mod_T = mesh.element_surface [iel]
    contribution = contribution + np.dot(proj_i_T-v_i_T, proj_j_T-v_j_T)/mod_T # mod_T "=" h_T^2
    #print ("Element: ", proj_i_T, v_i_T)
    # add face contributions
    edge_per_elem = len(mesh.elem2edge[iel])
    for ie in range(edge_per_elem):
        # get geometry of the element
        xT = mesh.element_barycenter[iel]
        glob_ie = mesh.elem2edge[iel][ie]
        xE = mesh.edge_xE[glob_ie]
        mod_E = mesh.edge_length[glob_ie]
        # get projection (value at xE) of r^{k+1}_T
        proj_i_E = b_i + np.dot(A_i, xE-xT)
        proj_j_E = b_j + np.dot(A_j, xE-xT)
        # get face polynomials
        v_i_E =  get_local_polynomials (mesh, iel, i)[1+ie] # skip first, it is v_T
        v_j_E =  get_local_polynomials (mesh, iel, j)[1+ie]
        contribution = contribution + np.dot(proj_i_E -v_i_E, proj_j_E -v_j_E)/mod_E
        #print ("Face ", ie, ": ", proj_i_E, v_i_E)

    return contribution


def local_contribution_bT(mesh, iel, i):
    """
    Calculate local contribution bT (coupling v,p)

    Args:
        mesh (ddr.Mesh): mesh
        iel  (int): element index
        i    (int): local dof associated to v

    Returns:
        real: local contribution
    """
    # get discrete divergence
    disc_div_i = get_disc_div(mesh, iel, i)
    mod_T = mesh.element_surface [iel]

    return -mod_T*disc_div_i


def local_contribution_fT(mesh, iel, f, dof):
    """
    Calculate local contribution fT (forcing term)

    Args:
        mesh        (ddr.Mesh): mesh
        iel              (int): element index
        f    (lambda function): forcing term
        dof              (int): local dof

    Returns:
        real: local contribution

    """
    # evaluate f in element centre (or, later, maybe use an average)
    x_T = mesh.element_barycenter[iel]
    f_T = f(x_T[0], x_T[1])
    v_T = get_local_polynomials(mesh, iel, dof) [0] # recall convention: first is element value (p0(T)^2)
    mod_T = mesh.element_surface [iel]

    return mod_T*np.dot(f_T, v_T)


def assemble_A (mesh, v_rec, nu_in, nu_ex):
    """
    Assembles matrix A (constant viscosity)

    Args:
        mesh      (Mesh2D): mesh
        v_rec       (list): velocity reconstrucntion (list with elems [A,b])
        nu_in      (float): viscosity internal
        nu_ext     (float): viscosity external

    Returns:
        np.array(no_v_dofs, no_v_dofs): matrix A

    Raises:
        ErrorType: Description of when this error might be raised.
    """
    no_p_dofs = count_dof(mesh)[0]
    no_v_dofs = count_dof(mesh)[1]
    A = np.zeros([no_v_dofs, no_v_dofs])

    for iel in range(len(mesh.elem2node)):

        # determine element viscosity
        if (mesh.side_mask[iel]==0):
            nu = nu_in
        else:
            nu = nu_ex

        edge_per_elem = len(mesh.elem2edge[iel])
        no_loc_dofs = 1 + 2*(1 + edge_per_elem)

        for i in range(1, no_loc_dofs):   # skip pressure dof
            for j in range(1, no_loc_dofs):
                glob_i = dof_loc2glob(mesh, iel, i)
                glob_j = dof_loc2glob(mesh, iel, j)
                shifted_i = glob_i - no_p_dofs # offset in matrix which has dim: no_v_dofs*no_v_dofs
                shifted_j = glob_j - no_p_dofs # offset in matrix which has dim: no_v_dofs*no_v_dofs

                A[shifted_i, shifted_j] = A[shifted_i, shifted_j] +\
                                    local_contribution_aT (mesh, v_rec, iel, i, j, nu) +\
                                    local_contribution_sT (mesh, v_rec, iel, i, j)      # stab

    return A

def assemble_STAB (mesh):
    """
    Assembles only stabilisation component of matrix A (debugging purpose)

    Args:
        mesh (ddr.Mesh): mesh

    Returns:
        np.array(no_v_dofs, no_v_dofs): matrix STAB

    """
    no_p_dofs = count_dof(mesh)[0]
    no_v_dofs = count_dof(mesh)[1]
    STAB = np.zeros([no_v_dofs, no_v_dofs])

    for iel in range(len(mesh.elem2node)):
        edge_per_elem = len(mesh.elem2edge[iel])
        no_loc_dofs = 1 + 2*(1 + edge_per_elem)

        for i in range(1, no_loc_dofs):   # skip pressure dof
            for j in range(1, no_loc_dofs):
                glob_i = dof_loc2glob(mesh, iel, i)
                glob_j = dof_loc2glob(mesh, iel, j)
                shifted_i = glob_i - no_p_dofs # offset in matrix which has dim: no_v_dofs*no_v_dofs
                shifted_j = glob_j - no_p_dofs # offset in matrix which has dim: no_v_dofs*no_v_dofs

                STAB[shifted_i, shifted_j] = STAB[shifted_i, shifted_j] +\
                                             local_contribution_sT (mesh, iel, i, j)

    return STAB


def assemble_B (mesh):
    """
    Assembles matrix B

    Args:
        mesh (ddr.Mesh): mesh

    Returns:
        np.array(no_v_dofs, no_p_dofs): matrix B
    """
    no_p_dofs, no_v_dofs = count_dof(mesh)[0:2]
    B = np.zeros([no_v_dofs, no_p_dofs])

    for iel in range(len(mesh.elem2node)):
        edge_per_elem = len(mesh.elem2edge[iel])
        no_loc_dofs = 1 + 2*(1 + edge_per_elem)
        glob_j = dof_loc2glob (mesh, iel, 0) # global dof of pressure (no shift needed)

        for i in range(1, no_loc_dofs):
                glob_i = dof_loc2glob(mesh, iel, i)
                shifted_i = glob_i - no_p_dofs

                B[shifted_i, glob_j] = B[shifted_i, glob_j] +\
                                    local_contribution_bT (mesh, iel, i)
    return B

def assemble_JP (mesh, v_rec, verbose = False):
    """
    Assembles the Jump Penalization term,
    necessary to recover stability at lowest order (k=0)
    See di2020hybrid, sec. 7.6

    Args:
       mesh (mesh2D): mesh
       v_rec (list): list of velocity reconstruction format [A,b]
       verbose (bool): verbosity flag

    Returns:
       np.array: JP (jump penalization matrix)
    """
#    def quadrature_order_3 (f, x0, x1):
#        """
#        Cavalieri-Simpson quadrature formula
#        Exact for polynomials with degree <= 3
#
#        Args:
#            f (lambda): function to integrate
#            x0 (np.array): first extreme
#            x1 (np.array): second extreme
#        Returns:
#            float (integral)
#        """
#        points = [x0, 0.5*(x0+x1), x1]
#        values = [f(p[0], p[1]) for p in points]
#        h = mema.R2_norm(x1-x0) # interval length
#        weights = [h/6, 4*h/6, h/6]
#        return sum ([value*weight for [value, weight] in zip(values, weights)])

    no_p_dofs, no_v_dofs = count_dof(mesh)[0:2]
    JP = np.zeros([no_v_dofs, no_v_dofs])

    # get edge to element connectivity
    if mesh.edge2elem == []:
        mesh.edge2elem = mesh.generate_edge2elem()
    edge2elem = mesh.edge2elem

    # loop over edges
    for ie in range(len(edge2elem)):

        if (verbose):
            print ("> adding contributions for edge: ", ie)
            print ("")

        # get extrema of edge and lenght
        first_iel = edge2elem[ie][0] # at least one elem is associated to edge
        first_elem = mesh.elem2node[first_iel]
        first_elem2edge = mesh.elem2edge[first_iel] # list of global idxs of edges of first_iel
        local_ie = [k for (k, iedge) in enumerate(first_elem2edge) if iedge==ie][0] # get local idx of edge wrt first_ie,
                                                                              # which is also local idx of its first vertex
        xE0 = mesh.coords[first_elem[local_ie]]
        xE1 = mesh.coords[first_elem[(local_ie+1)%len(first_elem)]]
        hE = mema.R2_norm (xE1 - xE0)

        # get table of DOFs connected to the edge
        # table is a list of lists of the type:
        # [element, local_dof, is_shared, other_side],
        # with other_side beign a list [opposite_iel, local_dof_on_opposite_iel]
        # (necessary to reconstruct on other side)
        # is_shared is stored because for shared dofs the reconstruction comes from both sides
        # we treat in the same way el/el interface edges and boundary edges
        dof_table = []

        # loop over associated elements to assemble dof_table
        associated_elems = edge2elem[ie]
        for side, iel in enumerate(associated_elems):
            elem = mesh.elem2node[iel]
            # add element dof (2 components)
            for component in range(2):
                dof_table.append ([iel, 1 + component, False, []]) # beware the dof for pressure
            # loop on sides to add edge dofs
            for local_ie in range(len(elem)):
                # check if side is shared (in case of boundary, simply it is not)
                edge = mesh.elem2edge[iel][local_ie]
                opposite_side = [k for k in associated_elems if k!= iel]
                if (opposite_side !=[]):
                    # edge is elem/elem interface
                    opposite_iel = opposite_side[0]
                    check_local_ie_opposite = [k for (k, opposite_edge) in enumerate (mesh.elem2edge[opposite_iel]) if opposite_edge==edge]
                    if (check_local_ie_opposite!=[]):
                        is_shared = True
                        local_ie_opposite = check_local_ie_opposite[0]
                    else:
                        is_shared = False
                else:
                    # edge on boundary
                    is_shared = False

               # loop on components (2 for each edge)
                for component in range(2):
                    local_dof = 1+ 2+ 2*local_ie + component # beware of pressure and elem velocity
                    if is_shared:
                        local_dof_opposite = 1 + 2 + 2*local_ie_opposite + component
                        other_side = [opposite_iel, local_dof_opposite]
                    else:
                        other_side = []

                    dof_table.append([iel, local_dof, is_shared, other_side])

        if (verbose):
            print (">>dof_table: ")
            for dof_descriptor in dof_table:
                print (dof_descriptor)


        # loop twice over dofs in dof_table (i, j)
        for dof_descriptor_i in dof_table:

          # get local dofs and elements
          iel_i, loc_dof_i = dof_descriptor_i[0:2]
          # get if shared
          is_shared_i = dof_descriptor_i[2]
          # get global dofs
          glob_dof_i = dof_loc2glob (mesh, iel_i, loc_dof_i)

          ## side 0 (side of the considered dof)
          [A_i_0, b_i_0] = v_rec[iel_i][loc_dof_i - 1]
          xT_i_0 = mesh.element_barycenter[iel_i]

          #pot_i_el_0 = lambda x,y: np.dot(A_i_0, np.array([x,y]) - xT_i_0 ) + b_i_0

          # get potential reconstructions from 2 sides (as lambda functions)

          ## side 1 (side opposite to the dof)
          # if dof is not shared, the reconstruction on the other side is just 0,
          # otherwise it has to be calculated taking the local dof on the opposite side
          if (is_shared_i):
              # dof is shared
              opposite_iel_i, opposite_loc_dof_i = dof_descriptor_i[3]
              opposite_glob_dof_i = dof_loc2glob (mesh, opposite_iel_i, opposite_loc_dof_i)
              [A_i_1, b_i_1] = v_rec[opposite_iel_i][opposite_loc_dof_i - 1]
              xT_i_1 = mesh.element_barycenter[opposite_iel_i]

              #pot_i_el_1 = lambda x,y: np.dot(A_i_1, np.array([x,y]) - xT_i_1 ) + b_i_1
          else:
              #pot_i_el_1 = lambda x, y: 0.0*x*y
              [A_i_1, b_i_1] = [np.array([[0., 0.], [0., 0.]]), np.array([0., 0.])]
              xT_i_1 = np.array([0., 0.]) # any value, anyway multiplied by 0

          # by convention jump is value on first associated element minus value on second
          # associated element and jump is null if edge is on boundary

          #jump_pot_i = lambda x, y: (-pot_i_el_1(x, y) +pot_i_el_0(x, y))*(2*(iel_i==associated_elems[0]) - 1)*(len(associated_elems)==2)

          for dof_descriptor_j in dof_table:

                # get local dofs and elements
                iel_j, loc_dof_j = dof_descriptor_j[0:2]
                # get if shared
                is_shared_j = dof_descriptor_j[2]
                # get global dofs
                glob_dof_j = dof_loc2glob (mesh, iel_j, loc_dof_j)

                # get potential reconstructions from 2 sides (as lambda functions)

                ## side 0
                [A_j_0, b_j_0] = v_rec[iel_j][loc_dof_j - 1]
                xT_j_0 = mesh.element_barycenter[iel_j]

                #pot_j_el_0 = lambda x,y: np.dot(A_j_0, np.array([x,y]) - xT_j_0 ) + b_j_0

                ## side 1
                if (is_shared_j):
                    # dof is shared
                    opposite_iel_j, opposite_loc_dof_j = dof_descriptor_j[3]
                    opposite_glob_dof_j = dof_loc2glob (mesh, opposite_iel_j, opposite_loc_dof_j)
                    [A_j_1, b_j_1] = v_rec[opposite_iel_j][opposite_loc_dof_j - 1]
                    xT_j_1 = mesh.element_barycenter[opposite_iel_j]
                    #pot_j_el_1 = lambda x,y: np.dot(A_j_1, np.array([x,y]) - xT_j_1 ) + b_j_1

                else:
                    #pot_j_el_1 = lambda x, y: 0.0*x*y
                    [A_j_1, b_j_1] = [np.array([[0., 0.], [0., 0.]]), np.array([0., 0.])]
                    xT_j_1 = np.array([0., 0.]) # any value, anyway multiplied by 0

                # by convention jump is value on first associated element minus value on second
                # associated element

                #jump_pot_j = lambda x, y: (-pot_j_el_1(x, y) +pot_j_el_0(x, y))*(2*(iel_j==associated_elems[0]) - 1)*(len(associated_elems)==2)

                # Avoid Lambda Functions and get reconstruction via their representation [A, b]
                # also, embed quadrature without calling an internal function

                if (len(associated_elems)==2):
                    sign_i = 2*(iel_i==associated_elems[0]) - 1
                    sign_j = 2*(iel_j==associated_elems[0]) - 1

                    integral_prod_jumps = 0.0
                    quad_points = [xE0, 0.5*(xE0 +xE1), xE1]
                    quad_weights = [hE/6, 4*hE/6, hE/6]
                    for i_quad in range(3):
                        x_quad = quad_points[i_quad]
                        pot_i_0 = np.dot(A_i_0, x_quad - xT_i_0) + b_i_0
                        pot_i_1 = np.dot(A_i_1, x_quad - xT_i_1) + b_i_1
                        pot_j_0 = np.dot(A_j_0, x_quad - xT_j_0) + b_j_0
                        pot_j_1 = np.dot(A_j_1, x_quad - xT_j_1) + b_j_1
                        pot_jump_i = (-pot_i_1 + pot_i_0)*sign_i
                        pot_jump_j = (-pot_j_1 + pot_j_0)*sign_j
                        integral_prod_jumps += quad_weights[i_quad]*np.dot(pot_jump_i, pot_jump_j)
                else:
                    integral_prod_jumps = 0.0

                # get product of jumps over edge (remark jump should b 0 for DOF on edge ie)

                #product_of_jumps = lambda x, y: np.dot(jump_pot_i(x, y), jump_pot_j(x, y))

                # add contribution
                # contribution is halved if a dof is shared
                # for example: the same contribution is calculated 2*2 = 4 times if dof_i and dof_j
                # are both at the interface between the 2 elements

                # need to shift the index because matrix has size no_v_dof
                #JP [glob_dof_i - no_p_dofs , glob_dof_j - no_p_dofs ] += quadrature_order_3(product_of_jumps, xE0, xE1)/hE\
                JP [glob_dof_i - no_p_dofs , glob_dof_j - no_p_dofs ] += integral_prod_jumps/hE\
                                               *(1-0.5*(is_shared_i))\
                                               *(1-0.5*(is_shared_j))


    return JP


def assemble_b_f (mesh, f):
    """
    Assembles vector b_f associated to <f, v>

    Args:
        mesh (ddr.Mesh): mesh
        f (lambda function): forcing term

    Returns:
        np.array(no_v_dofs): vector b
    """
    no_p_dofs, no_v_dofs = count_dof(mesh)[0:2]
    b = np.zeros(no_v_dofs)

    for iel in range(len(mesh.elem2node)):
        edge_per_elem = len(mesh.elem2edge[iel])
        no_loc_dofs = 1 +2*(1 + edge_per_elem) # number of velocity local dofs

        for i in range(1, no_loc_dofs):
                glob_i = dof_loc2glob(mesh, iel, i)
                shifted_i = glob_i - no_p_dofs #offset in array which has dim: no_v_dofs

                b[shifted_i] = b[shifted_i] +\
                                    local_contribution_fT (mesh, iel, f, i)
    return b

def assemble_b_gamma(mesh, t_gamma):
    """
    Assembles operator associated to surface tension
    $tau_\Gamma$

    Args:
        mesh           (mema.Mesh): mesh
        tau_gamma  (list(np.array): edge-wise surface tension

    Returns:
        np.array(no_v_dofs): linear system system vector tau
    """
    no_p_dofs, no_v_dofs = count_dof(mesh)[0:2]
    b_gamma = np.zeros(no_v_dofs)

    # loop over interface couples
    for icut, cut in enumerate(mesh.cuts):

        iel_in, iel_ex = mesh.cuts[icut][0]
        edge2intface = mesh.cuts[icut][2] # list of intface edges of the cut

        # get no of intface edges associated to couple
        no_edges = len(edge2intface)
        first_ie_in = mesh.cuts[icut][1][0]
        # loop over intface edges of couple
        for ie in range(no_edges):
            # get local index of edge
            local_edge_idx = first_ie_in + ie
            # get index of edge on interface
            intf_ie = edge2intface[ie]
            # get edge_dofs
            [dof_x, dof_y] = get_edge_dofs(mesh, iel_in, local_edge_idx)
            shift_dof_x = dof_x - no_p_dofs
            shift_dof_y = dof_y - no_p_dofs
            # get edge length
            p1 = mesh.coords[mesh.elem2node[iel_in][local_edge_idx]]
            p2 = mesh.coords[mesh.elem2node[iel_in][(local_edge_idx+1)%len(mesh.elem2node[iel_in])]]
            hE = mema.R2_norm(p2-p1)
            # update with contribution
            b_gamma[shift_dof_x] = b_gamma[shift_dof_x] + t_gamma[intf_ie][0]*hE
            b_gamma[shift_dof_y] = b_gamma[shift_dof_y] + t_gamma[intf_ie][1]*hE

    return b_gamma

def interpolate_pressure (mesh, ref_p):
    """
    Args:
        mesh (ddr.Mesh): mesh
        ref_p  (lambda): reference pressure

    Returns:
        np.array(no_p_dofs): discrete pressure

    Remarks: L2 projection over T is replaced by value in barycenter
    """
    no_p_dofs = count_dof(mesh)[0]
    no_elems = len(mesh.elem2node)
    # initialize discrete pressure
    p_h = np.zeros(count_dof(mesh)[0])
    for iel in range(no_elems):
        xT = mesh.element_barycenter[iel]
        pT = ref_p(xT[0], xT[1])
        p_glob_dof = dof_loc2glob (mesh, iel, 0)
        p_h[p_glob_dof] = pT
    return p_h


def interpolate_velocity (mesh, ref_v):
    """
    Args:
        mesh (ddr.Mesh): mesh
        ref_v  (lambda): reference velocity

    Returns:
        np.array(no_v_dofs): discrete velocity

    Remarks: L2 projection over T/E replaced by value in barycenter
    """
    no_p_dofs, no_v_dofs = count_dof(mesh)[0:2]
    no_elems = len(mesh.elem2node)
    # initialize discrete velocity
    v_h = np.zeros(no_v_dofs)
    for iel in range(no_elems):
        xT = mesh.element_barycenter[iel]
        vT = ref_v(xT[0], xT[1])
        vTx_glob_dof = dof_loc2glob (mesh, iel, 1)
        vTy_glob_dof = dof_loc2glob (mesh, iel, 2)
        vTx_shifted_dof = vTx_glob_dof - no_p_dofs # need to shift because array has dim no_v_dofs
        vTy_shifted_dof = vTy_glob_dof - no_p_dofs
        v_h[vTx_shifted_dof] = vT[0]
        v_h[vTy_shifted_dof] = vT[1]

        no_edges = len(mesh.elem2edge[iel])   #assignments are repeated (could be optimised)
        for ie in range(no_edges):
            glob_ie = mesh.elem2edge[iel][ie]
            xE = mesh.edge_xE[glob_ie]
            vE = ref_v(xE[0], xE[1])
            vEx_glob_dof = dof_loc2glob (mesh, iel, 3 + 2*ie)
            vEy_glob_dof = dof_loc2glob (mesh, iel, 3 + 2*ie + 1)
            vEx_shifted_dof = vEx_glob_dof - no_p_dofs # need to shift because array has dim no_v_dofs
            vEy_shifted_dof = vEy_glob_dof - no_p_dofs
            v_h[vEx_shifted_dof] = vE[0]
            v_h[vEy_shifted_dof] = vE[1]

    return v_h


def interpolate_solution(mesh, ref_p, ref_v):
    """
    Args:
        mesh (ddr.Mesh): mesh
        ref_p  (lambda): reference pressure
        ref_v  (lambda): reference velocity

    Returns:
        np.array(no_tot_dofs): discrete solution in the conventinal format [p_h, v_h]

    Remarks: L2 projection over T/E replaced by value in barycenter
    """
    p_h = interpolate_pressure (mesh, ref_p)
    v_h = interpolate_velocity (mesh, ref_v)

    return np.concatenate([p_h, v_h])


def impose_bc(mesh, S, b, ref_sol_v, zero_mean=True):
    """
    Forces Dirichlet boundary conditions on velocity
    Enlarges the system with one line to account for zero mean pressure

    Args:
        mesh (ddr.Mesh): mesh
        S    (np.array): global system matrix
        b    (np.array): global system vector
        ref_sol_v  (lambda): reference velocity
        zero_mean (boolean): True if you want to force zero mean for pressure

    Returns: S (np.array): global system matrix
             b (np.array): global system array
    """

    # force dirichlet boundary conditions on velocity by adapting linear system
    for iel in range(len(mesh.elem2node)):
        no_edges = len(mesh.elem2edge[iel])
        for ie in range(no_edges):
            if (mesh.edge_bnd_mask[mesh.elem2edge[iel][ie]]>0):

                # get v dofs
                [dof1, dof2] = get_edge_dofs(mesh, iel, ie)

                # get position of edge
                glob_ie = mesh.elem2edge[iel][ie]
                [x, y] = mesh.edge_xE[glob_ie]

                # adapt linear system
                b[dof1] = ref_sol_v(x, y)[0]
                S[dof1,:] = 0.0
                S[dof1,dof1] = 1

                b[dof2] = ref_sol_v(x, y)[1]
                S[dof2,:] = 0.0
                S[dof2,dof2] = 1

    # add bottom line to enforce zero average of pressure
    if (zero_mean):
        no_p_dof = count_dof(mesh)[0]
        no_tot_dof = count_dof(mesh)[2]
        S_aug = np.zeros((no_tot_dof + 1, no_tot_dof + 1))
        b_aug = np.zeros(no_tot_dof + 1)
        S_aug[0:no_tot_dof, 0:no_tot_dof] = S
        b_aug[0:no_tot_dof] = b

        S_aug[0:no_p_dof, no_tot_dof] = 1

        for iel in range(len(mesh.elem2node)):
            mod_T = mesh.element_surface [iel]
            S_aug[no_tot_dof, iel] = mod_T

    return [S_aug, b_aug]



def visualize_solution (mesh, v_p, fig, axes, cmaps = ["magma", "viridis"], arrows="edge", arrow_density = 6):
    """
    Plots solution to Stokes Problem; on first axes velocity (magnitude),
    on second axes pressure

    Args:
        mesh     (ddr.Mesh): mesh
        v_p      (np.array): pressure_velocity_solution
        fig    (plt.figure): figure
        axes     (plt.axes): axes
        cmaps    (colormap): colormaps
        arrows     (string): if "edge" edge velocity is represented with arrows,
                             otherwise element velocity.
        arrow_density (int): from 0 to 6 to increase arrows that are displayed

    Returns: void
    """


    # separate solution for velocity and pressure and reshape velocity over elements
    no_p_dofs, no_v_dofs, tot_dofs = count_dof(mesh)
    sol_p = v_p[0:no_p_dofs]
    sol_v = v_p[no_p_dofs: tot_dofs]
    no_elems = len(mesh.elem2node)
    v_T_x = sol_v[0:2*no_elems:2]
    v_T_y = sol_v[1:2*no_elems:2]
    v_T =[np.array([v_T_x[k], v_T_y[k]]) for k in range(no_elems)]
    # take v norm over elements
    v_T_norm = [mema.R2_norm(v_T[k]) for k in range(no_elems)]

    # On first axes represent velocity (colormap on elements + arrows along interface)
    ax = axes[0]
    cmap = plt.get_cmap(cmaps[0])
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')

    max_v = np.max(v_T_norm)
    min_v = np.min(v_T_norm)

    for iel in range(len(mesh.elem2node)):
        verts = []
        nodal_values = []
        node_per_face = len(mesh.elem2node[iel])
        x_T = mesh.element_barycenter[iel]

        # draw polygon
        for ino in range(node_per_face):
            point = mesh.coords[mesh.elem2node[iel][ino]]
            verts.append(point[0:2])
        color = cmap((v_T_norm[iel]-min_v)/(max_v -min_v))
        element = Polygon(verts, closed=True,\
                          facecolor=color, alpha=0.8)
        ax.add_patch(element)

    # maximal length of velocity arrow is max elem size
    max_elem_size = max([np.sqrt(mesh.element_surface[iel]) for iel in range(len(mesh.elem2node))])
    if (arrows=="edge"):
        edge2elem = mesh.generate_edge2elem()
        for ie in range(len(edge2elem)):
            iel = edge2elem[ie][0]
            elem = mesh.elem2edge[iel]
            local_ie = [ i for i, k in enumerate(elem) if k==ie] [0]
            glob_ie = mesh.elem2edge[iel][local_ie]
            x_E = mesh.edge_xE [glob_ie]
            v_E_x = sol_v [2*len(mesh.elem2node) + 2*ie]
            v_E_y = sol_v [2*len(mesh.elem2node) + 2*ie + 1]
            # draw arrow
            k = random.randint (1, 6)
            if (k<=arrow_density):
                ax.arrow(x_E[0],x_E[1], v_E_x/max_v*max_elem_size,\
                         v_E_y/max_v*max_elem_size, color = 'white', alpha=0.8)
    else:
        for iel in range(len(mesh.elem2node)):
            x_T = mesh.element_barycenter[iel]
            # draw arrow
            k = random.randint (1, 6)
            if (k<=arrow_density):
                ax.arrow(x_T[0],x_T[1], v_T_x[iel]/max_v*max_elem_size,\
                         v_T_y[iel]/max_v*max_elem_size, color = 'white', alpha=0.8)

    # On second axes represent pressure
    ax = axes[1]
    cmap = plt.get_cmap(cmaps[1])
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')

    max_p = np.max(sol_p)
    min_p = np.min(sol_p)

    if (max_p==min_p):
        p = max_p
        max_p = p+1
        min_p = p-1


    for iel in range(len(mesh.elem2node)):
        verts = []
        nodal_values = []
        node_per_face = len(mesh.elem2node[iel])

        # draw polygon
        for ino in range(node_per_face):
            point = mesh.coords[mesh.elem2node[iel][ino]]
            glob_p_dof = dof_loc2glob (mesh, iel, 0)
            verts.append(point[0:2])
            p_T = sol_p[glob_p_dof]
        color = cmap((p_T-min_p)/(max_p -min_p))
        element = Polygon(verts, closed=True,\
                          facecolor=color, alpha=0.8)
        ax.add_patch(element)

    # colorbar settings for velocity axes
    ax = axes[0]
    norm = mcolors.Normalize(vmin=min_v, vmax=max_v)
    sm = cm.ScalarMappable(cmap=cmaps[0], norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('$|\!|v|\!|$', fontsize = 40)
    # colorbar settings for pressure axes
    ax = axes[1]
    norm = mcolors.Normalize(vmin=min_p, vmax=max_p)
    sm = cm.ScalarMappable(cmap=cmaps[1], norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('$p$', fontsize = 40)


def transfer_velocity_and_advect_interface(mesh, intface, p_v_lambda, ref_vol, dt, advect = True):
    """
    Displace interface transferring velocity from mesh edges

    Args:
        mesh (mema.Mesh2D): mesh
        intface (mema.disk_intface): interface
        p_v_lambda (np.array): stokes solution format (p,v,lambda)
        ref_vol (float): reference volume
        advect (boolean): whether or not to advect

    Returns:
        disk_intface: moved interface
    """

    # Loop over mesh interface edges and update new_intface.velocity
    for icut, cut in enumerate(mesh.cuts):
        iel_in, iel_ext = cut[0]
        no_edges = len(cut[2])
        first_ie_in = cut[1][0]
        for ie in range(no_edges):
            local_edge_idx = ie + first_ie_in
            intface_edge = mesh.cuts[icut][2][ie]
            [dof_x, dof_y] = get_edge_dofs(mesh, iel_in, local_edge_idx)
            v_x = p_v_lambda[dof_x]
            v_y = p_v_lambda[dof_y]
            intface.velocity[intface_edge] = np.array([v_x, v_y])
    # advect intface

    if (advect):
        new_intface = intface.advect(ref_vol, dt)
    else:
        new_intface = copy.deepcopy(intface)

    return new_intface

def solve_stokes (mesh, ref_sol_v, vol_force, nu_in, nu_ex, intface = None,\
                  with_surface_tension = False, external_tension = None, jump_penalization = False):
    """
    Solve Stokes problem with an interface and surface tension

    Args:
        mesh  (mema.Mesh2D): mesh
        intface (mema.disk_interface): interface
        ref_sol_v  (lambda): reference velocity
        vol_force  (lambda):
        nu          (float): viscosity (up to now uniform)
        external_tension (list(np.array)): external tension (e.g. Maxwell)
        jump_penalization (boolean): whether o apply jump penalization

    Returns:
             p_v_lambda  (np.array): solution (p, v, lambda) where lambda
                                     is Lagrange multiplier
             S (np.array): global system matrix (after bnd conditions)
             b (np.array): global system array  (after bnd conditions)
             A (np.array): matrix grad_s:grad_s
             B (np.array): matrix div*q
             PJ(np.array): jump penalization
             b_gamma (np.array): global system array  (after bnd conditions)
    """

    no_dof_p, no_dof_v, no_tot_dof = count_dof (mesh)

    # Calculate velocity reconstruction for every velocity element and velocity DOF
    # List such that v_rec[iel][loc_v_dof] = [A, b] (such that r^+1 = A(x-x_T) + b)
    # attention: when accessing v_rec[iel][v_dof], v_dof is shifted by 1 with respect to indexes of total dofs (skip pressure dof)

    no_elems = len(mesh.elem2node)
    v_rec = []

    for iel in range(no_elems):
        element_recs = []
        # Reconstruction for element v dofs
        for coord in range(2): # loop over x,y
            v_dof_T_coord = 1 + coord

            A, b = get_v_rec(mesh, iel, v_dof_T_coord)
            element_recs.append([A, b])

        no_edge = len(mesh.elem2edge[iel])
        for ie in range(no_edge):
            for coord in range(2):
                v_dof_E_coord = 3 + ie*2 + coord

                A, b = get_v_rec(mesh, iel, v_dof_E_coord)
                element_recs.append([A, b])

        v_rec.append(element_recs)

    # Assemble matrices and vector of linear system
    A = assemble_A (mesh, v_rec, nu_in, nu_ex)
    B = assemble_B (mesh)
    if (jump_penalization):
        JP = assemble_JP (mesh, v_rec, verbose=False)
    else:
        JP = 0*A
    b_f = assemble_b_f (mesh, vol_force)
    if (with_surface_tension):
        t_gamma = intface.calc_t_gamma()
        if (external_tension!=None):
            t_gamma = [t_gamma[k] - external_tension[k] for k in range(len(t_gamma))] # attention to sign
        b_gamma = assemble_b_gamma(mesh, t_gamma)
    else:
        b_gamma = 0*b_f

    b_v = b_f
    b_v = b_f + b_gamma #----------------> sign change of b_gamma

    # build global system (follow convention (p, v) for block ordering)

    zero_block = np.zeros((no_dof_p, no_dof_p))
    S = np.block([[zero_block, np.transpose(B)], [B, A + JP]])
    b_p = np.zeros(no_dof_p)
    b = np.concatenate([b_p, b_v])

    # Enforce boundary conditions (dirichlet bnd conds on velocity, zero avg cond on pressure)
    ## attention, system dimension augmented by 1 to account for a Lagrange multiplier
    [S, b] = impose_bc(mesh, S, b, ref_sol_v)

    ## Convert to sparse and solve (pay attention to Lagrange multiplier)
    S_sparse = csr_matrix(S)
    p_v_lambda = spsolve(S_sparse, b)

    return [p_v_lambda, S, b, A, B, JP, b_gamma]

def elem_velocity_energy_norm (mesh, iel, v_h):
    """
    Args:
        mesh(mesh2D): mesh
        iel (int): elem index
        v_h(np.array(float)): velocity (coefficients in HHO space), dim=no_v_dofs (only velocity array)
    """
    elem_energy_norm_sq = 0
    elem = mesh.elem2node[iel]
    # get local p0 velocities [v_T, {v_E}]
    elem_velocities = get_local_velocities (mesh, iel, v_h)
    # element contribution is 0 when k = 0
    # boundary contribution  1/hE *\int_E (v_E-v_T)^2
    v_T = elem_velocities [0]
    for ie in range(len(elem)):
        v_E = elem_velocities [1+ie]
        elem_energy_norm_sq += mema.R2_norm(v_T-v_E)**2

    return np.sqrt(elem_energy_norm_sq)

def velocity_energy_norm(mesh, v_h):
    """
    Args:
        mesh(mesh2D): mesh
        v_h(np.array(float)): velocity (coefficients in HHO space), dim=no_v_dofs (only velocity array)
    """
    energy_norm_sq = 0
    for iel in range(len(mesh.elem2node)):
        energy_norm_sq += elem_velocity_energy_norm (mesh, iel, v_h)**2

    return np.sqrt(energy_norm_sq)

def pressure_L2_norm(mesh, p_h):
    """
    Args:
        mesh(mesh2D): mesh
        p_h(np.array(float)): pressure (coefficients in HHO space) dim=no_p_dofs (only pressure array)
    """
    L2_norm_sq = 0
    for iel in range(len(mesh.elem2node)):
        L2_norm_sq += mesh.element_surface[iel]*p_h[iel]**2

    return np.sqrt(L2_norm_sq)

def p_v_error (mesh, p_v_lambda, ref_p, ref_v, normalized=True):
    """
    Args:
        mesh (mesh2D): mesh
        p_v_lambda (np.array(float)): HHO-stokes solution (with 1 DOF for Lagrange multiplier)
        ref_p (lambda): reference pressure
        ref_v (lambda): reference velocity
    Returns:
        list(float): [pressure L2 error, velocity energy error]
    """
    p_v_ref = interpolate_solution(mesh, ref_p, ref_v)
    no_p_dof, no_v_dof, no_tot_dof = count_dof(mesh)
    err = p_v_lambda[0:no_tot_dof] - p_v_ref # recall that p_v_lambda contains value of lagrange multiplie
    err_p = err[0:no_p_dof]
    err_v = err[no_p_dof:no_tot_dof]

    if (normalized):
        norm_p = pressure_L2_norm (mesh, p_v_ref[0:no_p_dof])
        norm_v = velocity_energy_norm (mesh, p_v_ref[no_p_dof: no_tot_dof])
        return [pressure_L2_norm(mesh, err_p)/(norm_p+1*(norm_p==0)), velocity_energy_norm(mesh, err_v)/(norm_v+1*(norm_v==0))]
    else:
        return [pressure_L2_norm(mesh, err_p), velocity_energy_norm(mesh, err_v)]
