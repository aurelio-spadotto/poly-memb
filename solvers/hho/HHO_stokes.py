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
            
def compute_dof_table (mesh):
    """
    Pre-compute the map loc2glob
    Args:
         mesh (mesh2D): mesh
    Returns:
         dof_table (list): list of local to global dof mapping for each element 
    """
    no_elems = len(mesh.elem2node)
    dof_table = [] 
    for iel in range(no_elems):
        element_dofs = []
        no_local_dofs = 1+2*(1+len(mesh.elem2node[iel])) # number of DOFs in element
        for dof in range(no_local_dofs): # loop over x,y
            element_dofs.append(dof_loc2glob(mesh, iel, dof))
        
        dof_table.append(element_dofs)
    
    return dof_table



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

def get_v_rec_from_localV_fast(mesh, iel, localV):
    """
    Made faster by removing numpy functions.

    Args:
        mesh                   (Mesh2D): mesh
        iel                         (int): element index
        localV   list(list): local function (v_T, {v_E}_E)

    Returns:
        tuple: [A11, A12, A21, A22, B1, B2]
    """
    # Average coincides with vT
    b = localV[0]
    b1, b2 = b[0], b[1]

    # Initialize grad{ r^{k+1}_T}
    A11, A12, A21, A22 = 0.0, 0.0, 0.0, 0.0

    # Loop over edges
    no_edges = len(mesh.elem2edge[iel])
    mod_T = mesh.element_surface[iel]  # NB: mod_T = h_T**2

    for ie in range(no_edges):
        nTE = mesh.edge_normal[iel][ie]
        vE = localV[1 + ie]

        # Compute outer product manually
        A11 += vE[0] * nTE[0]
        A12 += vE[0] * nTE[1]
        A21 += vE[1] * nTE[0]
        A22 += vE[1] * nTE[1]

    # Scale by 1/mod_T
    scale = 1.0 / mod_T
    A11 *= scale
    A12 *= scale
    A21 *= scale
    A22 *= scale

    return A11, A12, A21, A22, b1, b2


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

def get_v_rec_fast(mesh, iel, dof): # version that uses get_v_rec_from_localV
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

    return get_v_rec_from_localV_fast(mesh, iel, polynomials)



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

def local_contribution_aT_fast(mesh, v_rec, iel, i, j, nu):
    """
    Calculate local contribution aT (grad:grad)

    Args:
        mesh (Mesh2D): mesh
        v_rec (list): list of velocity reconstruction format tuple
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
    # Axy_i stands for A_i[x][y]
    A11_i, A12_i, A21_i, A22_i, b_1_i, b_2_i = v_rec[iel][i - 1] # skip pressure dof
    A11_j, A12_j, A21_j, A22_j, b_1_j, b_2_j = v_rec[iel][j - 1] # skjp pressure dof
    # take symmetric part of gradient
    sym_grad_11_i = A11_i
    sym_grad_12_i = 0.5*(A12_i + A21_i)
    sym_grad_21_i = sym_grad_12_i
    sym_grad_22_i = A22_i

    sym_grad_11_j = A11_j
    sym_grad_12_j = 0.5*(A12_j + A21_j)
    sym_grad_21_j = sym_grad_12_j
    sym_grad_22_j = A22_j    
    # get element surface
    mod_T = mesh.element_surface [iel]
    # get D(u):D(v)
    double_dot_product = (sym_grad_11_i*sym_grad_11_j + sym_grad_12_i*sym_grad_12_j +
                          sym_grad_21_i*sym_grad_21_j + sym_grad_22_i*sym_grad_22_j)
    return nu*mod_T*2*double_dot_product



def local_contribution_sT_fast(mesh, v_rec, iel, i, j):
    """
    Calculate local contribution sT (stabilization)

    Args:
        mesh (ddr.Mesh): mesh
        v_rec (list): list of velocity reconstrunction format tuple
        iel  (int): element index
        i    (int): local dof 1
        j    (int): local dof 2

    Returns:
        real: local contribution
    """
    contribution = 0.0

    # Get velocity reconstruction coefficients
    A11_i, A12_i, A21_i, A22_i, b1_i, b2_i = v_rec[iel][i - 1]
    A11_j, A12_j, A21_j, A22_j, b1_j, b2_j = v_rec[iel][j - 1]

    # Get element geometry
    mod_T = mesh.element_surface[iel]  # Element area
    xT = mesh.element_barycenter[iel]

    # Element contribution
    proj_i_T_x = b1_i
    proj_i_T_y = b2_i
    proj_j_T_x = b1_j
    proj_j_T_y = b2_j

    v_i_T_x, v_i_T_y = get_local_polynomials(mesh, iel, i)[0]
    v_j_T_x, v_j_T_y = get_local_polynomials(mesh, iel, j)[0]

    contribution += (
        ((proj_i_T_x - v_i_T_x) * (proj_j_T_x - v_j_T_x) +
         (proj_i_T_y - v_i_T_y) * (proj_j_T_y - v_j_T_y)) / mod_T
    )

    # Face contributions
    edge_per_elem = len(mesh.elem2edge[iel])
    for ie in range(edge_per_elem):
        # Get edge geometry
        glob_ie = mesh.elem2edge[iel][ie]
        xE = mesh.edge_xE[glob_ie]
        mod_E = mesh.edge_length[glob_ie]

        # Projection at edge
        proj_i_E_x = b1_i + A11_i * (xE[0] - xT[0]) + A12_i * (xE[1] - xT[1])
        proj_i_E_y = b2_i + A21_i * (xE[0] - xT[0]) + A22_i * (xE[1] - xT[1])

        proj_j_E_x = b1_j + A11_j * (xE[0] - xT[0]) + A12_j * (xE[1] - xT[1])
        proj_j_E_y = b2_j + A21_j * (xE[0] - xT[0]) + A22_j * (xE[1] - xT[1])

        # Face polynomials
        v_i_E_x, v_i_E_y = get_local_polynomials(mesh, iel, i)[1 + ie]
        v_j_E_x, v_j_E_y = get_local_polynomials(mesh, iel, j)[1 + ie]

        contribution += (
            ((proj_i_E_x - v_i_E_x) * (proj_j_E_x - v_j_E_x) +
             (proj_i_E_y - v_i_E_y) * (proj_j_E_y - v_j_E_y)) / mod_E
        )

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

def assemble_A_fast (mesh, v_rec, nu_in, nu_ex):
    """
    Assembles matrix A (constant viscosity)

    Args:
        mesh      (Mesh2D): mesh
        v_rec       (list): velocity reconstruction (list with elems [A,b])
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
            glob_i = dof_loc2glob(mesh, iel, i)
            for j in range(1, no_loc_dofs):
                glob_j = dof_loc2glob(mesh, iel, j)
                shifted_i = glob_i - no_p_dofs # offset in matrix which has dim: no_v_dofs*no_v_dofs
                shifted_j = glob_j - no_p_dofs # offset in matrix which has dim: no_v_dofs*no_v_dofs

                A[shifted_i, shifted_j] = A[shifted_i, shifted_j] +\
                                    local_contribution_aT_fast (mesh, v_rec, iel, i, j, nu) +\
                                    local_contribution_sT_fast (mesh, v_rec, iel, i, j)      # stab

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


def assemble_JP(mesh, v_rec, verbose=False):
    """
    Faster implementation of the Jump Penalization term assembly.
    Necessary to recover stability at lowest order (k=0).
    See di2020hybrid, sec. 7.6.

    Args:
       mesh (mesh2D): mesh
       v_rec (list): list of velocity reconstruction format [A,b]
       verbose (bool): verbosity flag

    Returns:
       np.array: JP (jump penalization matrix)
    """
    no_p_dofs, no_v_dofs = count_dof(mesh)[0:2]
    JP = np.zeros([no_v_dofs, no_v_dofs])

    # Ensure edge-to-element connectivity is available
    if not mesh.edge2elem:
        mesh.edge2elem = mesh.generate_edge2elem()
    edge2elem = mesh.edge2elem

    # Precompute quadrature points and weights
    quad_points = [0.0, 0.5, 1.0]
    quad_weights = [1.0 / 6, 4.0 / 6, 1.0 / 6]

    # Loop over edges
    for ie, associated_elems in enumerate(edge2elem):
        if verbose:
            print(f"> Adding contributions for edge: {ie}\n")

        # Get edge geometry
        first_iel = associated_elems[0]
        local_ie = next(k for k, iedge in enumerate(mesh.elem2edge[first_iel]) if iedge == ie)
        xE0 = mesh.coords[mesh.elem2node[first_iel][local_ie]]
        xE1 = mesh.coords[mesh.elem2node[first_iel][(local_ie + 1) % len(mesh.elem2node[first_iel])]]
        hE = mema.R2_norm(xE1 - xE0)

        # Precompute edge midpoint and scaled quadrature points
        edge_midpoint = 0.5 * (xE0 + xE1)
        scaled_quad_points = [xE0 + t * (xE1 - xE0) for t in quad_points]

        # Build DOF table
        dof_table = []
        for iel in associated_elems:
            elem = mesh.elem2node[iel]
            for component in range(2):  # Element velocity components
                dof_table.append([iel, 1 + component, False, []])
            for local_ie, edge in enumerate(mesh.elem2edge[iel]):
                is_shared = edge in edge2elem and len(edge2elem[edge]) > 1
                for component in range(2):  # Edge velocity components
                    local_dof = 1 + 2 + 2 * local_ie + component
                    other_side = []
                    if is_shared:
                        opposite_iel = next(e for e in edge2elem[edge] if e != iel)
                        local_dof_opposite = 1 + 2 + 2 * next(
                            k for k, opp_edge in enumerate(mesh.elem2edge[opposite_iel]) if opp_edge == edge
                        ) + component
                        other_side = [opposite_iel, local_dof_opposite]
                    dof_table.append([iel, local_dof, is_shared, other_side])

        # Compute contributions
        for dof_i in dof_table:
            iel_i, loc_dof_i, is_shared_i, other_side_i = dof_i
            glob_dof_i = dof_loc2glob(mesh, iel_i, loc_dof_i)
            A_i_0, b_i_0 = v_rec[iel_i][loc_dof_i - 1]
            xT_i_0 = mesh.element_barycenter[iel_i]
            A_i_1, b_i_1, xT_i_1 = (np.zeros((2, 2)), np.zeros(2), np.zeros(2))
            if is_shared_i:
                opposite_iel_i, opposite_loc_dof_i = other_side_i
                A_i_1, b_i_1 = v_rec[opposite_iel_i][opposite_loc_dof_i - 1]
                xT_i_1 = mesh.element_barycenter[opposite_iel_i]

            for dof_j in dof_table:
                iel_j, loc_dof_j, is_shared_j, other_side_j = dof_j
                glob_dof_j = dof_loc2glob(mesh, iel_j, loc_dof_j)
                A_j_0, b_j_0 = v_rec[iel_j][loc_dof_j - 1]
                xT_j_0 = mesh.element_barycenter[iel_j]
                A_j_1, b_j_1, xT_j_1 = (np.zeros((2, 2)), np.zeros(2), np.zeros(2))
                if is_shared_j:
                    opposite_iel_j, opposite_loc_dof_j = other_side_j
                    A_j_1, b_j_1 = v_rec[opposite_iel_j][opposite_loc_dof_j - 1]
                    xT_j_1 = mesh.element_barycenter[opposite_iel_j]

                # Compute integral of product of jumps
                integral_prod_jumps = 0.0
                if len(associated_elems) == 2:
                    sign_i = 2 * (iel_i == associated_elems[0]) - 1
                    sign_j = 2 * (iel_j == associated_elems[0]) - 1
                    for i_quad, x_quad in enumerate(scaled_quad_points):
                        pot_i_0 = np.dot(A_i_0, x_quad - xT_i_0) + b_i_0
                        pot_i_1 = np.dot(A_i_1, x_quad - xT_i_1) + b_i_1
                        pot_j_0 = np.dot(A_j_0, x_quad - xT_j_0) + b_j_0
                        pot_j_1 = np.dot(A_j_1, x_quad - xT_j_1) + b_j_1
                        pot_jump_i = (-pot_i_1 + pot_i_0) * sign_i
                        pot_jump_j = (-pot_j_1 + pot_j_0) * sign_j
                        integral_prod_jumps += quad_weights[i_quad] * np.dot(pot_jump_i, pot_jump_j)

                JP[glob_dof_i - no_p_dofs, glob_dof_j - no_p_dofs] += (
                    integral_prod_jumps / hE * (1 - 0.5 * is_shared_i) * (1 - 0.5 * is_shared_j)
                )

    return JP

def assemble_JP_fast(mesh, v_rec, glob_dof_table, verbose=False):
    """
    Faster implementation of the Jump Penalization term assembly.
    It relies on the global dof table to avoid recomputing the local
    to global mapping for each element. It avoids using numpy functions
    to speed up the process, especially in the internal loops

    Args:
       mesh (mesh2D): mesh
       glob_dof_table (list): glob2loc DOF table
       v_rec (list): list of velocity reconstruction format [A,b]
       verbose (bool): verbosity flag

    Returns:
       np.array: JP (jump penalization matrix)
    """
    no_p_dofs, no_v_dofs = count_dof(mesh)[0:2]
    JP = np.zeros([no_v_dofs, no_v_dofs])

    # Ensure edge-to-element connectivity is available
    if not mesh.edge2elem:
        mesh.edge2elem = mesh.generate_edge2elem()
    edge2elem = mesh.edge2elem

    # Precompute quadrature points and weights
    quad_points = [0.0, 0.5, 1.0]
    quad_weights = [1.0 / 6, 4.0 / 6, 1.0 / 6]

    # Loop over edges
    for ie, associated_elems in enumerate(edge2elem):
        if verbose:
            print(f"> Adding contributions for edge: {ie}\n")

        # Get edge geometry
        first_iel = associated_elems[0]
        local_ie = next(k for k, iedge in enumerate(mesh.elem2edge[first_iel]) if iedge == ie)
        xE0 = mesh.coords[mesh.elem2node[first_iel][local_ie]]
        xE1 = mesh.coords[mesh.elem2node[first_iel][(local_ie + 1) % len(mesh.elem2node[first_iel])]]
        hE = mema.R2_norm(xE1 - xE0)

        # Precompute edge midpoint and scaled quadrature points
        edge_midpoint = 0.5 * (xE0 + xE1)
        scaled_quad_points = [xE0 + t * (xE1 - xE0) for t in quad_points]

        # Build DOF table
        dof_table = []
        for iel in associated_elems:
            elem = mesh.elem2node[iel]
            for component in range(2):
                local_dof = 1 + component    # Element velocity components
                global_dof = glob_dof_table[iel][local_dof]
                dof_table.append([iel, local_dof, global_dof, False, []])
            for local_ie, edge in enumerate(mesh.elem2edge[iel]):
                is_shared = edge in edge2elem and len(edge2elem[edge]) > 1
                for component in range(2):  # Edge velocity components
                    local_dof = 1 + 2 + 2 * local_ie + component
                    try:
                        global_dof = glob_dof_table[iel][local_dof]
                    except Exception as e:
                        print ("LOOK:", local_dof, glob_dof_table, len(mesh.elem2node[iel]))
                        raise
                    other_side = []
                    if is_shared:
                        opposite_iel = next(e for e in edge2elem[edge] if e != iel)
                        local_dof_opposite = 1 + 2 + 2 * next(
                            k for k, opp_edge in enumerate(mesh.elem2edge[opposite_iel]) if opp_edge == edge
                        ) + component
                        other_side = [opposite_iel, local_dof_opposite]
                    dof_table.append([iel, local_dof, global_dof, is_shared, other_side])

        # Compute contributions
        for dof_i in dof_table:
            iel_i, loc_dof_i, glob_dof_i, is_shared_i, other_side_i = dof_i
            A_11_i0, A_12_i0, A_21_i0, A_22_i0, b_1_i0, b_2_i0 = v_rec[iel_i][loc_dof_i - 1]
            xT_i_0 = mesh.element_barycenter[iel_i]
            A_11_i1, A_12_i1, A_21_i1, A_22_i1, b_1_i1, b_2_i1 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            xT_i_1 = (0.0, 0.0)
            if is_shared_i:
                opposite_iel_i, opposite_loc_dof_i = other_side_i
                A_11_i1, A_12_i1, A_21_i1, A_22_i1, b_1_i1, b_2_i1 = v_rec[opposite_iel_i][opposite_loc_dof_i - 1]
                xT_i_1 = mesh.element_barycenter[opposite_iel_i]

            for dof_j in dof_table:
                iel_j, loc_dof_j, glob_dof_j, is_shared_j, other_side_j = dof_j
                A_11_j_0, A_12_j_0, A_21_j_0, A_22_j_0, b_1_j_0, b_2_j_0 = v_rec[iel_j][loc_dof_j - 1]
                xT_j_0 = mesh.element_barycenter[iel_j]
                A_11_j_1, A_12_j_1, A_21_j_1, A_22_j_1, b_1_j_1, b_2_j_1 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                xT_j_1 = (0.0, 0.0)
                if is_shared_j:
                    opposite_iel_j, opposite_loc_dof_j = other_side_j
                    A_11_j_1, A_12_j_1, A_21_j_1, A_22_j_1, b_1_j_1, b_2_j_1 = v_rec[opposite_iel_j][opposite_loc_dof_j - 1]
                    xT_j_1 = mesh.element_barycenter[opposite_iel_j]

                # Compute integral of product of jumps
                integral_prod_jumps = 0.0
                if len(associated_elems) == 2:
                    sign_i = 2 * (iel_i == associated_elems[0]) - 1
                    sign_j = 2 * (iel_j == associated_elems[0]) - 1
                    for i_quad, x_quad in enumerate(scaled_quad_points):
                        # Compute potentials for element i
                        pot_i_0_x = A_11_i0 * (x_quad[0] - xT_i_0[0]) + A_12_i0 * (x_quad[1] - xT_i_0[1]) + b_1_i0
                        pot_i_0_y = A_21_i0 * (x_quad[0] - xT_i_0[0]) + A_22_i0 * (x_quad[1] - xT_i_0[1]) + b_2_i0
                        pot_i_1_x = A_11_i1 * (x_quad[0] - xT_i_1[0]) + A_12_i1 * (x_quad[1] - xT_i_1[1]) + b_1_i1
                        pot_i_1_y = A_21_i1 * (x_quad[0] - xT_i_1[0]) + A_22_i1 * (x_quad[1] - xT_i_1[1]) + b_2_i1

                        # Compute potentials for element j
                        pot_j_0_x = A_11_j_0 * (x_quad[0] - xT_j_0[0]) + A_12_j_0 * (x_quad[1] - xT_j_0[1]) + b_1_j_0
                        pot_j_0_y = A_21_j_0 * (x_quad[0] - xT_j_0[0]) + A_22_j_0 * (x_quad[1] - xT_j_0[1]) + b_2_j_0
                        pot_j_1_x = A_11_j_1 * (x_quad[0] - xT_j_1[0]) + A_12_j_1 * (x_quad[1] - xT_j_1[1]) + b_1_j_1
                        pot_j_1_y = A_21_j_1 * (x_quad[0] - xT_j_1[0]) + A_22_j_1 * (x_quad[1] - xT_j_1[1]) + b_2_j_1

                        # Compute jumps
                        pot_jump_i_x = (-pot_i_1_x + pot_i_0_x) * sign_i
                        pot_jump_i_y = (-pot_i_1_y + pot_i_0_y) * sign_i
                        pot_jump_j_x = (-pot_j_1_x + pot_j_0_x) * sign_j
                        pot_jump_j_y = (-pot_j_1_y + pot_j_0_y) * sign_j

                        # Compute dot product of jumps
                        dot_product_jumps = pot_jump_i_x * pot_jump_j_x + pot_jump_i_y * pot_jump_j_y
                        integral_prod_jumps += quad_weights[i_quad] * dot_product_jumps

                JP[glob_dof_i - no_p_dofs, glob_dof_j - no_p_dofs] += (
                    integral_prod_jumps / hE * (1 - 0.5 * is_shared_i) * (1 - 0.5 * is_shared_j)
                )

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
        dt (float): time step
        rf_vol (float): volume of the interface
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
    dof_table = compute_dof_table(mesh) # precompute dof table

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

def solve_stokes_fast (mesh, ref_sol_v, vol_force, nu_in, nu_ex,\
                       intface = None, with_surface_tension = False,\
                       external_tension = None, jump_penalization = False):
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
    glob_dof_table = compute_dof_table(mesh) # precompute dof table

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

            element_recs.append(get_v_rec_fast(mesh, iel, v_dof_T_coord))

        no_edge = len(mesh.elem2edge[iel])
        for ie in range(no_edge):
            for coord in range(2):
                v_dof_E_coord = 3 + ie*2 + coord

                element_recs.append(get_v_rec_fast(mesh, iel, v_dof_E_coord)) # as a tuple 

        v_rec.append(element_recs)

    # Assemble matrices and vector of linear system
    A = assemble_A_fast (mesh, v_rec, nu_in, nu_ex)
    B = assemble_B (mesh)
    if (jump_penalization):
        JP = assemble_JP_fast (mesh, v_rec, glob_dof_table, verbose=False)
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
