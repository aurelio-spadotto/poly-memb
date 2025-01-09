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

def disc_grad (mesh, iel):
    """
    Computes the discrete gradient matrix of an element,
    such that GRAD*[nodal values] returns the gradient of the local
    distribution of dofs [nodal values], i.e.:
    j-th column is discrete gradient of basis function of node j

    Args:
        mesh (mema.mesh2D): mesh
        iel     (int): element index
    """
    elem = mesh.elem2node[iel]
    node_per_elem = len(elem)
    GRAD = np.zeros((2, node_per_elem))
    for ino in range(node_per_elem):
        start1 = end2 = mesh.coords[elem[ino]]
        start2 = mesh.coords[elem[(ino-1)%node_per_elem]]
        end1   = mesh.coords[elem[(ino+1)%node_per_elem]]
        tangent1 = (end1-start1)
        tangent2 = (end2-start2)
        # get normal by rotation
        normal1 = np.array([[0,1],[-1,0]])@tangent1
        normal2 = np.array([[0,1],[-1,0]])@tangent2
        norm1 = normal1/np.linalg.norm(normal1)
        norm2 = normal2/np.linalg.norm(normal2)
        mod_T = mesh.calc_surface(iel)
        GRAD[:,ino] = 0.5/mod_T*(normal1+normal2)
    return GRAD

def p1_rec_matrix (mesh,iel):
    """
    Returns the P1 reconstruction matrix such that [p1 rec nodal values] = R*[nodal_values]

    Args:
        mesh (mesh.mesh2D): mesh
        iel          (int): element index
    Returns:
        np.array
    """
    def simpson (a, b, fun, with_s = False, sign = 'plus'):
        """
        Simpson quadrature
        """
        # a, b: points in 2D space (np arrays)
        # fun: real function with double argument
        mid1 = a + (b-a)/4
        mid2 = a + (b-a)*2/4
        mid3 = a + (b-a)*3/4
        if   (with_s and sign =='minus'):
            return  np.linalg.norm(b-a)/90*(7*fun(a[0], a[1]) +
                                            24*fun(mid1[0],mid1[1]) +
                                            6*fun(mid2[0],mid2[1]) +
                                            8*fun(mid3[0],mid3[1]))
        elif (with_s and sign =='plus'):
            return  np.linalg.norm(b-a)/90*(8*fun(mid1[0],mid1[1]) +
                                            6*fun(mid2[0],mid2[1]) +
                                           24*fun(mid3[0],mid3[1]) +
                                            7*fun(b[0], b[1]))
        else:
            return  np.linalg.norm(b-a)/90*(7*fun(a[0], a[1]) +
                                            32*fun(mid1[0],mid1[1]) +
                                            12*fun(mid2[0],mid2[1]) +
                                            32*fun(mid3[0],mid3[1]) +
                                            7*fun(b[0], b[1]))

    elem = mesh.elem2node[iel]
    node_per_elem = len(elem)
    dim_basis     = 3 #dimension of basis of R,c2(T)
    V = np.zeros ((node_per_elem, dim_basis)) #evaluation matrix
    M = np.zeros ((dim_basis, dim_basis))     #the mass matrix to invert
    B1 = np.zeros ((dim_basis,node_per_elem))  #the rhs matrix (part 1)
    B2 = np.zeros ((dim_basis,node_per_elem))  #the rhs matrix (part 2)
    x_T = mesh.barycenter(iel)
    #define the psi_i (for edge contribution of B)
    psi = [None for _ in range(dim_basis)]
    psi [0] = lambda x,y:  np.array([x-x_T[0],y-x_T[1]])
    psi [1] = lambda x,y:  np.array([x-x_T[0],y-x_T[1]])*(x-x_T[0])
    psi [2] = lambda x,y:  np.array([x-x_T[0],y-x_T[1]])*(y-x_T[1])
    #define the phi_i (for V)
    phi = [None for _ in range(dim_basis)]
    phi [0] = lambda x,y: 2.
    phi [1] = lambda x,y: 3*(x - x_T[0])
    phi [2] = lambda x,y: 3*(y - x_T[1])
    #define the csi_ij (integrated phi_i*phi_j, for M)
    csi = [[None for _ in range(dim_basis)] for _ in range (dim_basis)]
    csi [0][0] =             lambda x,y: np.array([2*(x-x_T[0])                    , 2*(y-x_T[1])])
    csi [0][1] = csi[1][0] = lambda x,y: np.array([0.                              , 6*(y-x_T[1])*(x-x_T[0])])
    csi [0][2] = csi[2][0] = lambda x,y: np.array([6*(x-x_T[0])*(y-x_T[1])         , 0.])
    csi [1][1] =             lambda x,y: np.array([0.                              , 9*(y-x_T[1])*((x-x_T[0])**2)])
    csi [1][2] = csi[2][1] = lambda x,y: np.array([(9/4)*((x-x_T[0])**2)*(y-x_T[1]), (9/4)*(x-x_T[0])*((y-x_T[1])**2)])
    csi [2][2] =             lambda x,y: np.array([9*(x-x_T[0])*((y-x_T[1])**2)    , 0.])
    #define the chi_iN  (for cell contribution in B)
    chi_fac = [None for _ in range (dim_basis)]
    chi_fac[0] = lambda  x,y: np.array([(x-x_T[0])*(y-x_T[1])         , (x-x_T[0])*(y-x_T[1])])
    chi_fac[1] = lambda  x,y: np.array([0.5*((x-x_T[0])**2)*(y-x_T[1]), (y-x_T[1])*((x-x_T[0])**2)])
    chi_fac[2] = lambda  x,y: np.array([(x-x_T[0])*((y-x_T[1])**2)    , 0.5*((y-x_T[1])**2)*(x-x_T[0])])
    #fill V
    for i in range(node_per_elem):
          xx = mesh.coords[elem[i]][0]
          yy = mesh.coords[elem[i]][1]
          for j in range(dim_basis):
              V[i][j] = phi[j](xx,yy)
    #fill M
    for i in range (dim_basis):
          for j in range (dim_basis):
              # sum integrals over E of csi*n
              for k in range(node_per_elem):
                  # get direction of edge
                  start = mesh.coords[elem[k]]
                  end   = mesh.coords[elem[(k+1)%node_per_elem]]
                  tangent = (end-start)/np.linalg.norm(end-start)
                  # get normal by rotation
                  normal = np.array([[0,1],[-1,0]])@tangent
                  M [i][j] = M [i][j] + simpson(start, end, lambda x,y: np.dot(normal,csi[i][j] (x,y)))
    #fill B
    for i in range(dim_basis):
          for j in range (node_per_elem):
                  #face contribution form left and right (need 2 normals)
                  start1 = end2 = mesh.coords[elem[j]] #dx
                  start2 = mesh.coords[elem[(j-1)%node_per_elem]] #sx
                  end1   = mesh.coords[elem[(j+1)%node_per_elem]]
                  tangent1 = (end1-start1)
                  tangent2 = (end2-start2)
                  # get normal by rotation
                  normal1 = np.array([[0,1],[-1,0]])@tangent1
                  normal2 = np.array([[0,1],[-1,0]])@tangent2
                  norm1 = normal1/np.linalg.norm(normal1)
                  norm2 = normal2/np.linalg.norm(normal2)
                  B2 [i][j] = B2 [i][j] + simpson(start1, end1,
                                                lambda x,y: np.dot(norm1,psi[i](x,y)), True, 'minus')#dx
                  B2 [i][j] = B2 [i][j] + simpson(start2, end2,
                                                lambda x,y: np.dot(norm2,psi[i](x,y)), True, 'plus')#sx
                  #cell contribution: must sum integrals over E of chi*n
                  e_avg = 0.5*(normal1 + normal2)
                  chi_ij = lambda x,y: np.multiply (np.flip(e_avg), chi_fac[i](x,y))
                  for k in range(node_per_elem):
                      start = mesh.coords[elem[k]]
                      end = mesh.coords[elem[(k+1)%node_per_elem]]
                      tangent = (end-start)/np.linalg.norm(end-start)
                      # get normal by rotation
                      normal = np.array([[0,1],[-1,0]])@tangent
                      # get surface of face
                      mod_T = mesh.calc_surface (iel)
                      B1[i][j] = B1[i][j] -(1/mod_T)* simpson(start, end, lambda x,y: np.dot(normal,chi_ij(x,y)))

    #calculate the final local matrix R and store it
    R = np.dot(V, np.linalg.solve(M, (B1+B2)))

    return R


def glob_idx (mesh, iel, ino):
    """
    Local DOF to global DOF connectivity for solver DDR_interfaces
    Convention for dof ordering: [internal unknowns, intface unknowns side in, intface unknowns side ex]

    Args:
        mesh (mema.mesh2D): mesh
        iel  (int): element index
        ino  (int): local node index
    """
    idx_point = mesh.elem2node[iel][ino]
    # get number of points lying on interface
    points_on_intface = sum([ len(mesh.cuts[icut][2]) for icut in range(len(mesh.cuts))])
    points_in_bulk = len(mesh.coords) - points_on_intface
    if (idx_point<points_in_bulk):
    # point is from the original mesh: not on the int.face
        return idx_point
    else:
    # interface point, doubled: unknown structure:
    # [internal unknowns, internal intface unk, external intface unknowns]
        side = mesh.side_mask[iel]
        if (side==0):
            return points_in_bulk + (idx_point-points_in_bulk)
        elif (side==1):
            #return points_on_intface +(Npoints-points_on_intface)+(idx_point-points_on_intface)
            return idx_point + points_on_intface
        else:
            print ("Error: this face is cut")

def count_dof (mesh):
    """
    Total DOF number
    """
    points_on_intface = sum([ len(mesh.cuts[icut][2]) for icut in range(len(mesh.cuts))])
    points_in_bulk = len(mesh.coords) - points_on_intface

    return points_in_bulk + 2*points_on_intface

def assemble_G(mesh, sigma_in, sigma_ex):
    """
    Assemble matrix G (grad-grad)

    Args:
        mesh (mema.mesh2D): mesh
        sigma_in (float): diff. coefficient internal
        sigma_ex (float): diff. coefficient external
    """
    no_tot_dof = count_dof(mesh)
    G = np.zeros((no_tot_dof, no_tot_dof))
    for iel in range(len(mesh.elem2node)):
        side = mesh.side_mask[iel]
        elem = mesh.elem2node[iel]
        node_per_elem = len(elem)
        GRAD = disc_grad (mesh,iel)
        mod_T = mesh.calc_surface (iel)
        sigma = (1-side)*sigma_in + side*sigma_ex
        for i in range(node_per_elem):
            for j in range(node_per_elem):
                glob_i = glob_idx(mesh,iel,i)
                glob_j = glob_idx(mesh,iel,j)
                G[glob_i,glob_j] = G[glob_i,glob_j] +sigma*mod_T*np.dot(GRAD[:,i],GRAD[:,j])
    return G

def assemble_S(mesh, sigma_in, sigma_ex):
    """
    Assemble matrix S (stabilization)

    Args:
        mesh (mema.mesh2D): mesh
        sigma_in (float): diff. coefficient internal
        sigma_ex (float): diff. coefficient external
    """
    no_tot_dof = count_dof(mesh)
    S = np.zeros((no_tot_dof, no_tot_dof))
    for iel in range(len(mesh.elem2node)):
       elem = mesh.elem2node[iel]
       node_per_elem = len(elem)
       sigma = sigma_ex*mesh.side_mask[iel] + sigma_in*(1-mesh.side_mask[iel])
       R = p1_rec_matrix (mesh, iel)
       for i in range(node_per_elem):

           for j in range(node_per_elem):
               glob_i = glob_idx(mesh,iel,i)
               glob_j = glob_idx(mesh,iel,j)

               for ino in range (node_per_elem):
                   iA = R[ino, i]- mema.kron(i, ino)
                   iB = R[(ino+1)%node_per_elem, i]- mema.kron(i, (ino+1)%node_per_elem)
                   jA = R[ino, j]- mema.kron(j, ino)
                   jB = R[(ino+1)%node_per_elem, j]- mema.kron(j, (ino+1)%node_per_elem)

                   S[glob_i,glob_j] += sigma*calc_r2_prod_integ(iA, iB, jA, jB)
    return S

def assemble_M_gamma(mesh, sigma_in, sigma_ex):
    """
    Assembles matrix M_gamma (gradient jump at interface)

    Args:
        mesh (mema.mesh2D): mesh
        sigma_in (float): diff. coefficient internal
        sigma_ex (float): diff. coefficient external
   """

    no_tot_dof = count_dof(mesh)
    M_gamma = np.zeros((no_tot_dof, no_tot_dof))

    for icut in range(len(mesh.cuts)):

        iel_in = mesh.cuts[icut][0][0]
        iel_ex = mesh.cuts[icut][0][1]

        first_ie_in = mesh.cuts[icut][1][0]
        first_ie_ex = mesh.cuts[icut][1][1]

        # weights for average
        lambda_in = sigma_ex/(sigma_ex +sigma_in)
        lambda_ex = sigma_in/(sigma_ex +sigma_in)

        no_intf_nodes  = len(mesh.cuts[icut][2]) +1

        edge_dofs = [] # list of lists of dofs activated by edge [global_dof, elem]

        node_per_elem_in = len(mesh.elem2node[iel_in])
        node_per_elem_ex = len(mesh.elem2node[iel_ex])
        tot_dofs = node_per_elem_in + node_per_elem_ex

        # dofs iel_in
        for ino_in in range(node_per_elem_in):
            glob_dof = glob_idx(mesh, iel_in, ino_in)
            iel = iel_in
            edge_dofs.append([glob_dof, iel])
        # dofs iel_ex
        for ino_ex in range(node_per_elem_ex):
            glob_dof = glob_idx(mesh, iel_ex, ino_ex)
            iel = iel_ex
            edge_dofs.append([glob_dof, iel])


        # loop over dof1
        for idof1 in range(tot_dofs):
            dof1 = edge_dofs[idof1][0] #global idx
            if (idof1<node_per_elem_in): #local idx
                dof1_loc = idof1
            else:
                dof1_loc = idof1-node_per_elem_in
            iel1 = edge_dofs[idof1][1] #idx of face

            GR1 = disc_grad(mesh, iel1)
            sigma1 = sigma_in*int(iel1==iel_in) + sigma_ex*int(iel1==iel_ex)
            lambda1 = lambda_in*int(iel1==iel_in) + lambda_ex*int(iel1==iel_ex)
            sign1 = int(iel1==iel_in)-int(iel1==iel_ex)

            # loop over dof2
            for idof2 in range(tot_dofs):
                dof2 = edge_dofs[idof2][0] #global idx
                if (idof2<node_per_elem_in): #local idx
                    dof2_loc = idof2
                else:
                    dof2_loc = idof2-node_per_elem_in
                iel2 = edge_dofs[idof2][1] #idx of face

                # loop over interface edges of the couple
                for ied in range(no_intf_nodes - 1):
                    ino1_edge = mesh.elem2node[iel_in][(first_ie_in + ied)%node_per_elem_in]
                    ino2_edge = mesh.elem2node[iel_in][(first_ie_in + ied+1)%node_per_elem_in]
                    ino_dof2  = mesh.elem2node[iel2][dof2_loc]

                    # non null contribution if edge in support of phi_j
                    if (ino_dof2==ino1_edge or ino_dof2==ino2_edge):
                        # find normal
                        tangent = mesh.coords[mesh.elem2node[iel_in][(first_ie_in + ied + 1)%node_per_elem_in]] \
                                 -mesh.coords[mesh.elem2node[iel_in][(first_ie_in + ied)%node_per_elem_in]]
                        #normal  = sign1*np.array([[0,1],[-1,0]])@tangent
                        normal   = np.array([[0,1],[-1,0]])@tangent # since elem_in is taken no need to multiply by sign1
                        # G(phi_i) times normal
                        mean_sigma_grad_normal = lambda1*sigma1*np.dot(GR1[:,dof1_loc], normal)
                        # sign of jump
                        sign2 = int(iel2==iel_ex) - int(iel2==iel_in)
                        # add contribution
                        M_gamma[dof1,dof2] += mean_sigma_grad_normal*sign2*0.5

    return M_gamma

def calc_r2_prod_integ(v1A,v1B,v2A,v2B):
    """
    Calculates the integral over the [0,1] of the product between 2 affine functions assigned through their
    values at vertices 0 and 1 (first values at 0 of f1 and f2 an then values at 1)
    (a bit redeundant now that there is quadrature)

    Args:
        v1A (float): value of p1 function 1 at node A (0)
        v1B (float): value of p1 function 1 at node B (1)
        v2A (float): value of p1 function 2 at node A (0)
        v2B (float): value of p1 function 2 at node A (1)
    Returns:
        float
    """
    return v1A*v2A*1/3 + (v1A*v2B+v1B*v2A)*1/6 + v1B*v2B*1/3


def assemble_N_gamma(mesh, sigma_in, sigma_ex):
    """
    Assembles matrix N_gamma (jump at interface of edge values)

    Args:
        mesh (mema.mesh2D): mesh
        sigma_in (float): diff. coefficient internal
        sigma_ex (float): diff. coefficient external
   """

    no_tot_dof = count_dof(mesh)
    N_gamma = np.zeros((no_tot_dof, no_tot_dof))

    alpha = sigma_in*sigma_ex/(sigma_in +sigma_ex)

    for icut in range(len(mesh.cuts)):

        iel_in = mesh.cuts[icut][0][0]
        iel_ex = mesh.cuts[icut][0][1]

        first_ie_in = mesh.cuts[icut][1][0]
        first_ie_ex = mesh.cuts[icut][1][1]

        no_intf_nodes  = len(mesh.cuts[icut][2]) + 1

        edge_dofs = []

        node_per_elem_in = len(mesh.elem2node[iel_in])
        node_per_elem_ex = len(mesh.elem2node[iel_ex])
        tot_dofs = node_per_elem_in + node_per_elem_ex

        # dofs iel_in
        for ino_in in range(node_per_elem_in):
            glob_dof = glob_idx(mesh, iel_in, ino_in)
            iel = iel_in
            edge_dofs.append([glob_dof, iel])
        # dofs iel_ex
        for ino_ex in range(node_per_elem_ex):
            glob_dof = glob_idx(mesh, iel_ex, ino_ex)
            iel = iel_ex
            edge_dofs.append([glob_dof, iel])

        # loop over dof1
        for idof1 in range(tot_dofs):

            dof1 = edge_dofs[idof1][0] #global idx
            if (idof1<node_per_elem_in): #local idx
                dof1_loc = idof1
            else:
                dof1_loc = idof1-node_per_elem_in
            iel1 = edge_dofs[idof1][1] #idx of elem

            # loop over dof2
            for idof2 in range(tot_dofs):

                dof2 = edge_dofs[idof2][0] #global idx
                if (idof2<node_per_elem_in): #local idx
                    dof2_loc = idof2
                else:
                    dof2_loc = idof2-node_per_elem_in
                iel2 = edge_dofs[idof2][1] #idx of elem


                # loop over interelem edges of the couple
                for ied in range(no_intf_nodes - 1):

                    ino1_edge = mesh.elem2node[iel_in][(first_ie_in+ied)%node_per_elem_in]
                    ino2_edge = mesh.elem2node[iel_in][(first_ie_in+ied+1)%node_per_elem_in]
                    ino_dof1  = mesh.elem2node[iel1][dof1_loc]
                    ino_dof2  = mesh.elem2node[iel2][dof2_loc]

                    v1A = mema.kron(ino_dof1, ino1_edge)
                    v1B = mema.kron(ino_dof1, ino2_edge)
                    v2A = mema.kron(ino_dof2, ino1_edge)
                    v2B = mema.kron(ino_dof2, ino2_edge)

                    sign1 = int(iel1==iel_ex)-int(iel1==iel_in)
                    sign2 = int(iel2==iel_ex)-int(iel2==iel_in)

                    tangent = mesh.coords[mesh.elem2node[iel_in] [(first_ie_in+ied + 1)%node_per_elem_in]] \
                             -mesh.coords[mesh.elem2node[iel_in] [(first_ie_in+ied)%node_per_elem_in]]
                    mod_E = np.linalg.norm(tangent)

                    # add contribution
                    N_gamma[dof1,dof2] += alpha*sign1*sign2*calc_r2_prod_integ(v1A,v1B,v2A,v2B)

    return N_gamma



def assemble_b_J(mesh, J_datum, sigma_in, sigma_ex):
    """
    Assembles rhs B_J (jump at interface)

    Args:
        mesh (mema.mesh2D): mesh
        J_datum (lambda): jump value
        sigma_in (float): diff. coefficient internal
        sigma_ex (float): diff. coefficient external
   """
    no_tot_dof = count_dof(mesh)
    B_J_gamma = np.zeros(no_tot_dof)
    alpha = sigma_in*sigma_ex/(sigma_in +sigma_ex)

    for icut in range(len(mesh.cuts)):

        iel_in = mesh.cuts[icut][0][0]
        iel_ex = mesh.cuts[icut][0][1]

        no_intf_nodes  = len(mesh.cuts[icut][2]) + 1

        first_ie_in = mesh.cuts[icut][1][0]
        first_ie_ex = mesh.cuts[icut][1][1]

        tot_dofs = 2*no_intf_nodes + 3 # intf nodes counted 2 times + original vertices
        edge_dofs = []

        node_per_elem_in = len(mesh.elem2node[iel_in])
        node_per_elem_ex = len(mesh.elem2node[iel_ex])
        tot_dofs = node_per_elem_in + node_per_elem_ex

        # dofs iel_in
        for ino_in in range(node_per_elem_in):
            glob_dof = glob_idx(mesh, iel_in, ino_in)
            iel = iel_in
            edge_dofs.append([glob_dof, iel])
        # dofs iel_ex
        for ino_ex in range(node_per_elem_ex):
            glob_dof = glob_idx(mesh, iel_ex, ino_ex)
            iel = iel_ex
            edge_dofs.append([glob_dof, iel])

        # loop over dof
        for idof in range(tot_dofs):
            dof = edge_dofs[idof][0] #global idx
            if (idof<node_per_elem_in): #local idx
                dof_loc = idof
            else:
                dof_loc = idof-node_per_elem_in
            iel = edge_dofs[idof][1] #idx of face

            sign = int(iel==iel_ex)-int(iel==iel_in)

            # loop over interface edges of the couple
            for ied in range(no_intf_nodes - 1):

                ino1_edge = mesh.elem2node[iel_in][(first_ie_in+ied)%node_per_elem_in]
                ino2_edge = mesh.elem2node[iel_in][(first_ie_in+ied+1)%node_per_elem_in]
                ino_dof   = mesh.elem2node[iel][dof_loc]

                v1A = mema.kron(ino_dof, ino1_edge)
                v1B = mema.kron(ino_dof, ino2_edge)

                v2A = J_datum(mesh.coords[ino1_edge][0], mesh.coords[ino1_edge][1])
                v2B = J_datum(mesh.coords[ino2_edge][0], mesh.coords[ino2_edge][1])

                tangent = mesh.coords[mesh.elem2node[iel_in] [(first_ie_in+ied+1)%node_per_elem_in]] \
                         -mesh.coords[mesh.elem2node[iel_in] [(first_ie_in+ied)%node_per_elem_in]]
                mod_E = np.linalg.norm(tangent)

                # add contribution
                B_J_gamma[dof] += alpha*sign*calc_r2_prod_integ(v1A,v1B,v2A,v2B)

    return B_J_gamma



# reference solution (takes dof as argument to treat intface nodes)
def reference_solution (mesh, dof, rho, ref_sol_in, ref_sol_ex, t = 0, t_dep=False):
    if (dof<mesh.Npoints):
        point = mesh.coords[dof,:]
    else:
        shift = mesh.Npoints-mesh.Npoints_init
        point = mesh.coords[dof-shift,:]
    # position_code: 0 internal, 1 intface in, 2 intface ex
    on_intface = 0
    if (dof>=mesh.Npoints_init):
        on_intface = 1
    if (dof>=mesh.Npoints):
        on_intface = 2
    radius = np.sqrt(point[0]**2+point[1]**2)
    if (t_dep):
        if (radius<rho and  on_intface==0):
            return ref_sol_in(point[0],point[1], t)
        if (radius>rho and on_intface==0):
            return ref_sol_ex(point[0],point[1], t)
        if (on_intface==1):
            return ref_sol_in(point[0],point[1], t)
        if (on_intface==2):
            return ref_sol_ex(point[0],point[1], t)
        else:
            print ("Error: something is wrong with point coordinates")
    else:
        if (radius<rho and  on_intface==0):
            return ref_sol_in(point[0],point[1])
        if (radius>rho and on_intface==0):
            return ref_sol_ex(point[0],point[1])
        if (on_intface==1):
            return ref_sol_in(point[0],point[1])
        if (on_intface==2):
            return ref_sol_ex(point[0],point[1])
        else:
            print ("Error: something is wrong with point coordinates")

def impose_bc(mesh, A, b, ref_sol):
    """
    Modifies the system to impose boundary conditions

    Args:
        mesh (mema.mesh2D): mesh
        A    (np.array): lhs matrix
        b    (np.array): rhs array
        ref_sol (lambda): reference solution
    """

    for iel in range(len(mesh.elem2node)):
        node_per_elem = len(mesh.elem2node[iel])
        for ino in range(node_per_elem):
            node = mesh.elem2node[iel] [ino]
            dof = glob_idx(mesh, iel, ino)
            if (mesh.node_bnd_mask[node]>0):
                b[dof] = ref_sol(mesh, iel, ino)
                A[dof,:] = 0.0
                A[dof,dof] = 1
    return [A,b]

def calc_L0_error (mesh, u, ref_sol):
    """
    Calculate L0 error
    Args:
        mesh (mema.mesh2D): mesh
        u    (np.array):    numeric solution
        ref_sol  (lambda):  reference solution (wrapper to be called onto (mesh, dof))
    """
    no_tot_dof = count_dof(mesh)
    u_ref = np.array([ ref_sol(mesh,dof) for dof in range(no_tot_dof)])
    err = u - u_ref
    return np.max(np.abs(err))

def calc_L2_error (mesh, u, ref_sol):
    """
    Calculate L2 error
    Args:
        mesh (mema.mesh2D): mesh
        u    (np.array):    numeric solution
        ref_sol  (lambda):  reference solution (wrapper to be called onto (mesh, dof))
    """
    no_tot_dof = count_dof(mesh)
    u_ref = np.array([ ref_sol(mesh,dof) for dof in range(no_tot_dof)])
    err = u - u_ref
    square_norm = 0

    for iel in range(len(mesh.elem2node)):
        elem = mesh.elem2node[iel]
        node_per_elem = len(elem)
        for ino in range(node_per_elem):
            uA = err[glob_idx(mesh, iel, ino)]
            uB = err[glob_idx(mesh, iel, (ino+1)%node_per_elem)]
            e = mesh.coords[mesh.elem2node[iel] [(ino+1)%node_per_elem]] \
              - mesh.coords[mesh.elem2node[iel] [ino]]

            square_norm += 0.5*np.linalg.norm(e)**2*calc_r2_prod_integ(uA,uB,uA,uB)

    return np.sqrt(square_norm)

def calc_energy_error (mesh, u, ref_sol, G, N_gamma, S):
    """
    Calculate energy error
    Args:
        mesh (mema.mesh2D): mesh
        u    (np.array):    numeric solution
        ref_sol  (lambda):  reference solution as function of mesh and dof
        G (np.array): grad.grad matrix
        N_gamma (np.array): interface jump penalization matrix
        S (np.array): stabilization matrix
    """
    no_tot_dof = count_dof(mesh)
    u_ref = np.array([ ref_sol(mesh,dof) for dof in range(no_tot_dof)])
    err = u - u_ref
    contrib_1 = np.dot(np.dot(G,err),err)
    contrib_2 = np.dot(np.dot(N_gamma,err),err)
    contrib_3 = np.dot(np.dot(S,err),err)

    return np.sqrt(contrib_1+contrib_2+contrib_3)

def pretty_visualize(mesh, u, fig, ax, cmap = 'magma'):
    """
    Visualize numeric solution

    Args:
        mesh (mema.mesh2D): mesh
        u    (np.array): numeric solution
        fig, ax: figure
        cmap: colormap
    """
    colmap = plt.get_cmap(cmap)

    ax.set_xlim(-0.5, 0.5)  # x-axis limits from 2 to 8
    ax.set_ylim(-0.5, 0.5)  # y-axis limits from -1 to 1
    ax.set_aspect('equal')

    max_u = np.max(u)
    min_u = np.min(u)

    for iel in range(len(mesh.elem2node)):
        verts = []
        nodal_values = []
        node_per_elem = len(mesh.elem2node[iel])

        # draw polygon
        for ino in range(node_per_elem):
            point = mesh.coords[mesh.elem2node[iel][ino]]
            dof = glob_idx(mesh, iel, ino)
            verts.append(point[0:2])
            nodal_values.append(u[dof])
        color = colmap((np.mean(nodal_values)-min_u)/(max_u-min_u))
        element = Polygon(verts, closed=True,\
                          facecolor=color, alpha=0.8)
        ax.add_patch(element)

    # colorbar settings
    norm = mcolors.Normalize(vmin=min_u, vmax=max_u)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('$u$', fontsize = 40)


def pretty_visualize_gradient(mesh, u, fig, ax, cmap = 'magma', random_threshold = 3):

    colmap = plt.get_cmap(cmap)

    ax.set_xlim(-0.5, 0.5)  # x-axis limits from 2 to 8
    ax.set_ylim(-0.5, 0.5)  # y-axis limits from -1 to 1
    ax.set_aspect('equal')

    min_norm_gr = 1e30
    max_norm_gr = 0

    # preliminary loop to determine extrema of gradient
    for iel in range(len(mesh.elem2node)):
        verts = []
        nodal_values = []
        node_per_elem = len(mesh.elem2node[iel])

        # collect nodal values
        for ino in range(node_per_elem):
            point = mesh.coords[mesh.elem2node[iel][ino]]
            dof = glob_idx(mesh, iel, ino)
            verts.append(point)
            nodal_values.append(u[dof])

        # get gradient and trace arrow in barycenter
        G = disc_grad(mesh, iel)
        gradient = np.dot(G, nodal_values)
        norm_gr = np.linalg.norm(gradient)
        min_norm_gr = min (norm_gr, min_norm_gr)
        max_norm_gr = max (norm_gr, max_norm_gr)

    for iel in range(len(mesh.elem2node)):
        verts = []
        nodal_values = []
        node_per_elem = len(mesh.elem2node[iel])

        # collect nodal values
        for ino in range(node_per_elem):
            point = mesh.coords[mesh.elem2node[iel][ino]]
            dof = glob_idx(mesh, iel, ino)
            verts.append(point[0:2])
            nodal_values.append(u[dof])

        # get gradient and trace arrow in barycenter
        bary = mesh.barycenter (iel)
        G = disc_grad(mesh, iel)
        gradient = np.dot(G, nodal_values)
        norm_gr = np.linalg.norm(gradient)
        color = colmap((norm_gr-min_norm_gr)/(max_norm_gr-min_norm_gr))
        element = Polygon(verts, closed=True,\
                          facecolor=color, alpha=0.8)
        ax.add_patch(element)

        k = random.randint (1, 6)
        if (k<random_threshold):
            ax.arrow(bary[0], bary[1], gradient[0]/150, gradient[1]/150, \
                 color = 'white', zorder = 5)

    # colorbar settings
    norm = mcolors.Normalize(vmin=min_norm_gr, vmax=max_norm_gr)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r'$ \nabla u $', fontsize = 40)

def solve (mesh, ref_sol, sigma_in, sigma_ex, eta, J_datum):
    """
    Solves an elliptic interface problem, given mesh and data.

    Args:
        mesh (mema.mesh2D): mesh
        ref_sol (lambda): reference solution (or boundary data)
        sigma_in (float): diffusion coefficient internal
        sigma_ex (float): diffusion coefficient external
        eta     (float): user-dependent parameter for jump conditoins "a la nitsche"
        J_datum (lambda): interface jump
    Returns:
        np.array : solution u
        np.array : global system matrix A
        np.array : global system r.h.s. b
        np.array : block matrix G (grad*grad)
        np.array : block matrix M_gamma (interface jump of gradient)
        np.array : block matrix N_gamma (interface jump)
        np.array : block r.h.s. b_j (interface jump)
    """

    # assemble blocks
    G = assemble_G(mesh, sigma_in,sigma_ex)
    S = assemble_S(mesh, sigma_in, sigma_ex)
    M_gamma = assemble_M_gamma(mesh, sigma_in, sigma_ex)
    N_gamma = assemble_N_gamma(mesh, sigma_in, sigma_ex)
    b_J = assemble_b_J (mesh, J_datum, sigma_in, sigma_ex)
    # define system, apply bc conds an solve
    A = G+S+eta*N_gamma + np.transpose(M_gamma)
    b = eta*b_J
    [A, b] = impose_bc(mesh, A, b, ref_sol)
    u = np.linalg.solve(A, b)

    return [u, A, b, G, S, M_gamma, N_gamma, b_J]

def calc_t_maxwell (mesh, intface, u, eps_in, eps_ext):
    """
    Calculates Maxwell tension from solution of DDRIN
    Args:
        mesh (mesh2D): mesh
        intface (disk_intface): interface
        u: node_wise potential
        eps_in: internal permittivity
        eps_ext: external permittivity
    """

    # Loop over cuts, get intface edges, get gradient from 2 sides, compute tensor,
    # jump and take normal component

    t_maxwell = [0]*len(intface.edges)

    # get normal
    normal = intface.calc_normal()
    hE = intface.calc_edge_length()
    normed_normal = [hE[k]*normal[k] for k in range(len(intface.edges))]

    for icut, cut in enumerate(mesh.cuts):
        iel_in, iel_ext = cut[0]
        no_edges = len(cut[2])
        first_ie_in = cut[1][0]

        # get gradient in and ext
        ## gradient matrices
        GR_in = disc_grad (mesh, iel_in)
        GR_ext = disc_grad (mesh, iel_ext)

        ## get local potential nodal values
        nodal_values_in = [u[glob_idx(mesh, iel_in, ino)] for ino in range(len(mesh.elem2node[iel_in]))]
        nodal_values_ext = [u[glob_idx(mesh, iel_ext, ino)] for ino in range(len(mesh.elem2node[iel_ext]))]

        ## get local gradient (electric field E)
        E_in  = np.dot(GR_in,  nodal_values_in)
        E_ext = np.dot(GR_ext, nodal_values_ext)

        # get maxwell tensor
        mw_tensor_in = eps_in*(np.tensordot(E_in, E_in, axes = 0)  -0.5*np.linalg.norm(E_in)*np.eye(2))
        mw_tensor_ext = eps_ext*(np.tensordot(E_ext, E_ext, axes = 0)  -0.5*np.linalg.norm(E_ext)*np.eye(2))

        for ie in range(no_edges):
            intface_edge = mesh.cuts[icut][2][ie]
            # get maxwell stress
            t_maxwell[intface_edge] = np.dot(mw_tensor_ext - mw_tensor_in, normed_normal[intface_edge])

    return t_maxwell

