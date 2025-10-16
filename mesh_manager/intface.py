from . import load_mesh as lm
from . import geometric_info as gmi
from . import dof_tools as dt
import numpy as np
from scipy.optimize import minimize
import copy
import warnings

class disk_interface:
    """

    Interface class; implemented as chain of edges

    Attributes:
        edges (list(list(int))): edge to node connectivity
        coords (list(np.array)): node coordinates
        velocity(list(np.array)): edge velocity
        k_b (float): bending modulus
        k_str (float): stretching modulus
        init_edge_length (list(float)): reference edge length
        velocity (list(np.array)): edge velocity (0 if not assigned)

    """
    def __init__(self, edges, coords, k_b, k_str, initial=True, init_edge_length=None, velocity=None):
         self.edges =edges
         self.coords = coords
         self.k_b = k_b
         self.k_str = k_str

         # if initial calculate edge length and initialize zero velocity
         if (initial):
             self.init_edge_length = self.calc_edge_length()
             self.velocity = [np.array([0., 0.])]*len(edges)
         else:
             if (init_edge_length==None):
                 raise ValueError ("When creating interface that is not initial the original edge length must be provided")
             else:
                 self.init_edge_length = init_edge_length

         if (velocity !=None):
             self.velocity = velocity
         else:
             self.velocity = [np.array([0., 0.])]*len(edges)

         # calculate edge tension
         self.edge_tension = self.calc_t_gamma()

    def calc_normal(self):
        """
        Calculate edge normal

        """
        # reinitialize
        normal = []
        for edge in self.edges:
            p0 = edge[0]
            p1 = edge[1]
            tangent = self.coords[p1] - self.coords[p0]
            clockwise_90 = np.array([[0, 1],[-1, 0]])
            normal.append(np.dot(clockwise_90, tangent)/gmi.R2_norm(tangent))

        return normal

    def calc_curvature(self):
        """
        Calculate node curvature

        Returns:
            list(float): node curvature
        """
        curvature = []

        N = len(self.edges)
        for ino in range(N):
            edge_l    = self.edges[(ino-1)%N]
            edge_r    = self.edges[ino]
            tangent_l = self.coords[edge_l[1]] - self.coords[edge_l[0]]
            tangent_r = self.coords[edge_r[1]] - self.coords[edge_r[0]]
            norm_l    = gmi.R2_norm(tangent_l)
            norm_r    = gmi.R2_norm(tangent_r)
            cos_theta = np.dot(tangent_l, tangent_r)/(norm_l*norm_r)
            curvature.append(np.arccos(cos_theta)/(0.5*(norm_l + norm_r)))

        return curvature

    def calc_edge_length(self):
        """
        Calculates length of the edges

        Returns:
            list(float): list of edge lengths
        """
        edge_length = []

        for edge in self.edges:
            tangent = self.coords[edge[1]] - self.coords[edge[0]]
            edge_length.append(gmi.R2_norm(tangent))

        return edge_length


    def apply_LB(self, data):
        """
        Applies discrete Laplace-Beltrami Operator

        Args:
            data (list): node-valued field
        Returns
            list: node_valued curve Laplacian (LB) of data
        """
        LB_data = []

        N = len(self.edges)
        for ino in range(N):
            edge_l = self.edges[(ino-1)%N]
            edge_r = self.edges[ino]
            tangent_l = self.coords[edge_l[1]] - self.coords[edge_l[0]]
            tangent_r = self.coords[edge_r[1]] - self.coords[edge_r[0]]
            norm_l = gmi.R2_norm(tangent_l)
            norm_r = gmi.R2_norm(tangent_r)
            data_l = data[edge_l[0]]
            data_0 = data[edge_l[1]]
            data_r = data[edge_r[1]]

            LB = (2/(norm_l+norm_r))*((data_l - data_0)/norm_l + (data_r - data_0)/norm_r)
            LB_data.append(LB)

        return LB_data

    def calc_nodal_forces(self):
        """
        Calculates nodal forces;
        $F = F_{bending} + F_{stretching}$

        Returns:
            list(np.array): edge_wise constant surface tension (vector)
        """
        return [F_bend + F_str for [F_bend, F_str] in \
                zip(self.calc_F_bending (), self.calc_F_stretching ())]

    def calc_t_gamma (self):
        """
        Calculates surface tension $t_Gamma$ by calculating nodal
        forces and transferring onto edges with correct scaling

        Returns:
            list(np.array): edge_wise constant surface tension (vector)
        """
        N = len(self.edges)
        F = self.calc_nodal_forces()
        hE = self.calc_edge_length()

        t_gamma = [0.5*(F[k] + F[(k+1)%N])/hE[k]  for k in range(N)]

        return t_gamma

    def calc_F_bending (self):
        """
        Calculates nodal force $F_{bending}$;

        Returns:
            list(np.array): nodal force (vector)
        """

        # calculate curvature at nodes with turning angle method
        curvature          = self.calc_curvature ()
        # apply Laplace_Beltrami operator (1D laplacian) to curvature
        LB_curvature       = self.apply_LB(curvature)
        # get list of normals at nodes (have to transfer) #XXX: not sure this is safe
        edge_normal        = self.calc_normal()
        hE                 = self.calc_edge_length()
        scaled_edge_normal = [n*L for [n,L] in zip(edge_normal, hE)]
        normal             = self.transfer_edge2node(scaled_edge_normal)
        # calculate Helfrich bending stress at nodes
        F_bnd              = [self.k_b*(1/2*curv**3 + LB_curv)*n \
                           for [curv, LB_curv, n] in zip(curvature, LB_curvature, normal)]

        return F_bnd


    def calc_F_stretching (self, drop_model = False, gamma = 1):
        """
        Calculates nodal force $F_{stretching}$;
        Hooke's elastic law

        Returns:
            list(np.array): nodal force (vector)
        """
        N              = len(self.edges)
        F_str          = [np.array([0., 0.])]*N
        init_length    = self.init_edge_length
        current_length = self.calc_edge_length()

        for ie in range(N):
            edge    = self.edges[ie]
            p0      = edge[0]
            p1      = edge[1]
            tangent = self.coords[p1] - self.coords[p0]
            tangent = tangent/gmi.R2_norm(tangent)

            if drop_model:
                tension = gamma
            else:
                tension = self.k_str*(current_length[ie]-init_length[ie])/init_length[ie]

            F_str[ie]       = F_str[ie] + tension*tangent
            F_str[(ie+1)%N] = F_str[(ie+1)%N] - tension*tangent

        return F_str

    def transfer_edge2node(self, data):
        """
        Tranfers an interface property from edges to nodes
        or viceversa applying average

        Args:
            data (list): a data defined as a list over edges or nodes
        Returns:
            list: ransferred data
        """

        data_left          = data
        data_right         = [data[(k - 1)%len(data)] for k in range(len(data))]
        transferred_data   = [ 0.5*(data_left[k] + data_right[k]) for k in range(len(data))]

        return transferred_data


    def advect(self, ref_vol, delta_t):
        """
        Advect interface using velocity

        Args:
            ref_vol (float): reference volume
            delta_t (float): time step

        Returns:
            disk_intface: moved interface
        """

        # transfer velocity at nodes
        vel_at_nodes = self.transfer_edge2node(self.velocity)

        # advance node position
        new_coords = []
        for ino in range(len(self.coords)):
            new_coords.append(self.coords[ino] + delta_t*vel_at_nodes[ino])

        coords_adjust = volume_conservation_postproc (new_coords, ref_vol)
        return disk_interface(self.edges, coords_adjust, self.k_b, self.k_str, initial=False,\
                              init_edge_length=self.init_edge_length, velocity = self.velocity)


def initialize_disk_interface(N, rho, k_b, k_str):
    """
    Create a brand-new disk interface

    Args:
        N (int): number of edges/nodes
        rho (float): radius of the disk
    """
    coords = []
    edges = []

    for ied in range(N): # numbering edges

        #create and register node (first)
        x = rho*np.cos(ied*2*np.pi/N)
        y = rho*np.sin(ied*2*np.pi/N)
        coord = np.array([x,y])
        coords.append(coord)

        # create and register edge
        if (ied < N-1):
            edge = [ied, ied+1]
        else:
            edge = [ied, 0]
        edges.append(edge)

    return disk_interface (edges, coords, k_b, k_str, initial=True)

def initialize_interface_from_stl(intface_filename, k_b, k_str):
    """
    Create the interface reading an stl file

    Args:
        intface_filename (string): path to stl file
        rho (float): radius of the disk
    """
    coords = extract_first_vertex_coordinates(intface_filename)
    N = len(coords)
    edges = [[ied, (ied+1)%N] for ied in range(N)]

    return disk_interface (edges, coords, k_b, k_str, initial=True)

def extract_first_vertex_coordinates(stl_file_path):
    """
    Extract list of coordinated from file

    Args:
        stl_file_path (string): path to file
    Returns:
        list
    """
    vertices = []
    with open(stl_file_path, 'r') as file:
        lines = file.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("outer loop"):
                # Leggi la riga successiva per il primo vertex
                vertex_line = lines[i+1].strip()
                if vertex_line.startswith("vertex"):
                    parts = vertex_line.split()
                    x = float(parts[1])
                    y = float(parts[2])
                    vertices.append(np.array([x, y]))
            i += 1

    return vertices

def calc_membrane_volume (intface):
    """
    Calculates  volume of surface

    Args:
         u (np.array): P^1(Gamma_h)
    """
    def normalize (x):
        """
        Applies operator N: P^1 (Gamma_h)-->P^0 (Gamma_h)
        calculates the normal n_Gamma given a position vector x

        Args:
            x (np.array): nodal position x
        Returns:
              (np.array): normal
        """
        rot = np.array([[0, 1],[-1, 0]])

        return [ np.dot(rot, x[(k+1)%len(x)] - x[k]) for k in range(len(x))]

    integ = 0
    x0 = intface.coords
    normal = normalize(x0)

    for k in range(len(x0)):
        x0_min  =  x0[k]
        x0_plus =  x0[(k+1)%len(x0)]
        integ += 0.5*np.dot( normal[k], 0.5*(x0_min + x0_plus))
    return integ

def shoelace_volume (coords):
    """
    Calculates volume of intface with shoelace formula
    (given its coords)
    Args:
        coords list(np.array): formatted as intface.coords
    """
    vol = 0.0
    for edge in range(len(coords)):
        p0   = coords[edge]
        p1   = coords[(edge+1)%len(coords)]
        vol += 0.5*(p0[0]*p1[1] - p1[0]*p0[1])
    return vol

def volume_conservation_postproc (vert_pos, ref_vol):
    """
    Adjust the position of vertices to restore initial volume
    of the surface. Displacement with respect to position of the
    surface after advection minimal. Solve the minimization problem:

    min_{x} (x-vert_pos)**2 with constraint: volume (vert_pos)= ref_vol

    For minimization, use Sequential Quadratic Programming (SQP),
    implemented in library scipy.optimize

    Args:
        vert_pos (list(np.array)): vertices formatted as field intface.coords
        ref_vol (float): reference volume
    Return:
        list(np.array): optimized vertices position
    """
    def flatten_coords(x0):
        """
        Auxiliary function to flatten vertices coordinates in correct format for scipy.optimize.minimize

        Args:
            x0(list(np.array))
        Returns:
            np.array
        """
        flattened_x0 = np.zeros(2*len(x0))
        for k in range(len(x0)):
            flattened_x0[2*k] = x0[k][0]
            flattened_x0[2*k+1] = x0[k][1]

        return flattened_x0

    def coord_list(flattened_x0):
        """
        Reshape vertex coordinates as in intface.coords

        Args:
            flattened_x0 (np.array)
        Returns:
            x0(list(np.array))
        """
        x0 = []
        N = int(flattened_x0.shape[0]/2)
        for k in range(N):
            x0.append(np.array([flattened_x0[2*k], flattened_x0[2*k+1]]))
        return x0

    def objective(flat_x, flat_x_star):
        """
        Objective function penalizing distance w.r.t. advected vertex position
        """
        return sum((flat_x-flat_x_star)**2)

    def flat_volume(flat_x0):
        """
        Volume calculated with flattened input
        Args:
            flat_x0 (np.array)
        """
        return shoelace_volume(coord_list(flat_x0))

    def volume_constraint(flat_x0, ref_vol):
        return flat_volume(flat_x0) - ref_vol


    flat_vert_pos = flatten_coords(vert_pos)
    obj = lambda flat_x: objective(flat_x, flat_vert_pos) # only one argument
    constraints = {'type': 'eq', 'fun': lambda flat_x: volume_constraint(flat_x, ref_vol)}
    argmin = minimize(obj, flat_vert_pos, constraints=constraints, method='SLSQP', options={'disp': True})

    return coord_list(argmin.x) # nb: argmin is an OptimizeResult object



def calc_membrane_energy (intface):
    """
    Calculates total energy of the membrane
    """
    N = len(intface.edges)
    # calculate total energy
    edge_hE_0 = intface.init_edge_length
    edge_hE = intface.calc_edge_length()
    stretching_energy = 0.5*intface.k_str*sum([(edge_hE[k]-edge_hE_0[k])**2 for k in range(N)])

    nodal_curvature = intface.calc_curvature()
    nodal_hE = intface.transfer_edge2node(edge_hE)
    bending_energy = 0.5*intface.k_b*sum([nodal_curvature[k]**2*nodal_hE[k] for k in range(N)])

    total_energy = stretching_energy + bending_energy
    return total_energy

def calc_membrane_power (intface):
    """
    Calculates power acting on the membrane
    """
    N = len(intface.edges)
    nodal_velocity = intface.transfer_edge2node(intface.velocity)
    nodal_force = intface.calc_nodal_forces()
    power = sum ([np.dot(nodal_velocity[k], nodal_force[k]) for k in range(N)])

    return power

def calc_time_step (intface, eta):
    """
    Calculates time step such that energy increase is controlled
    Args:
        intface (mema.disk_intface): interface
        eta (float): user-dependent parameter
    """

    total_energy = calc_membrane_energy (intface)
    power = calc_membrane_power (intface)

    if (total_energy>0):
        return eta*total_energy/abs(power)
    else:
        return 1e6 # when taking min with tau_default this is not taken

def visualize_intface(fig, ax, intface, color = 'red', show_velocity=False, show_tension=False,\
                      show_nodes = False, mark_first_node = False, show_edge_idx = False):
    """
    Plots interface

    Args:
        fig: figure
        ax:  axis
        intface (mema.disk_interface): interface to plot
        show_velocity(boolean): whether to show arrows for velocity
        show_tension (boolean): whether to show arrows for tension
    """
    no_intf_edges = len(intface.edges)
    intface.edge_tension = intface.calc_t_gamma()
    for ied in range(no_intf_edges):
        p1 = intface.coords[intface.edges[ied][0]][:]
        p2 = intface.coords[intface.edges[ied][1]][:]
        midpoint = 0.5*(p1 + p2)
        dx = p2-p1
        xx = [p1[0],p2[0]]
        yy = [p1[1],p2[1]]
        v = intface.velocity[ied]
        ax.plot(xx,yy, color, markersize = 4, linewidth=2)
        if (show_velocity):
            ax.arrow(p1[0],p1[1], v[0],v[1],head_width=0.005)
        if (show_tension):
            t_gamma = intface.edge_tension[ied]
            ax.arrow(midpoint[0], midpoint[1], t_gamma[0], t_gamma[1], head_width=0.005)
        if (show_nodes):
            p = intface.coords[ied]
            ax.scatter(p[0], p[1], s = 50, color = 'black')
        if (show_edge_idx):
            text = str(ied)
            pos = 0.5*(p1+p2)
            ax.text(pos[0], pos[1], text, fontsize= 10, clip_on=True)

    if (mark_first_node):
        p = intface.coords[0]
        ax.scatter(p[0], p[1], s = 60, color = 'orange')

def visualize_intface_vector(fig, ax, intface, vector_data):
    """
    Plots interface

    Args:
        fig: figure
        ax:  axis
        intface (mema.disk_interface): interface
        vector_data: edge-wise data to plot
    """
    no_intf_edges = len(intface.edges)
    for ied in range(no_intf_edges):
        p1 = intface.coords[intface.edges[ied][0]][:]
        p2 = intface.coords[intface.edges[ied][1]][:]
        midpoint = 0.5*(p1 + p2)
        dx = p2-p1
        xx = [p1[0],p2[0]]
        yy = [p1[1],p2[1]]
        ax.plot(xx,yy,'r', markersize = 4, linewidth=3)
        ax.arrow(midpoint[0],midpoint[1], vector_data[ied][0],vector_data[ied][1],head_width=0.005)

def visualize_intface_nodal_vector(fig, ax, intface, vector_data):
    """
    Plots interface

    Args:
        fig: figure
        ax:  axis
        intface (mema.disk_interface): interface
        vector_data: edge-wise data to plot
    """
    no_intf_edges = len(intface.edges)
    for ied in range(no_intf_edges):
        p1 = intface.coords[intface.edges[ied][0]][:]
        p2 = intface.coords[intface.edges[ied][1]][:]
        xx = [p1[0],p2[0]]
        yy = [p1[1],p2[1]]
        ax.plot(xx,yy,'r', markersize = 4, linewidth=3)
        ax.arrow(p1[0], p1[1], vector_data[ied][0], vector_data[ied][1], head_width=0.005)

