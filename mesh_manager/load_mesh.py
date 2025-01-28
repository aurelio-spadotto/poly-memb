import meshio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as Polygon
from . import geometric_info as gmi

class mesh2D:
    """
    Mesh Class. Supports broken elements.
    It features both elem2edge and elem2node connectivity.
    It incorporates information about the phase and interface edges.

    Attributes:

    elem2node   (list(list(int))): element 2 node connectivity
    elem2edge  (list(list(int))): element 2 edge connectivity
    no_edges (int): number of edges
    edge2elem  (list(list(int))): edge 2 element connectivity
    coords (list(np.array)): node coordinates, 2 components
    side_mask (list): element mask to define phase (0 int, 1 ext)
    node_bnd_mask  (list(int)): node mask to define if on border and what border (0 if internal, k if on boundary k)
    edge_bnd_mask  (list(int)): edge mask to define if on border
    cuts (list): for each cut a list in the form: [couple, starting_ie, edge2intface]
                                                   couple: elements along the cut [in, ex]
                                                   starting_ie: local idx of first edge along cut[in, ex]
                                                   edge2intface: map from edge on cut to idx on intface
    intface_edges       (list): list of interface edges (global index)
    d                  (float): semi-side of box (necessary for marking boundary nodes)

    How to loop over interface edges:
    loop over cuts, and run over element edges using index of first and len(edge2intface)
    """
    def __init__(self, elem2node, coords, bnd_dof_type, d):
        """
        Intializes the class mesh2D
        """
        self.elem2node  = elem2node
        self.coords = coords

        # attributes linked to mesh cutting
        self.side_mask = [-1]*len(self.elem2node)
        self.cuts = []
        self.intface_edges = []

        # generate elem2edge connectivity
        self.elem2edge, self.no_edges = self.generate_elem2edge()
        self.edge2elem = []

        # attributes linked to boundary
        self.node_bnd_mask = [-1]*len(self.coords)
        self.edge_bnd_mask = [-1]*self.no_edges


        # set boundary mask
        self.d = d
        if (bnd_dof_type=="node"):
            self.mark_bnd_points()
        if (bnd_dof_type=="edge"):
            self.mark_bnd_edges()

        # Pre-calculate metrics
        self.edge_normal = []
        self.edge_xE = [np.array([0., 0.]) for k in range(self.no_edges)]
        self.edge_length = [0 for k in range(self.no_edges)]
        self.element_surface = []
        self.element_barycenter = []

        self.no_elems = len(self.elem2node)

        for iel in range(self.no_elems):
            self.element_surface.append(self.calc_surface(iel))
            self.element_barycenter.append(self.barycenter(iel))
            no_edge = len(self.elem2edge[iel])
            element_edge_normal = []
            for ie in range(no_edge):
                glob_ie = self.elem2edge[iel][ie]
                self.edge_length[glob_ie] = self.get_edge_length(iel, ie)
                self.edge_xE[glob_ie] = self.get_xE(iel,ie)
                element_edge_normal.append(self.get_edge_normal(iel, ie))
            self.edge_normal.append(element_edge_normal)



    def generate_intface_edges(self):
        """
        Creates self.intface_edges (list of global indexes of interface edges)
        Edges are not in ascending ordered,
        position is correlated to order on interface (not perfectly though)

        Parameters:
        - self (self2D): self

        Returns:
        - list: list to replace as self.intface_edges
        """

        intface_edges_list = []

        intface_couples = [self.cuts[k][0] for k in range(len(self.cuts))]
        first_ie = [self.cuts[k][1] for k in range(len(self.cuts))]
        mesh2intface= [self.cuts[k][2] for k in range(len(self.cuts))]

        for icut in range(len(self.cuts)):
            intface_edges = []
            iel_in = intface_couples[icut][0]
            first_ie_in = first_ie[icut][0]
            no_edges = len(mesh2intface[icut])
            for ie in range(no_edges):
                intface_edges.append(self.elem2edge[iel_in][(first_ie_in + ie) % len(self.elem2node[iel_in])])

            intface_edges_list.append(intface_edges)

        return intface_edges_list

    def generate_elem2edge(self):
        """
        Creates the local to global map for edges self.elem2edge

        Parameters:
        - self (self2D): self

        Returns:
        - list : elem2edge connectivity
        """
        elem2edge = []
        edge_dict = {} # edge_dictionary: key is pair of nodes, value is index
        max_edge_idx = 0
        intface_edges = []
        for iel in range(len(self.elem2node)):
            # initialize local list
            edges = []
            edge_per_elem = len(self.elem2node[iel])
            # loop over edge: if already there retrieve index, otherwise add
            for ie in range(edge_per_elem):
                ino1= self.elem2node[iel][ie]
                ino2= self.elem2node[iel][(ie+1)%edge_per_elem]
                edge = tuple(sorted((ino1, ino2)))
                # search if edge is already there by searching the dicitionary
                # if not there add it to dictionary
                if (edge in edge_dict):
                    edge_idx = edge_dict[edge]
                else:
                    edge_idx = max_edge_idx
                    edge_dict[edge] = max_edge_idx
                    max_edge_idx += 1
                edges.append(edge_idx)
            elem2edge.append(edges)

        return [elem2edge, max_edge_idx]


    def generate_edge2elem(self):
        """
        Calculates edge 2 element connectivity
        """
        no_total_edges = self.no_edges
        edge2elem = [[] for k in range(no_total_edges)]

        no_elems = len(self.elem2edge)
        for iel in range(no_elems):
            elem2edge = self.elem2edge[iel]
            no_edges = len(elem2edge)
            for ie in range(no_edges):
                edge2elem[elem2edge[ie]].append(iel)

        return edge2elem

    def generate_elem2elem(self):
        """
        Generates elem2elem connectivity (elements sharing one edge standing on same side)
        """
        no_elems = len(self.elem2node)
        elem2elem = [[] for k in range(no_elems)]
        edge2elem = self.edge2elem

        if (self.edge2elem == []):
            self.edge2elem = self.generate_edge2elem()
        edge2elem = self.edge2elem
        no_edges = len(edge2elem)

        for ie in range(no_edges):
            elements = edge2elem[ie]
            if len(elements)==2: # otherwise it is a boundary edge
                iel1, iel2 = elements
                if (self.side_mask[iel1]+self.side_mask[iel2]!=1):
                    elem2elem[iel1].append(iel2)
                    elem2elem[iel2].append(iel1)

        return elem2elem


    def mark_bnd_points(self):
       """
       Assigns the boundary mask for points;
       It only works for square boxes

       Convention:
      -1: internal
       1: x=-d
       2: y=-d
       3: x= d
       4: y= d
       """
       d = self.d
       #XXX should change load_mesh to directly recognize nodes on boundary when reading,mark_bnd_edges same problem
       for ino in range(len(self.coords)):
            if (abs(self.coords[ino][0]+d)<1e-6):
                self.node_bnd_mask[ino]= 1
            if (abs(self.coords[ino][1]+d)<1e-6):
                self.node_bnd_mask[ino] = 2
            if (abs(self.coords[ino][0]-d)<1e-6):
                self.node_bnd_mask[ino] = 3
            if (abs(self.coords[ino][1]-d)<1e-6):
                self.node_bnd_mask[ino] = 4

    def mark_bnd_edges(self):
        """
        Sets mask for edges lying on boundary (no distinction between sections of boundary)

        Returns: void
        """
        d = self.d
        for iel in range(len(self.elem2node)):
            no_edges = len(self.elem2edge[iel])
            for ie in range(no_edges):
                x_E = self.get_xE (iel, ie)
                if ( abs(abs(x_E[0]) -d)<1e-6 or abs(abs(x_E[1]) -d)<1e-6):
                    glob_ie = self.elem2edge[iel][ie]
                    self.edge_bnd_mask[glob_ie]= 1


    def calc_surface(self,iel):
        #shoelace formula
        mod_F = 0
        node_per_elem = len(self.elem2node[iel][:])
        for ino in range(node_per_elem):
            x1 = self.coords[self.elem2node[iel][ino]][0]
            y1 = self.coords[self.elem2node[iel][ino]][1]
            x2 = self.coords[self.elem2node[iel][(ino+1)%node_per_elem]][0]
            y2 = self.coords[self.elem2node[iel][(ino+1)%node_per_elem]][1]
            mod_F = mod_F + x1*y2 - x2*y1
        mod_F = 0.5*mod_F
        return mod_F


    def barycenter (self,iel):
        """
        Determines barycenter of polygonal element
        (using shoelace formula)

        Args:
            iel (int): element index
        Returns:
            np.array(float): barycenter
        """

        A = 0  # Signed Area
        C_x = 0 # centroid x coord
        C_y = 0 # centroid y coord

        vertices = [self.coords[ino] for ino in self.elem2node[iel]]

        for i in range(len(vertices)):
            x_i, y_i = vertices[i]
            x_next, y_next = vertices[(i + 1)%len(vertices)]

            common_term = x_i * y_next - x_next * y_i
            A += common_term
            C_x += (x_i + x_next) * common_term
            C_y += (y_i + y_next) * common_term

        A *= 0.5
        C_x /= (6 * A)
        C_y /= (6 * A)

        return np.array([C_x, C_y])


    def get_xE (self, iel, ie):
        """
        Args:
            iel         (int): element index
            ie          (int): index of edge

        Returns:
            float            : barycenter of edge
        """
        no_edges = len(self.elem2edge[iel])
        p1 = self.coords[self.elem2node[iel][ ie]]
        p2 = self.coords[self.elem2node[iel][ (ie+1)%no_edges]]

        return 0.5*(p1+p2)


    def get_edge_length(self, iel, ie):
        """
        Args:
            iel         (int): element index
            ie    (int): index of face

        Returns:
            float: length of the edge ie

        Raises:
        ValueError: If the result is 0.
        """
        no_edges = len(self.elem2edge[iel])
        p1 = self.coords[self.elem2node[iel][ ie]]
        p2 = self.coords[self.elem2node[iel][ (ie +1)%no_edges]]
        hE = gmi.R2_norm(p2-p1)
        if (hE==0):
            raise ValueError("Issue with element ", iel, "at edge ", ie,": Edge size cannot be 0")

        return gmi.R2_norm(p2-p1)


    def get_edge_normal(self, iel, num_face):
        """
        Args:
            self (ddr.self2D): self
            iel         (int): element index
            num_face    (int): index of face

        Returns:
            np.array(float): edge normal
        """
        no_edges = len(self.elem2edge[iel])
        p1 = self.coords[self.elem2node[iel][ num_face]]
        p2 = self.coords[self.elem2node[iel][(num_face +1)%no_edges]]
        rotation = np.array([[0,1],[-1,0]])

        return np.dot(rotation, p2-p1)

    def get_mesh_size (self):
        """
        Provides minimal and maximal element size
        (taken as sqr of surface)
        """

        min_hT = 1e6
        max_hT = 0.0

        for iel in range(len(self.elem2node)):
            hT = np.sqrt(self.calc_surface(iel))
            if (hT<min_hT):
                min_hT = hT
            if (hT>max_hT):
                max_hT = hT

        return [min_hT, max_hT]

def load_square_mesh(filename, bnd_dof_type="edge"):
    """
    Loads and parses a 2D mesh in msh format

    Parameters:
    - filename: path to the mesh file
    - no_points_per_elem = increase size of elem2edge columns to fit polygonal elements
    - bnd_dof_type (logical): type of dofs on the boundary

    Returns:
    - Mesh object
    """
    # LOAD MESH:
    mesh_file_path = filename
    mesh = meshio.read(mesh_file_path)
    # elem2node connectivity (convert to list)
    elem2node = mesh.cells[0].data
    elem2node = elem2node.astype(int)
    elem2node = [list(elem) for elem in elem2node]
    # node coordinates (convert to list, but a list of np.arrays)
    coords = mesh.points [:,0:2]
    coords = list(coords)
    # determine semiside
    d = max([max(abs(point)) for point in coords])

    mesh = mesh2D (elem2node, coords, bnd_dof_type, d)

    return mesh


def move_critical_points(mesh, rho):
    """
    Displace critical points which are too close to the interface
    (avoid rare but nasty situations of elements not properly cut)
    """
    # get mesh_size
    p1 = mesh.elem2node[0][0]
    p2 = mesh.elem2node[0][1]
    h = gmi.R2_norm(mesh.coords[p1]-mesh.coords[p2])
    # loop over points
    for ino in range(mesh.no_points):
        x = mesh.coords[ino][0]
        y = mesh.coords[ino][1]
        radius = np.sqrt(x**2+y**2)
        cos_theta = x/radius
        sin_theta = y/radius
        # If node is close to interface less than 1/10*mesh_size
        if (np.abs(rho-radius)<0.1*h):
            # internal nodes towards interior
            if (radius<rho):
                new_radius = radius - 0.1*h
            if (radius>rho):
                new_radius = radius + 0.1*h
            new_x = new_radius*cos_theta
            new_y = new_radius*sin_theta
            mesh.coords[ino][0] = new_x
            mesh.coords[ino][1] = new_y

def display_element (mesh, elem, ax, color, width = 1e-4, display_nodes= True):
    """
    Display single element
    """
    ax.set_aspect('equal')
    no_nodes = len(elem)
    for ino in range(no_nodes):
        p1 = mesh.coords[elem[ino]]
        p2 = mesh.coords[elem[(ino+1)%no_nodes]]
        xE = 0.5*(p1 + p2)
        if (display_nodes):
            ax.text(p1[0], p1 [1], str(elem[ino])+","+str(ino), fontsize = 20, color=color, clip_on = True)
        #nTE = mesh.get_edge_normal(iel, ie)
        ax.arrow(p1[0], p1[1], -p1[0] + p2[0], -p1[1] + p2[1], width = width, color = color)
        #ax.arrow(xE[0], xE[1], nTE[0], nTE[1], width = 1)

def visualize_mesh(mesh, ax, display_no_nodes = False,\
                             display_elem_id = False,
                             display_side = True,\
                             display_couples = False,\
                             cmap='magma', \
                             display_node_id= False,\
                             show_barycenter= False):
    """
    Display mesh and related info

    Args:
        display_no_nodes (boolean): whther to show number of nodes of each element
        display_side     (boolean): whether to display color of side
        display_node_id  (boolean): whether to show index of nodes
        show_barycenter  (boolean): whether to show element barycenter
    """

    colmap = plt.get_cmap(cmap)
    colors = ['red', 'blue', 'grey']

    if (display_side and display_couples):
        raise ValueError("cannot show both side and couples")

    for iel, elem in enumerate(mesh.elem2node):
        verts = [mesh.coords[ino] for ino in elem]

        # draw polygon
        if (display_node_id):
            for ino,point in enumerate(verts):
                text_pos = point
                text= str(elem[ino])
                ax.text(text_pos[0], text_pos[1], text, fontsize=13, color='black', clip_on=True)

        if (display_no_nodes):
            node_per_elem = len(mesh.elem2node[iel])
            element = Polygon(verts, closed=True, edgecolor='black',\
                              facecolor=colmap(1-node_per_elem/20), alpha=0.8)
            ax.add_patch(element)
            # write no of vertices
            bary = mesh.barycenter(iel)
            text_pos = bary
            text= str(node_per_elem)
            ax.text(text_pos[0], text_pos[1], text, fontsize=13, color='black', clip_on=True)


        elif (display_side):
            element = Polygon(verts, closed=True, edgecolor='black',\
                              facecolor=colors[int(mesh.side_mask[iel])], alpha=0.2)
            ax.add_patch(element)

        if (display_elem_id):
            # write index of elem
            bary = mesh.barycenter(iel)
            text_pos = bary
            text= str(iel)
            ax.text(text_pos[0], text_pos[1], text, fontsize=13, color='black', clip_on=True)


        if (show_barycenter):
            xT = mesh.barycenter(iel)
            ax.scatter(xT[0], xT[1], c = 'black')

    if (display_couples):
        gamma_couples = [mesh.cuts[k][0] for k in range(len(mesh.cuts))]
        for iel_in, iel_ex in gamma_couples:
            verts_in = [mesh.coords[k] for k in mesh.elem2node[iel_in]]
            verts_ex = [mesh.coords[k] for k in mesh.elem2node[iel_ex]]
            element_in = Polygon(verts_in, closed=True, edgecolor='black',\
                          facecolor='red', alpha=0.2)
            element_ex = Polygon(verts_ex, closed=True, edgecolor='black',\
                          facecolor='blue', alpha=0.2)
            ax.add_patch(element_in)
            ax.add_patch(element_ex)

def visualize_mesh_element_data (mesh, data, ax, binary=True, cmap='magma' ):
    """
    Display mesh and related info

    Args:
        data (list(float)): element-wise scalar data
        binary (boolean): wether the data assumes only two values
    """

    colmap = plt.get_cmap(cmap)
    colors = ['red', 'blue']

    for iel, elem in enumerate(mesh.elem2node):
        verts = [mesh.coords[ino] for ino in elem]

        # if binary only use red and blue, otherwise use colormap
        if (binary):
            element = Polygon(verts, closed=True, edgecolor='black',\
                              facecolor=colors[data[iel]], alpha=0.2)
        else:
            min_data = min(data)
            max_data = max(data)
            element = Polygon(verts, closed=True, edgecolor='black',\
                              facecolor=cmap((data[iel]-min_data)/max(max_data-min_data, 1e-30)), alpha=0.2)
        ax.add_patch(element)
