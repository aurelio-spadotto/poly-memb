from . import load_mesh as lm
from . import geometric_info as gmi
from . import dof_tools as dt
import numpy as np
import copy
import warnings
import sys

def break_mesh(mesh, intface, execute_fill_side_mask = True, verbose = False): # NEW VERSION
    """
    Breaks mesh along interface
    Remark: treat also case of element with multiple cuts

    Args:
        mesh (mesh2D): mesh to break
        intface (disk_interface): interface along which to cut
    Returns:
       mesh2D: new broken mesh, with list of gamma_couples, mesh2intface
               and side mask
    """

    if (verbose):
        print ("RUNNING MESH_BREAK")

    # a list of points, which can only grow during the procedure
    # also a temporary list of elements that can only grow
    # elements thane are identified by an idx which cannot change
    # (elimination of destroyed elements happens only at the end)
    coords = copy.deepcopy(mesh.coords)
    no_initial_points = len(coords)
    elem2node = copy.deepcopy(mesh.elem2node)
    # a mask to keep track of cut elements (to be taken out at the end)
    # (0 if uncut, 1 if cut)
    cut_elems = [0]*len(elem2node)
    # for each cut I register:
    # > corresponding list of interface edges,
    # > list of points on the interface
    # > couple of elements associted to cut
    # > (we have to refresh it at the end as elements resulting from cuts can be cut again)

    cuts = []

    # obsolete, but kept for legacy
    mesh2intface = []
    points_on_cut = []
    gamma_couples = []

    # find first edge of intface intersecting an element (coming in)
    if (verbose):
        print (">> in find_first_intersecting_edge")
    [ied_initial, iel_to_cut, initial_enter_point] = find_first_intersecting_edge(mesh, intface)

    # STEP 1
    # loop along the interface to break elements;
    # cut elements and for each cut get the edge coming out;
    # stop when the exit intersection point coincides with
    # the initial enter point
    ied_enter = ied_initial
    ied_exit = -1
    exit_point = np.array([.0, .0])
    if (verbose):
        print (">> in loop to break elements")
        print (">> initial edge = ", ied_initial)
    while (gmi.R2_norm(exit_point - initial_enter_point) > 1e-10):
        # get element to cut as list of points
        elem_to_cut = elem2node[iel_to_cut]
        # cut element
        if (verbose):
            print ("")
            print (">>> in cut elem")
        [coords, new_points, intface_edge_indexes, child_elem_in, child_elem_ex, ied_exit] \
            = cut_elem (coords, elem_to_cut, intface, ied_enter,\
                        initial_enter_point, idx_initial_enter_point = no_initial_points, verbose=verbose)
        # get exit point
        exit_point = coords[new_points[-1]]
        # append cut data
        mesh2intface.append(intface_edge_indexes)
        points_on_cut.append(new_points)
        # append new elements, mark old element as cut, add couples to gamma_couples
        elem2node.append(child_elem_in)
        elem2node.append(child_elem_ex)
        #gamma_couples.append([len(elem2node)-2, len(elem2node)-1])
        cut_elems = cut_elems + [0, 0]  # generated elements are not cut
        cut_elems[iel_to_cut] = 1       # element being cut is marked

        # Compose and append cut data_structure
        # by construction first_ie is [0, 0], but it may change when doing agglomeration
        cut = [[len(elem2node)-2, len(elem2node)-1], [0, 0], intface_edge_indexes]
        cuts.append(cut)

        # find next element to cut (also elements generated previously can be cut again)
        if (verbose):
            print (">>> in find_next_element")
            print (">>> new total number of elements: ", len(elem2node))
        ied_enter = ied_exit
        intsec_points = [new_points[0], new_points[-1]]
        iel_to_cut = find_next_element(elem2node, coords, cut_elems, intface, ied_enter, iel_to_cut, intsec_points)

        if (verbose):
            print (">>> ied_exit = ", ied_exit,"; iel_to_cut = ", iel_to_cut)

    # STEP 2
    # generate a refreshed cuts;
    # start scrolling cuts
    # if an element is cut start looking for the active child that shares
    # the same points on the same cut
    # replace it in the couple
    if (verbose):
        print (">> in section to refresh cuts")
    refreshed_cuts = cuts
    for icut in range(len(cuts)):
        points_in_common = points_on_cut [icut]
        for pos, iel in enumerate(cuts[icut][0]): # cuts[icut][0]: couple [elem_in, elem_ex]
            if (cut_elems[iel]==1):
                if (verbose):
                    print (">>> cut element: ", iel, " is cut again")
                found = False
                elem_to_replace = elem2node[iel]
                next_icut = icut + 1
                while ( next_icut < len(cuts) and not found):
                    for next_pos, next_iel in enumerate(cuts[next_icut][0]):
                        possible_replacement = elem2node[next_iel]
                        # check if possible replacement has nodes of the cut
                        if (set(points_in_common).issubset(set(possible_replacement)) and cut_elems[next_iel]==0):
                            refreshed_cuts[icut][0][pos] = next_iel
                            if (verbose):
                                print (">>> and replacing element is found")
                            found = True
                        else:
                            if (verbose):
                                print (">>> but replacing element not found")
                    next_icut = next_icut + 1

    # STEP 3
    # trim elem2node eliminating cut elements and shift element indexes in refreshed_gamma_couples
    # index must be shifted counting how many preceding element have been cut
    if (verbose):
        print (">> in section to trim generated mesh")
    trimmed_elem2node = [elem2node[iel] for iel in range(len(elem2node)) if cut_elems[iel]==0]
    for icut in range(len(refreshed_cuts)):
        for pos, iel in enumerate(refreshed_cuts[icut][0]):
            refreshed_cuts[icut][0][pos] = iel - sum(cut_elems[0:iel])


    # initialize new mesh from new list of points and elem2node
    new_mesh = lm.mesh2D (trimmed_elem2node, coords, bnd_dof_type = "edge", d = mesh.d)

    # complete with interface connectivities
    new_mesh.cuts = refreshed_cuts

    # generate intface_edges
    if (verbose):
        print (">> in generate_intface_edges")
    new_mesh.intface_edges = new_mesh.generate_intface_edges()

    # set side mask
    if (verbose):
       print (">> in fill_side_mask")
    if execute_fill_side_mask:
        fill_side_mask(new_mesh)
    else:
        fill_side_mask(new_mesh, only_initialize = True)

    return new_mesh

def find_first_intersecting_edge(mesh, intface):
    """
    Find the first interface edge crossing a mesh edge
    and getting inside the element

    Args:
        mesh (2D_mesh): mesh
        intface (disk_intface): intface
    Returns:
        int: edge index
    """
    for ied in range(len(intface.edges)):

        # get vertices of the edge
        p0 = intface.coords[intface.edges[ied][0]]
        p1 = intface.coords[intface.edges[ied][1]]

        # find the element containing the first node
        for iel, elem in enumerate(mesh.elem2node):
            polygon = [mesh.coords[k] for k in elem]
            if (not point_is_inside (polygon, p0)):
                if( point_is_inside(polygon, p1)):
                    # get intersection and mark it as first intersection
                    for ie in range(len(elem)):
                       q0 = mesh.coords[elem[ie]]
                       q1 = mesh.coords[elem[(ie+1)%len(elem)]]
                       [found, enter_point] = gmi.calc_intersection_segments(q0, q1, p0, p1)
                       if (found):
                           return [ied, iel, enter_point]

def point_is_inside (polygon, q):
    """
    Determine if point is inside a polygon
    Args:
        polygon (list(np.array)): polygon as list of points
        q (np.array): point to check
    Returns:
        boolean
    """
    # decompose into triangles and loop over triangles
    # to llok for the point
    p1 = polygon[0]
    for ino in range(1, len(polygon)-1):
        p2 = polygon[ino]
        p3 = polygon[ino + 1]

        if (gmi.check_if_in_triangle(p1, p2, p3, q)):
            return True

    return False

def share_an_edge(elem2node, iel1, iel2):
    """
    Determine whether two elemnts share an edge

    Args:
        elem2node (list(list(int))): element2node connectivity
        iel1 (int): first elem index
        iel2 (int): second elem index

    Returns:
        boolean
    """

    elem1 = elem2node[iel1]
    elem2 = elem2node[iel2]
    intersection = [p1 for p1 in elem1 if p1 in elem2]
    if (len(intersection)>1):
        return True

    return False

def find_next_element(elem2node, coords, cut_elems, intface, ied, cut_iel, intsec_points):
    """
    Find what is the next element cut by the intface:
    it is entered by the edge, it is adjacent to the
    cut element and it is activated

    Args:
        elem2node (list(list(int))): elem2node connectivity of a partially cut mesh
        coords    (list(np.array)): list of points
        intface   (disk_interface): interface
        ied                  (int): idx of edge to check
        cut_iel              (int): last cut element
        intsec_points  (list(int)): list of points intersections with cut elem

    Returns:
        iel_to_cut (int): index of entered element (which is next to be cut)
    """
    # loop over elements and find the next one to be intersected
    # should be ok, even if in elem2node there are already the new elements that are cut

    for iel in range(len(elem2node)):
        if (cut_elems[iel]==0):
            elem = elem2node[iel]
            [intersections, cut_elem_edges] = get_intersections(coords, elem, intface, ied)
            if (len(intersections)>0 and iel in adjacent_elements(elem2node, cut_iel, intsec_points)):
                return iel

def adjacent_elements(elem2node, cut_iel, intsec_points):
    """
    List of adjacent elements (indexes)

    Args:
        elem2node (list(list(in))): elem2node connectivity
        cut_iel              (int): element index
    Returns:
        list(int): adjacent elements
    """
    elements = []
    elem = elem2node[cut_iel] + intsec_points # ugly way to avoid exception when edge is cut on same edge twice
    for iel, elem_to_check in enumerate(elem2node):
        if (len(set(elem_to_check).intersection(elem))>1):
            elements.append(iel)

    return elements

def cut_elem (coords, elem, intface, ied, initial_enter_point,\
              idx_initial_enter_point,verbose):
    """
    Function to cut an element, and enrich a list of mesh points

    Args:
        coords (list(np.array)): list of points
        elem (list(int)): list of node idxs of the element
        intface (disk_interface): interface
        ied (int): entering intface edge
        initial_enter_point (np.array): coords of initial point of break mesh
        idx_initial_enter_points (int): its index

    Returns:
        list (np.array): enriched list points
        list (int): list of intface edges of the cut
        list(int): first child element
        list(nt): second child element
        int: last edge of the cut (and first of the next)
    """

    # initialize list of edges of the cut
    # and list of nodes f the cut
    intface_edge_indexes = [] # one elemnt for each edge on the cut
                              # there is no 1to1 corresp. of intface edges and
                              # edges of the cut elems: consider edges that intersect
    points_on_cut = []        # list of points on the cut (coordinates):
                              # the first is already there (check if it is the case also for first cut element)
                              # the last is already there only if it is the first point added by
                              # the cut algorithm (initial_eneter_point)

    new_coords = copy.deepcopy(coords) # udated list of coords to return

    # 3 STEPS:

    # STEP 1: get coordinates of points along cut
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # first determine intersection(s) of piercing edge with element
    # if edge is long: it gets in, it gets out: 2 intersection
    # otherwise: a chain of edges make up the cut

    # old line maybe useless: polygon = [coords[k] for k in elem]

    if (verbose):
        print (">>>> in get_intersection")
    [intersections, cut_elem_edges] = get_intersections(new_coords, elem, intface, ied) # returns intsec coords and
                                                                                    # idx of elemnt edge that is cut

    no_intsecs = len(intersections) # no of intersections

    # compose cut, and get edges that are cut on the original element
    # consider 2 cases: 2 intsecs ---> ied_enter=ied_exit and cut is only one edge
    #                   1 intsec  ---> cut is a chain of edges
    if (verbose):
        print (">>>> in section to determine chain of new points")
    if (no_intsecs==2):
        # points on cut are the only two intersections
        # ied_enter = ied_exit
        points_on_cut = intersections
        intface_edge_indexes.append(ied)
        cut_edges = cut_elem_edges
        ied_exit = ied
    else:
        # fill the chain (loop)
        cut_edges = [] # edges of the element to cut
        points_on_cut.append(intersections[0])
        cut_edges.append(cut_elem_edges[0])
        intface_edge_indexes.append(ied)

        done = False
        current_ied = (ied + 1)%len(intface.edges) # the loop starts from the next edge:
                                                   # either it is inisde the elem or it goes out
                                                   # in which case end the loop

        if (verbose):
             print (">>>> in loop to fill a chain")

        while (not done):

            point_to_add = intface.coords[intface.edges[current_ied][0]]
            points_on_cut.append(point_to_add)

            intface_edge_indexes.append(current_ied)

            # check if current edge intersects
            [intersections,cut_elem_edges] = \
                 get_intersections(new_coords, elem, intface, current_ied)
            if (len(intersections)>0):
                # add second intersection, cut edge, assign ied_exit and finish
                cut_edges.append(cut_elem_edges[0])
                points_on_cut.append(intersections[0])
                ied_exit = current_ied
                done = True

            # advance loop
            current_ied = (current_ied + 1)%len(intface.edges)

    if (verbose):
        print (">>>> in section to add new points to list of all points")

    # STEP 2: add new points to list of points of output mesh and get their global indexes
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    indexes = [] # indexes of new point in the expanded node list

    # you have to retrieve the index of the first (the first intersection) that
    # is supposed to be already there if it is not the first cut element
    # the last is already there only if it is the last cut
    if ((gmi.R2_norm(points_on_cut[0]-new_coords[-1])>1e-10)):
        # point is not there: this is the first element cut
        if (verbose):
            print ("This is the first element cut: first intersection still has to be added")
        new_coords.append(points_on_cut[0]) # point is new, increase coord list
    # in both cases now it is the last point of the expanded new_coords
    index_first_point = len(new_coords) - 1      # and get its index
    # add index of first point to list of indexes of points on cut
    indexes.append(index_first_point)

    # if there are internal points add them (without checking if already there because it cannot be)
    if (no_intsecs<2):
        [new_coords, new_points] = no_check_and_add_new_points (new_coords, points_on_cut[1:-1]) # between first and last
        indexes += new_points

    # Check if last point is initial_enter_point,
    # if not, just add it, otherwise, retrieve its index and signal that
    # it is the last cut
    if (gmi.R2_norm(points_on_cut[-1]- initial_enter_point) <1e-10):
        if (verbose):
            print ("This is the last element cut")
        index_last_point = idx_initial_enter_point # only retrieve index
    else:
        new_coords.append(points_on_cut[-1])
        index_last_point = len(new_coords) - 1
    # add index (retrieved or added) to list of indexes of of points on cut
    indexes.append(index_last_point)


    # STEP 3: compose children elements
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # first child is the internal one, second the external
    # by convention, element nodes are listed counter-clockwise
    # and first nodes are on the interface
    if (verbose):
        print (">>>> in section to create children elements")
        print (">>>> size of original elem: ", len(elem), " first cut edge: ", cut_edges[0]," second cut edge: ", cut_edges[1])
    points_in = [] # edges from first cut edge to second
    points_ex = [] # edges from second to first

    # consider separately the case where same edge is cut twice
    if (cut_edges[0]!=cut_edges[1]):
        ie = (cut_edges[0] + 1)%len(elem)
        while (ie != (cut_edges[1]+1)%len(elem)):
            points_ex.append(elem[ie])
            ie = (ie + 1)%len(elem)

        ie = (cut_edges[1] + 1)%len(elem)
        while (ie != (cut_edges[0]+1)%len(elem)):
            points_in.append(elem[ie])
            ie = (ie + 1)%len(elem)
    else:
        points_in = []
        points_ex = [elem[(cut_edges[1]+1+k)%len(elem)] for k in range(len(elem))]

    if (verbose):
        print (">>>> list of nodes (intf, internal, external):")
        print (">>>>> new_points: ", indexes)
        print (">>>>> old points child in: ", points_in)
        print (">>>>> old points child out: ", points_ex)
    child_elem_in = indexes + points_in
    child_elem_ex = indexes[::-1] + points_ex
    if (verbose):
        print (">>>>> child_elem1: ", child_elem_in)
        print (">>>>> child_elem2: ", child_elem_ex)

    return [new_coords, indexes, intface_edge_indexes, child_elem_in, child_elem_ex, ied_exit]

def get_intersections(coords, elem, intface, ied):

    """
    Finds intersections between a simlpicial mesh element and an interface edge

    Args:
        coords (list(np.array)): list of points
        elem (list(np.array)): elem to intersect (list of points)
        intface (disk_interface): interface
        ied (int): edge index

    Returns:
        intersections (list): intersection coordinates (if any)
        cut_elem_edges (list): edges that are cut
    """

    no_edges = len(elem)
    intersections = []
    cut_elem_edges = []

    q1 = intface.coords[intface.edges[ied][0]]
    q2 = intface.coords[intface.edges[ied][1]]

    for ie in (range(no_edges)):

        p1 = coords[elem[ie]]
        p2 = coords[elem[(ie+1)%no_edges]]

        [cut, intsec_coords] = gmi.calc_intersection_segments(p1,p2,q1, q2)

        # format correctly
        if (cut):
            intersections.append(intsec_coords)
            cut_elem_edges.append(ie)

    # if double intersection one has to decide what is the edge
    # through which the interface enters, and which is the one
    # from which it exits; check the first and make a test on the normal
    # if it fails swap intersections and cut edges
    if (len(intersections)==2):
        intf_normal = gmi.rotate_90_clockwise(q2- q1)
        edge1 = coords[elem[(cut_elem_edges[0]+1)%no_edges]]\
              - coords[elem[ cut_elem_edges[0]]]
        if (np.dot(intf_normal, edge1)<0):
            intersections = intersections[::-1]
            cut_elem_edges = cut_elem_edges[::-1]

    return [intersections, cut_elem_edges]


def check_and_add_new_points (new_coords, new_points):
    """
    Add a bunch of new points to a list of points if not already there
    and provide indexes on the new entries

    Args:
        new_coords (list(np.array)): list of points
        new_points (list(np.arra)): points to add
    """
    indexes = []
    for new_point in new_points:
        found = False
        for ino, indexed_point in enumerate(new_coords):
            if (gmi.R2_norm(new_point - indexed_point)<1e-6):
                indexes.append(ino)
                found = True
        if (not found):
            new_coords.append(new_point)
            indexes.append(len(new_coords)-1)
    return [new_coords, indexes]

def no_check_and_add_new_points (new_coords, new_points):
    """
    Add a bunch of new points without checking if point is already there,
    and provide indexes on the new entries

    Args:
        new_coords (list(np.array)): list of points
        new_points (list(np.arra)): points to add
    """
    indexes = []
    for new_point in new_points:
        new_coords.append(new_point)
        indexes.append(len(new_coords)-1)
    return [new_coords, indexes]


########################################################################################
# FUNCTIONS I'M LOOKING TO REPLACE
########################################################################################

#def elem_edge_intersections(mesh, intface, iel, ied):
#    """
#    Finds intersections between a simlpicial mesh element and an interface edge
#
#    Args:
#        mesh (mesh2D): mesh
#        intface (disk_interface): interface
#        iel (int): element index
#        ied (int): edge index
#
#    Returns:
#        intersections (list): intersection coordinates (if any)
#        cut_elem_edges (list): edges that are cut
#        """
#    no_edges_init = 3  # original no of edges of elements
#    intersections = []
#    cut_elem_edges = []
#
#    ied_ino1 = intface.edges[ied][0]
#    ied_ino2 = intface.edges[ied][1]
#    ied_ino1_coords = intface.coords[ied_ino1]
#    ied_ino2_coords = intface.coords[ied_ino2]
#
#    for ie in (range(no_edges_init)):
#        iel_ino1 = mesh.elem2node[iel][ie]
#        iel_ino2 = mesh.elem2node[iel][ (ie+1)%no_edges_init]
#        iel_ino1_coords = mesh.coords[iel_ino1]
#        iel_ino2_coords = mesh.coords[iel_ino2]
#
#        [cut, intsec_coords] = gmi.calc_intersection_segments\
#                                    (iel_ino1_coords,iel_ino2_coords,\
#                                     ied_ino1_coords, ied_ino2_coords)
#        # format correctly
#        if (cut):
#            intersections.append(intsec_coords)
#            cut_elem_edges.append(ie)
#
#    return [intersections, cut_elem_edges]
#
#
#
#def split_elem(mesh, intface, iel, ied, intersections, cut_elem_edges, verbose = False):
#    """
#    Generate 2 new elements by cutting with the interface
#
#    Args:
#        mesh (mema.Mesh2D): mesh
#        intface (mema.disk_intface): interface
#        iel (int): mesh element index
#        ied (int): index of first (possibly only) interface edge intersecting element iel
#        intersections (list(np.array)): coordinates of intersections with ied
#                                        (either 2 if element cut only by edge ied
#                                         or 1 if cut by a chain of edges)
#        cut_elem_edges (list(int)): edges that are cut on the original triangular element
#        verbose (boolean): if True makes some printings during execution
#
#    Returns:
#         (list): inedexs of points of internal element
#         (list): indexes of points of external element
#         (list): indexes of interface edges common to the 2
#         (list): indexes of generated points coordinates (if new)
#    """
#    if (verbose):
#        print ("in execution of split_elem:")
#
#    No_intf_edges = len(intface.edges)
#
#    #initialize list of new points and point counter
#    new_point_coord = []
#    new_point_idx = []
#    current_no_points = mesh.no_points
#
#    #initialize list of interface edges with first intersecting edge
#    #list is used to map interface edges on mesh to edges on the interface
#    intface_edges = [ied]
#
#    # CASE of 2 intersections: ied cuts twice the element
#    # at two points on different sides: only one interface edge
#    if (len(intersections)==2):
#
#        if (verbose):
#            print ("    edge ", ied, " has 1 intersection with element ", iel)
#
#        # Add points coord and idx if not already there
#        for intsec in intersections:
#            [present, idx] = dt.check_if_present (mesh, intsec)
#            if(not present):
#                new_point_coord.append(intsec)
#                idx = current_no_points
#                current_no_points = current_no_points + 1
#            new_point_idx.append(idx)
#
#        # get index of original vertices (first is on edge cut by v1)
#        v1 = mesh.elem2node[iel][cut_elem_edges[0]]
#        v2 = mesh.elem2node[iel][(cut_elem_edges[0]+1)%3]
#        v3 = mesh.elem2node[iel][(cut_elem_edges[0]+2)%3]
#
#        # determine normal of interface edge ied cutting the element
#        edge_barycenter = 0.5*(intface.coords[(ied+1)%No_intf_edges] + intface.coords[ied])
#        edge_tangent = intface.coords[(ied+1)%No_intf_edges] - intface.coords[ied]
#        edge_normal = np.dot(np.array([[0, 1],[-1, 0]]), edge_tangent) / gmi.R2_norm(edge_tangent)
#
#
#        # return new elems (according to how intf edge cuts elem)
#        # Respect counterclockwise ordering.
#        # To check whether 3 or 4 points (resepctively for el3 and el4)
#        # check the spacing between cut edges (using %)
#        int1 = new_point_idx[0]
#        int2 = new_point_idx[1]
#
#        if  ((cut_elem_edges[1]-cut_elem_edges[0])%3 ==1):
#            el3 = [int2, int1, v2]
#            el4 = [int1, int2, v3, v1]
#        else:
#            el3 = [int1, int2, v1]
#            el4 = [int2, int1, v2, v3]
#
#        # determine whether el3 is external or internal by checking position of only original node of el3 (can be v1 or v2)
#        lonely_vertex = mesh.coords[el3[2]]
#        if (np.dot(lonely_vertex - edge_barycenter, edge_normal) > 0):
#           el_ext = el3
#           el_in  = el4
#        else:
#           el_ext = el4
#           el_in  = el3
#
#        return[el_in, el_ext, intface_edges, new_point_coord]
#
#
#    ########################################################################
#
#    # CASE of 1 intersection found? look for end of chain
#    # and fill list of new nodes
#    elif (len(intersections)==1):
#
#        if (verbose):
#            print ("    edge ", ied, " has several  intersections with element ", iel)
#
#
#        #--->whatch out for exception: first intf edge going out of elem
#        p1 = mesh.coords[mesh.elem2node[iel][0]]
#        p2 = mesh.coords[mesh.elem2node[iel][1]]
#        p3 = mesh.coords[mesh.elem2node[iel][2]]
#        q  = intface.coords[intface.edges[ied][0]]
#        # exception case if edge is getting out
#        exception = gmi.check_if_in_triangle(p1, p2, p3, q)
#
#        if (not exception):
#
#            # first elem cut is the only one cut by the first intf edge
#            cut_elem_edges1 = cut_elem_edges[0]
#
#            # at beginning of new_nodes add first intersection
#            intsec1 = intersections[0]
#            [present, idx] = dt.check_if_present (mesh, intsec1)
#            if(not present):
#                new_point_coord.append(intsec1)
#                idx = current_no_points
#                current_no_points = current_no_points +1
#            new_point_idx.append(idx)
#
#            # loop on interelem edges until you find
#            # the second intersection
#            done = False
#            current_ied = (ied + 1) %No_intf_edges
#
#            while (not done):
#
#                # add first node of each to point list (if not already present) and to new_nodes
#                idx_point_to_add = intface.edges[current_ied][0]
#                point_to_add = intface.coords[idx_point_to_add]
#                [present, idx] = dt.check_if_present (mesh, point_to_add) # idx in coords
#                                                             # whereas idx_point_to_add is in
#                                                             # intf_coords
#                if(not present):
#                   new_point_coord.append(point_to_add)
#                   idx = current_no_points
#                   current_no_points = current_no_points +1
#                new_point_idx.append(idx)
#
#
#                # check if intersection
#                # if intersection set done=true, add last point to new_nodes and register
#                [intersections,cut_elem_edges] = \
#                     elem_edge_intersections(mesh, intface, iel, current_ied)
#                if (len(intersections)>0):
#                    cut_elem_edges2 = cut_elem_edges[0]
#                    intsec2 = intersections[0]
#                    [present, idx] = dt.check_if_present (mesh, intsec2)
#                    if(not present):
#                        new_point_coord.append(intsec2)
#                        idx = current_no_points
#                        current_no_points = current_no_points +1
#                    new_point_idx.append(idx)
#                    done = True
#
#                # add current_ied to list of interface edges before updating
#                intface_edges.append(current_ied)
#                # advance loop
#                current_ied = (current_ied + 1)%No_intf_edges #---> loop can begin again
#
#
#            # get index of original vertices in order of cut (first is on cut edge)
#            v1 = mesh.elem2node[iel][cut_elem_edges1]
#            v2 = mesh.elem2node[iel][(cut_elem_edges1+1)%3]
#            v3 = mesh.elem2node[iel][(cut_elem_edges1+2)%3]
#
#            if (verbose):
#                print ("Original nodes: ", [v1, v2, v3])
#
#            no_new_nodes = len(new_point_idx)
#
#            if (verbose):
#                print ("    >>list of interface points", new_point_idx)
#
#            # To choose how to dispose nodes in generated elements determine if:
#            # case 1: intface gets in at edge k and gets out at edge k+1
#            # case 2: intface gets in at edge k and gets out at edge k+2
#            # case 0: intface gets in at edge k and gets out at edge k
#            # To attribute side consider that the new points are in order for the internal element
#
#            case = (cut_elem_edges2 - cut_elem_edges1)%3
#
#            if (verbose):
#                print ("    >> ordering situation no: ", case)
#
#            if (case == 0):
#
#                l1 = new_point_idx[:] # [:] to avoid copy by reference
#
#                l2 = new_point_idx[::-1]
#                l2 = l2 + [v2, v3, v1]
#
#                return [l1, l2, intface_edges, new_point_coord]
#
#            if (case == 1):
#
#                l1= new_point_idx[::-1]
#                l1.append(v2)
#
#                l2 = new_point_idx[:]
#                l2 = l2 + [v3, v1]
#
#                return [l2, l1, intface_edges, new_point_coord]
#
#            if (case == 2):
#
#                l1 = new_point_idx[:]
#                l1.append(v1)
#
#                l2 = new_point_idx[::-1]
#                l2 = l2 + [v2,v3]
#
#                return [l1, l2, intface_edges, new_point_coord]
#
#            if (verbose):
#                print ("cut edges: ", cut_elem_edges1, cut_elem_edges2)
#
#        #--------> Exception
#        # just do nothing: the loop will find the next intersection
#        else:
#            if (verbose):
#                print ("Exception")
#            # dummy return
#            return [[], [], [], []]
#
#
#
#def break_mesh(mesh, intface, verbose = False):
#    """
#    Generate a new mesh by cutting along the interface
#
#    Args:
#        mesh (Mesh2D): mesh
#        intface (disk_intface): interface
#
#    Returns:
#        Mesh2D: cut mesh (deep copy)
#    """
#    if (verbose):
#        print ("*****************************")
#        print ("        MESH BREAKING        ")
#        print ("*****************************")
#
#    # create a brand new mesh copying the original one (must perform a deep copy)
#    new_mesh = copy.deepcopy(mesh)
#
#    # Initialize cut_elems, a mask for elements that are cut (=1), generated (=2) or untouched (=0)
#    cut_elems = [0]*new_mesh.no_elems
#
#    # loop must be over original elements
#    no_elems_init = new_mesh.no_elems
#
#    for iel in range(no_elems_init):
#        # split elem only at first intersection
#        first_intsec_found = False
#        ied = 0
#        while (not(first_intsec_found) and ied < len(intface.edges)):
#
#            #look for intersection
#            [intersections, cut_elem_edges] = elem_edge_intersections(new_mesh, intface, iel, ied)
#
#            # proceed to break if intersections are found (non empty list)
#            if (intersections!=[]):
#
#                # Mask the Face
#                cut_elems[iel] = 1
#                # Create new nodes and elems
#                [f1, f2, intface_edges, new_point_coord] = split_elem(new_mesh, intface, iel, ied, \
#                                                     intersections, cut_elem_edges, verbose)
#                intf_elems = len(intface_edges)
#                # Check whether exception, if not proceed with splitting
#                if (f1!=[]):
#
#                    first_intsec_found = True
#
#                    if (verbose):
#                        print (">>>From element ", iel, " generated new elements: [",\
#                               len(new_mesh.elem2node), len(new_mesh.elem2node) + 1, "]")
#                        print ("with nodes: ", f1, f2)
#
#                    # increase list of nodes
#                    new_mesh.coords = new_mesh.coords + new_point_coord
#                    new_mesh.no_points = len(new_mesh.coords)
#
#                    # Update list of elems mask of cut elems
#                    new_mesh.elem2node.append(f1)
#                    new_mesh.elem2node.append(f2)
#                    new_mesh.no_elems = len(new_mesh.elem2node)
#
#                    # update mask of cut elements
#                    cut_elems = cut_elems + [2, 2]
#
#                    # Update sid_mask
#                    new_mesh.side_mask = new_mesh.side_mask + [0, 1]
#
#                    # Increase gamma_couples with two last added elements
#                    new_mesh.gamma_couples.append([new_mesh.no_elems-2, new_mesh.no_elems-1])
#
#                    # Increase mesh2intface
#                    new_mesh.mesh2intface.append(intface_edges)
#
#            ied = ied + 1
#
#    # count cut elements to eliminate
#    elems_to_eliminate = sum([1 for cut in cut_elems if cut==1])
#    #print (elems_to_eliminate)
#    #print (cut_elems)
#    # purge inactive elements
#    new_mesh.elem2node = [new_mesh.elem2node[k] for k in range(len(new_mesh.elem2node)) if cut_elems[k]!=1]
#    new_mesh.no_elems = len(new_mesh.elem2node)
#    # shift indexes of mesh.gamma_couples
#    new_mesh.gamma_couples = [ [iel_1-elems_to_eliminate, iel_2-elems_to_eliminate] for [iel_1, iel_2] in new_mesh.gamma_couples]
#    # purge side_mask
#    new_mesh.side_mask = [new_mesh.side_mask[k] for k in range(len(new_mesh.side_mask)) if cut_elems[k]!=1]
#    # recalculate elem2edge
#    new_mesh.elem2edge, new_mesh.no_edges = new_mesh.generate_elem2edge()
#    new_mesh.intface_edges = new_mesh.generate_intface_edges()
#    # reset boundary mask
#    new_mesh.edge_bnd_mask = [-1]*new_mesh.no_edges
#    new_mesh.mark_bnd_edges()
#
#    # set side_mask
#    fill_side_mask(new_mesh)
#
#    return new_mesh

########################################################################################
# END OF FUNCTIONS I'M LOOKING TO REPLACE
########################################################################################


def fill_side_mask(mesh, only_initialize = False):
    """
    Complete side mask by extrapolating value at interface up to the border
    (uses dofs associated to nodes as in ddrin)

    Args:
      mesh (mesh2D): mesh
      no_elems_init     (int): original numer of element
    """

    def glob_nodal_dof (mesh, iel, ino): # this may be useful in general as a mesh method, consider moving inside class
        """
        Provide global dof index of node of an element in the sense of ddrin
        (dofs are doubled along interface)

        Args:
           mesh (mesh2D): mesh
           iel     (int): element index
           ino     (int): node index
           no_elems_init     (int): original numer of element
        """


        # get number of interface points (= total number of interface edges)
        no_intface_points = sum([len(intface_edges_list) for intface_edges_list in mesh.intface_edges])
        # check if element is on interface (index appears in mesh.gamma_couples)
        side = -1
        for icut in range(len(mesh.cuts)):
            if (mesh.cuts[icut][0][0] == iel):
                side = 0
                first_ie = mesh.cuts[icut][1][0]
            elif(mesh.cuts[icut][0][1] == iel):
                side = 1
                first_ie = mesh.cuts[icut][1][1]
            no_intface_edges = len(mesh.intface_edges[icut])

        if (side > 0 and ino >= first_ie and ino < first_ie + no_intface_edges):
            # check side, if external dof is index of point + no_intface_points
            # (interface points come last in mesh.coords and in global dof ordering you have
            # (dof_internal_points, dof_intface_points_in, dof_intface_points_ext))
            return mesh.elem2node[iel][ino] + no_intface_points
        else:
            return mesh.elem2node[iel][ino]

    # IDEA: set a side mask of nodal dofs and initialize by looping on elements
    # (it is !=-1 only at interface initially);
    # then set up a loop where the side_mask of an element is set equal to the value
    # of the side mask of the nodal dofs (no conflicts are possible)
    # repeat until complete exploration of the domain

    # get total nodal dofs
    no_intface_points = sum([len(intface_edges_list) for intface_edges_list in mesh.intface_edges])
    no_nodal_dof = len(mesh.coords) + no_intface_points

    # initialization os side_mask
    # initialization of dof side mask
    # initialization of interface elements
    mesh.side_mask = [-1]*len(mesh.elem2node)
    dof_side_mask =  [-1]*no_nodal_dof
    intface_couples = [mesh.cuts[k][0] for k in range(len(mesh.cuts))]
    for ico, couple in enumerate(intface_couples):
        for pos in range(2):
            iel = couple[pos]
            if (pos==0):
                mesh.side_mask[iel] = 0
            else:
                mesh.side_mask[iel] = 1
            for ino in range(len(mesh.elem2node[iel])):
                dof = glob_nodal_dof(mesh, iel, ino)
                dof_side_mask[dof] = mesh.side_mask[iel]

    # loop until complete exploration
    done = False
    it = 0
    maxit = 10000
    while (not done and it <maxit):
        done= True
        # loop over elements, check if side still to assign and try to assign it
        for iel in range(len(mesh.elem2node)):
            side = mesh.side_mask[iel]
            if (side == -1):
                done = False
                node_per_elem = len(mesh.elem2node[iel])
                # get side of node if defined
                for ino in range(node_per_elem):
                    dof = glob_nodal_dof(mesh, iel, ino)
                    if (dof_side_mask[dof]>-1):
                        mesh.side_mask[iel] = dof_side_mask[dof]
                # distribute side to nodes
                for ino in range(node_per_elem):
                    dof = glob_nodal_dof(mesh, iel, ino)
                    dof_side_mask[dof] = mesh.side_mask[iel]
        it = it + 1
        if (it == 1 and only_initialize): # execute only first round if only interface elements are to be marked
            return

    if (it==maxit):
             warnings.warn("The procedure to assign side exceeds max iterations")

