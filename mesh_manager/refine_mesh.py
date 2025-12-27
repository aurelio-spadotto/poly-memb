from . import load_mesh as lm
from . import geometric_info as gmi
from . import dof_tools as dt
import numpy as np
import copy
import warnings
import sys

def default_cryterion (iel, elem_coords):
    """
    Check if cryterion to refine is respected
    XXX dummy cryterion 

    Args:
        iel (int): element index
        elem_coords: element node coordinates
    Returns:
        bool
    """
    node_dist_from_circle = [ (gmi.R2_norm(p) - 0.25) for p in elem_coords] # radius of the vertices
    first_value = node_dist_from_circle[0]
    for idx in range(1,len(node_dist_from_circle)):
        next_value = node_dist_from_circle[idx]
        if (first_value*next_value<0):
            return True
    return False


def refine_mesh(mesh0, cryterion=default_cryterion, additional_arguments=[], debug= False):
    """
    Refines a triangular mesh; triangular in shape,
    but not necessarily as for no of points
    (we want to be able to realize several levels of refinement)

    Args:
        mesh(mema.mesh2D): mesh to refine
        cryterion(callable): a function that specifies the cryterion to
                             refine an element. It has signature:
                             cryterion(iel, elem_coords, *add_args), where
                             elem_coords is the list of coordinates of nodes
                             of an element (as a pseudo-triangle), and add_args is a list of additional
                             arguments
        additional_arguments(list): addtional arguments for cryterion 
        debug (bool): print messages 
    Returns:
        mema.mesh2D
    """

    new_coords       = copy.deepcopy(mesh0.coords)
    new_elem2node    = copy.deepcopy(mesh0.elem2node)
    deleted_elements = [] # list of refined elements to delete
    added_elements   = [] # list of created child elements

    if debug: counter = 0 # 

    # new_eleme2node is a dynamical list of elements, whose
    # size is fixed, but whose elements can change size
    # during the refinement procedure;
    # during the loop there is a 1:1 correspondence between
    # elements of mesh0 end elements of new_eleme2node

    for iel in range(mesh0.no_elems):
        
        nodes = new_elem2node[iel] # the dynamical version, potentially already modified with new nodes
        if debug: print ("nodes: ", nodes)
        elem_node_coords = [new_coords[node] for node in nodes]
        v0, v1, v2 = vertex_indices = get_triangle_vertices(elem_node_coords)  # local indexes of triangle
                                                                               # element could be a pseudo-polygon
        elem_triangle_vertices = [elem_node_coords[k] for k in vertex_indices] # coordinates of vertices

        if (cryterion(iel, elem_node_coords, *additional_arguments)):

            #if debug: 
                # if counter > 2: break

            # get the nodes 
            
            if debug: print ("************************************")
            if debug: print ("Element to refine: ", iel)
            if debug: print ("Nodes of element to refine: ", nodes)
            if debug: print ("Coords of nodes of element to refine: ", elem_node_coords)
            # get the vertices (3 nodes thata are vertices of triangle)
            # it is ok to suppose that the vertices are three even when applying recursively,

            # get the three segments of the boundary (with extrema included)
            segments = [
                nodes[v0:v1+1],
                nodes[v1:v2+1],
                nodes[v2:]+nodes[0:v0+1]
            ]
            if debug: print (v0, v1, v2)
            if debug: print ("Three segments: ", segments)
            
            # inspect each segment
            midpoints = []
            subedges = [] # the two lists of points making up the subedges (with extrema)
            for segment in segments:
                v0             = segment[0]
                v1             = segment[-1]
                v0_coord       = new_coords[segment[0]]
                v1_coord       = new_coords[segment[-1]]
                midpoint_coord = 0.5*(v0_coord+v1_coord)
                # if segment composed of only two points, it is a proper edge, then it has to be broken
                if (len(segment)==2):
                    if debug: print ("Segment is still not cut")
                    # add midpoint to the global list of points (coords) and add it to set of midpoints
                    midpoint = len(new_coords) # index of new point
                    new_coords.append(midpoint_coord)
                    # also, get the two subedges
                    subedge0 = [v0, midpoint]
                    subedge1 = [midpoint, v1]
                    # look for element sharing the extrema of the edge (only one must be found)
                    if debug: print ("Edge to be split: ", [v0, v1])
                    neigh, split_ie = get_neighbor(new_elem2node, mesh0.elem_bnd_mask, iel, v0, v1)
                    # if split side is on boundary do nothing
                    if (not neigh== -1):
                        # add the point to the element
                        neigh_point_list   = new_elem2node[neigh]
                        neigh_size = len(neigh_point_list)
                        if debug: print ("Found a neighbor: ", neigh_point_list)
                        if debug: print ("Insert midpoint after node: ", split_ie)
                        if debug: print ("Midpoint: ", midpoint)
                        #new_elem2node[neigh] = neigh_point_list[0:(split_ie+1)%neigh_size]+[midpoint]+neigh_point_list[(split_ie+1)%neigh_size:]
                        new_elem2node[neigh] = neigh_point_list[0:split_ie+1]+[midpoint]+neigh_point_list[split_ie+1:]
                        if debug: print ("Modified neighbor:", new_elem2node[neigh])
                # else, get the two subedges
                else:
                    # get the midpoint; the one with the right coordinate
                    for idx, ino in enumerate(segment):
                        if gmi.R2_norm(new_coords[ino]-midpoint_coord)<1e-12:
                            midpoint_idx = idx # local index
                            midpoint = ino
                    if debug: print ("midpoint: ", midpoint)
                    subedge0 = segment[0:midpoint_idx + 1] # midpoint included
                    subedge1 = segment[midpoint_idx:]
                    if debug: print ("Edge already split divided into: ", subedge0, subedge1)
                # collect the midpoints and subedges
                midpoints.append(midpoint)
                subedges.append([subedge0, subedge1])               
 
            # compose the 4 children elements by connecting the subedges and the midpoints
            # XXX HERE WE MAKE A PRECISE CHOICE FOR ORDERING
            # XXX attention to excluding last point of each list of points

            child0 = subedges[0][0] + subedges[2][1][0:-1]
            child1 = subedges[1][0] + subedges[0][1][0:-1]
            child2 = subedges[2][0] + subedges[1][1][0:-1]
            child3 = midpoints

            if debug: print ("Children ", child0, child1, child2, child3)

            # add children elements to the list of new elements 
            added_elements += [child0, child1, child2, child3]
            # add the refined element to the list of elements to cancel
            deleted_elements.append(iel)

            if debug: counter +=1 # debug: stop after first refined element 

    # get the list of elements of the new refined mesh
    if debug: print("*************************")
    if debug: print("*************************")
    if debug: print ("Deleted Elements: ", deleted_elements)
    purged_new_elem2node = [elem for iel, elem in enumerate(new_elem2node) if iel not in deleted_elements]
    new_elem2node = purged_new_elem2node + added_elements            

    return lm.mesh2D(new_elem2node, new_coords, bnd_dof_type='edge', d=mesh0.d)


def get_triangle_vertices(elem_coords):
    """
    Given a polygon with a triangular shape (such as those that you can get after nonconformal refinement (hanging nodes))
    Extracts the vertices of the triangle (local index)
    Beware: it only works with actual triangles, don't apply after breaking along the mesh
    """
    nodes    = elem_coords
    size     = len(nodes)
    vertices = []
    tol = 1e-10

    for ino, node in enumerate(nodes): 
        # coords of node get adjacent points
        p       = node
        p_left  = nodes[(ino-1)%size]
        p_right = nodes[(ino+1)%size]
        # get the two vectors stemming from p
        v_left  = p_left  - p
        v_right = p_right - p
        # if they dont'make an acute angle they're not vertices
        if ( abs(-1.0 - (v_left[0]*v_right[0] + v_left[1]*v_right[1])/(gmi.R2_norm(v_left)*gmi.R2_norm(v_right))) ) > tol: # cos theta approx -1  for nodes between collinear adjacent points
             vertices.append(ino)

    return vertices

def get_neighbor(elem2node, elem_bnd_mask, iel, v0, v1):
    """
    Look for the element sharing with iel the vertices v0 and v1

    Args:
        elem2node(list): current list of element, already modified with new points
        elem_bnd_mak(list): to check if element lies on boundary
        v0 (int): vertex 0 (index)
        v1 (int): vertex 1 (index)
    Returns:
        int: index of found neighbor
        int: local index of edge to split
    """

    for iel_cand, candidate in enumerate(elem2node):
        if (iel_cand!=iel): # XXX: since indexes are swapped it should not be necessary to check this
            for index, node in enumerate(candidate):
                # beware: v0 and v1 are swapped in neighbor element since nodes are listed counterclock-wise
                if (node==v1 and candidate[(index+1)%len(candidate)]==v0):
                    return iel_cand, index
        
    # If not found and not on boundary
    if (elem_bnd_mask != -1):
        return [-1, -1]
    raise ValueError(f"No neighboring element found")

def intface_cryterion (iel, elem_coords, intface):
    """
    Check if element is intersected by interface

    Args:
        iel (int): element index
        elem_coords(list): element node coordinates
        intface (mema.disk_intface): interface
    Returns:
        bool
    """
    # get pseudo triangle
    vertices_idx = get_triangle_vertices(elem_coords)
    triangle_coords = [elem_coords[k] for k in vertices_idx]

    hT = gmi.R2_norm(triangle_coords[0]-triangle_coords[1])
    tol = 1e-6
    clip = gmi.get_clip(intface.coords, triangle_coords)
    # intersected if clip is not empty and does not coincide with clipping element (because the latter is entirely contained) 
    if (clip==[]):
        return False
    if (gmi.R2_norm(gmi.barycenter(triangle_coords) - gmi.barycenter(clip))/hT < tol):
        #print (mema.R2_norm(barycenter(elem_coords) - barycenter(clip))/hT)
        return False
    
    return True

def harmonize_cryterion (iel, elem_coords):
    """
    Check if element has neighbors that are too small

    Args:
        iel (int): element index
        elem_coords(list): element node coordinates
    Returns:
        bool
    """
    short_side = 1e16
    long_side  = 1e-16
    tol        = 1/2
    for ino in range(len(elem_coords)):
        h          = gmi.R2_norm(elem_coords[(ino+1)%len(elem_coords)]-elem_coords[ino])
        short_side = min(short_side, h)
        long_side  = max(long_side, h)
    
    if (short_side/long_side < tol):
        return True
    
    return False


