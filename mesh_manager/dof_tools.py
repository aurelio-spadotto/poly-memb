from . import load_mesh
import numpy as np

# check if point is present
def check_if_present(mesh,node_coords):
   for ino in range(mesh.no_points):
        diff = node_coords-mesh.coords[ino]
        if (np.dot(diff,diff)<1e-20):
            #print ("Point already there")
            return [True, ino]
   return [False, -1]


# Convention: internal unknowns, intface unknowns side in, intface unknowns side ex
def glob_idx (mesh,iel,ino):
    idx_point = mesh.elem2node[iel,ino]
    if (idx_point<mesh.no_points_init):
    # point is from the original mesh: not on the int.face
        return idx_point
    else:
    # interface point, doubled: unknown structure:
    # [internal unknowns, internal intface unk, external intface unknowns]
        side = mesh.side_mask[iel]
        if (side==0):
            return mesh.no_points_init + (idx_point-mesh.no_points_init)
        elif (side==1):
            #return no_points_init +(no_points-no_points_init)+(idx_point-no_points_init)
            return idx_point + (mesh.no_points-mesh.no_points_init)
        else:
            print ("Error: this face is cut")

def kron(i,j):
    """
    Kronecker delta
    """
    if (i==j):
        return 1
    else:
        return 0

