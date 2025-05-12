import mesh_manager as mema
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def compute_aspect_ratio(polygon_coords):
    """
    Compute the robust aspect ratio of a polygon using PCA
    (bounding box in the system of princiapl axes)
    """
    # Convert polygon coordinates to a NumPy array
    coords = np.array(polygon_coords)

    # Step 1: Compute the centroid
    centroid = np.mean(coords, axis=0)
    coords_centered = coords - centroid

    # Step 2: Perform PCA to find principal axes
    cov_matrix = np.cov(coords_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Eigenvectors give the directions of the principal axes
    principal_axes = eigenvectors[:, ::-1]  # Reverse order to sort by largest eigenvalue

    # Step 3: Transform the polygon to align with principal axes
    transformed_coords = coords_centered @ principal_axes

    # Step 4: Compute the bounding box in the rotated space
    min_x, min_y = transformed_coords.min(axis=0)
    max_x, max_y = transformed_coords.max(axis=0)
    lengths = (max_x - min_x, max_y - min_y)

    # Step 5: Aspect ratio is the ratio of the longest to shortest side
    aspect_ratio = max(lengths) / min(lengths)
    return aspect_ratio

def mark_bad_quality_elements (mesh, reference_mesh, cryt_size, cryt_skewness):
    """
    Provides a quality mask for mesh elements
    Args:
        mesh (mema.2Dmesh): mesh
        reference_mesh (mema.mesh2D): uncut mesh
        cryt_size (float): minimal accepted size relative to standard far-from interface simplex
        cryt_skewness (float): maximal accepted aspect ratio (skewness)
    Returns:
        list (int): 0 if good quality, 1 if bad
    """

    [min_hT, max_hT] = reference_mesh.get_mesh_size()

    quality_mask = []
    for iel in range(len(mesh.elem2node)):
        hT = np.sqrt(mesh.calc_surface(iel))
        elem = [mesh.coords[k] for k in mesh.elem2node[iel]]
        if (hT<cryt_size*min_hT or compute_aspect_ratio(elem)>cryt_skewness):
            quality_mask.append(1)
        else:
            quality_mask.append(0)

    return quality_mask

def agglomerate (mesh, ref_mesh, cryt_size, cryt_skewness, verbose=False):
    """
    Agglomerate bad quality elements
    Args:
        mesh (mema.mesh2D): mesh
        ref_mesh (mema.mesh2D): reference uncut mesh
        cryt_size (float): minimal accepted relative size for elements
        cryt_skewness (float): maximal accepted skewness for elements
        verbose (boolean): display partial outputs
    """

    if verbose: print ("Executing agglomerate:")

    elem2node = copy.deepcopy(mesh.elem2node)
    cuts = copy.deepcopy(mesh.cuts)

    # Select bad quality elements
    quality_mask = mark_bad_quality_elements(mesh, ref_mesh, cryt_size, cryt_skewness)
    bad_quality_elems = [k for k in range(len(quality_mask)) if quality_mask[k]==1]

    # Select agglomerating element
    # element can be selected more than once as agglomerating
    # element can be selected as agglomerating and then be agglomerated
    # Create a list of agglomerations couples [agglomerating, to agglomerate]
    agglomerations = []
    elem2elem = mesh.generate_elem2elem()
    for bad_elem in bad_quality_elems:
        candidates = elem2elem[bad_elem]
        candidate_surfaces = [mesh.calc_surface(iel) for iel in candidates]
        # Scores:
        # biggest: 20
        # good quality: 10
        # triangle: 5
        scores = np.zeros(len(candidates))
        for ica, iel in enumerate(candidates):
            # attribute scores
            if (candidate_surfaces[ica]==max(candidate_surfaces)):
                scores[ica] += 20
            if (quality_mask[iel]==0):
                scores[ica] += 10
            if (len(mesh.elem2node[iel])==3):
                scores[ica] += 5
            if (iel==bad_elem): # don' take element itself
                scores[ica] = 0
        # Take first one that has the maximal score (possibly several have)
        agglomerating_elem = [candidates[k] for k in range(len(scores)) if scores[k] == max(scores)][0]
        if verbose:
            if (max(scores)==0): print ("NO VIABLE AGGLOMERATORS")
        agglomerations.append([agglomerating_elem, bad_elem])

    if verbose:
        print ("Agglomerations to perform: ")
        print (agglomerations)

    # get size of agglomeration list
    no_agglom = len(agglomerations)

    # For each agglomeration defin several lists:

    # agglomerate_elems: contains list of points of agglomerate element resulting from agglomeration
    agglomerate_elems = [[] for k in range(no_agglom)] # agglomerate elements as lists of points

    # each agglomerate element can lie on several cuts, so that there are several
    # chains of edges each one referred to a single cut:
    # keep track of this with list intface_data:
    # triplets in the form:
    # [cut, len(cut), first_ie_of_cut (local to agglomerated elem)]
    # (will use intface_data to update cuts structure)
    intface_data = [[] for k in range(no_agglom)]

    # final_agglomerating list: index of the largest agglomerate element containing original
    # element to agglomerate (because several consecutive agglomerations are possible)
    final_agglomerating = []
    for k in range(no_agglom):
        if verbose: print ("Looking for final agglomerator: couples: ")
        final_agglomerating.append(get_final_agglomerating(agglomerations, k, verbose=verbose))

    # remaining_elems: elements output of agglomeration procedure
    # (no repetitions, so not same size of agglomerations)
    remaining_elems = []
    for k in range(no_agglom):
        if (final_agglomerating[k] not in remaining_elems):
            remaining_elems.append(final_agglomerating[k])

    # Index of agglomerate element if previously agglomerated
    # (necessary to perform second (or more) level agglomerations)
    agglomerating_previously_agglom = [-1 for k in range(no_agglom)]
    to_agglomerate_previously_agglom  = [-1 for k in range(no_agglom)]

    # flag to keep track of agglomerations already performed
    agglomeration_done = [False for k in range(no_agglom)]

    if verbose: print ("Done setting-up agglomerations metadata")

    # loop while until all of the agglomerations are performed
    # loop over agglomerations, and skip if the element to
    # agglomerate is still among those that have to agglomerate
    done = False
    while (not done):
        if verbose: print ("***")
        done = True
        for iag in range(no_agglom):
            if (not agglomeration_done[iag]):
                done = False

                # agglomerate if elem to agglomerate is not among elements that still have to agglomerate
                to_agglomerate = agglomerations[iag][1]
                if (not to_agglomerate in [agglomerations[k][0] for k in range(no_agglom)\
                                           if agglomeration_done[k] ==0] ):

                    # Get involved elements:
                    # either they are original elements of the mesh,
                    # or thy are the result of some previous agglomeration
                    # to check this use ***_previoulsy_agglom
                    # (get agglomerating and to_agglomerate as lists of nodes)

                    if verbose: print ("Agglomeration: ", agglomerations[iag])

                    agglomerating_elem_idx  = agglomerating_previously_agglom [iag]
                    if (agglomerating_elem_idx == -1):
                        iel_agglomerating = agglomerations[iag][0]
                        agglomerating_elem = elem2node[iel_agglomerating]
                        intface_data_agglomerating  = get_intface_data(mesh, iel_agglomerating)
                        if verbose: print ("agglomerating: ", iel_agglomerating)
                    else:
                        if verbose: print ("agglomerating already agglomerated in agglomeration", agglomerating_elem_idx)
                        agglomerating_elem = agglomerate_elems[agglomerating_elem_idx]
                        intface_data_agglomerating = intface_data [agglomerating_elem_idx]

                    to_agglomerate_elem_idx  = to_agglomerate_previously_agglom [iag]
                    if (to_agglomerate_elem_idx == -1):
                        iel_to_agglomerate = agglomerations[iag][1]
                        to_agglomerate_elem = mesh.elem2node[iel_to_agglomerate]
                        intface_data_to_agglomerate  = get_intface_data(mesh, iel_to_agglomerate)
                        if verbose: print ("to agglomerate: ", iel_to_agglomerate)
                    else:
                        if verbose: print ("to agglomerate already agglomerated in agglomeration: ", to_agglomerate_elem_idx)
                        to_agglomerate_elem = agglomerate_elems[to_agglomerate_elem_idx]
                        intface_data_to_agglomerate = intface_data [to_agglomerate_elem_idx]


                    # Agglomerate:

                    agglomerate_elem, intface_data_agglomerate = \
                         agglomerate_2_elems (agglomerating_elem, to_agglomerate_elem,\
                                              intface_data_agglomerating, intface_data_to_agglomerate, verbose=verbose)

                    if verbose: print ("agglomerate element: ", agglomerate_elem)
                    if verbose: print ("agglomerate_intface data: ", intface_data_agglomerate)

                    # add agglomerate element and relative intface_data
                    agglomerate_elems [iag] = agglomerate_elem
                    intface_data [iag] = intface_data_agglomerate

                    # If agglomerating appears elsewhere, mark it
                    for iag_bis in range(no_agglom):
                        if (iag_bis != iag):
                            if (agglomerations[iag_bis][0] == agglomerations[iag][0]):
                                agglomerating_previously_agglom [iag_bis] = iag
                            if (agglomerations[iag_bis][1] == agglomerations[iag][0]):
                                to_agglomerate_previously_agglom [iag_bis] = iag

                    # Remove agglomeration couple
                    agglomeration_done [iag] = True
        if verbose:
            print (">>> recap: ")
            print (to_agglomerate_previously_agglom)
            print (agglomerating_previously_agglom)

    if verbose:
        print ("Agglomerations done")



    # Correct eleme2node and cuts using final_agglomerating
    for k in range(len(remaining_elems)):
        iel = remaining_elems[k]
        # index of agglomerate element in agglomerations
        # the agglomerate element is the biggest among the
        # agglomerate elements originating by iel as agglomerating element
        agglomerate_elem_list = [l for l in range(no_agglom) if agglomerations[l][0]==iel]
        sizes_agglomerate = [len(agglomerate_elems[k]) for k in agglomerate_elem_list]
        size_final_agglomerate = max(sizes_agglomerate)
        iag = [idx for idx in agglomerate_elem_list if len(agglomerate_elems[idx])==size_final_agglomerate][0]
        agglomerate_elem = agglomerate_elems[iag]
        elem2node[iel] = agglomerate_elem
        if verbose:
            print ("Remaining elements: ", iel,"-->", agglomerate_elem)


    if verbose:
        print ("In elem2node replaced agglomerating elements with agglomerate elements")

    # To correct cuts, for each cut we need to correct the elements
    # in the couple, and first_ie (while no of edges is intact)
    agglomerated = [agglomerations[k][1] for k in range(no_agglom)]

    for icut in range(len(cuts)):
        couple = cuts[icut][0]
        for pos, iel in enumerate(couple):
            # Check if agglomerated
            for iag, iel_agglo in enumerate(agglomerated):
                if (iel==iel_agglo):
                    # change to elem of agglomerate elem
                    cuts[icut][0][pos] = final_agglomerating[iag]
                    # loop over intface data list of agglomerate elem
                    # to set first_ie
                    for info in intface_data[iag]:
                        if (info[0]==icut):
                            # get first_ie in new agglomerate elem
                            cuts[icut][1][pos] = info[2]

    if verbose:
        print ("Updated mesh.cuts")


    # Suppress agglomerated elements (if any)
    if (len(agglomerated)>0):
        elem2node, cuts = suppress_elements (elem2node, cuts, agglomerated)

    if verbose:
        print ("Suppressed agglomerated elements")

    # Generate agglomerated mesh (no nodes are suppressed, only agglomerated elements)
    # in constuctor, elem2edge conncetivity is recalculated
    agglomerated_mesh = mema.mesh2D (elem2node, mesh.coords, bnd_dof_type = "edge", d = mesh.d)
    # provide cuts
    agglomerated_mesh.cuts = cuts

    # TREAT side_mask and other structures
    agglomerated_mesh.intface_edges = agglomerated_mesh.generate_intface_edges()

    # set side mask
    mema.fill_side_mask(agglomerated_mesh, len(ref_mesh.coords))

    # agglomeration product
    agglomeration_products = [agglomerations, agglomerate_elems, final_agglomerating, intface_data]


    return agglomerated_mesh, agglomeration_products


def agglomerate_2_elems (elem_agglomerating, elem_to_agglomerate,\
                         intface_data_agglomerating, intface_data_to_agglomerate, verbose=False):
    """
    Agglomerate 2 elements provided as lists of points
    ALERT: currently, agglomerations are done under hp that elements share a connected chain of points

    Args:
    elem_agglomerating (list(int)): agglomerating element as list of points
    elem_to_agglomerate (list(int)): element to agglomerate as list of points
    intface_data_agglomerating (list(list(int))): interface data (formatted as detailed in get_intface_data) for agglomerating element
    intface_data_to_agglomerate(list(list(int))): interface data (formatted as detailed in get_intface_data) for element to agglomerate
    verbose (boolean): display in-function outputs
    """
    common_points = []

    # get local indexes of elem agglomerating
    # and on elem to agglomerate of common points
    for ino0 in range(len(elem_agglomerating)):
        point0 = elem_agglomerating[ino0]
        for ino1 in range(len(elem_to_agglomerate)):
            point1 = elem_to_agglomerate[ino1]
            if (point0 == point1):
                common_points.append([ino0, ino1])


    if verbose:
        print ("common_points: ", common_points)
        print ("intface_data_agglomerating: ",  intface_data_agglomerating)
        print ("intface_data_to_agglomerate: ", intface_data_to_agglomerate)

    # Compose list of points of agglomerate element
    agglomerate_elem = []

# >>>>>>>>
    # Get local indexes of endpoints, points at extrema of chain of common points
    # Get the first of the chain, which is not necessarily the first of common_point list;
    # to check, verify if consecutive point is also a common point
    # take it as first if it is not the case, and take the following as second

    ino = 0
    found = False
    while (ino<len(common_points) and not found):
        if (not((common_points[ino][0] +1)%len(elem_agglomerating) in [common_points[k][0] for k in range(len(common_points))])):
            endpoints_agglomerating =  [common_points[(ino+1)%len(common_points)][0], common_points[ino][0]]
            endpoints_to_agglomerate  =[common_points[(ino+1)%len(common_points)][1], common_points[ino][1]]
            found = True
        ino +=1


    if verbose:
        print ("endpoints")
        print (endpoints_agglomerating, endpoints_to_agglomerate)

    # start a loop from initial point of agglomerating elem,
    # if the treated point is the first of the chain of common points
    # add it and start looping forward over the points of the agglomerated elem (consider point are always ordered co-clock-wise).
    # End the internal loop over points of agglomerated when second endpoint is found

#    ino_agglomerating = 0
#    maxit_ex = 0 # in case of bug and infinite loop
#    maxit_in = 0
#    while (ino_agglomerating < len(elem_agglomerating) and maxit_ex<20):
#        agglomerate_elem.append(elem_agglomerating[ino_agglomerating])
#        if (ino_agglomerating==endpoints_agglomerating[0]):
#            ino_to_agglomerate = (endpoints_to_agglomerate[0] +1)%len(elem_to_agglomerate)
#            while (ino_to_agglomerate != endpoints_to_agglomerate[1] and maxit_in <20):
#                agglomerate_elem.append(elem_to_agglomerate[ino_to_agglomerate])
#                ino_to_agglomerate = (ino_to_agglomerate +1)%len(elem_to_agglomerate)
#                maxit_in +=1
#            # restart from second endpoint on elem_agglomerating
#            ino_agglomerating = endpoints_agglomerating[1] -1
#        ino_agglomerating += 1
#        maxit_ex +=1

    maxit_ex = 0 # in case of bug and infinite loop
    maxit_in = 0

    starting_ino_agglomerating = endpoints_agglomerating[0]
    ino_agglomerating = starting_ino_agglomerating
    while ((ino_agglomerating != starting_ino_agglomerating and maxit_ex<20) or maxit_ex==0):
        agglomerate_elem.append(elem_agglomerating[ino_agglomerating])

        if (ino_agglomerating==endpoints_agglomerating[0]):
            ino_to_agglomerate = (endpoints_to_agglomerate[0] +1)%len(elem_to_agglomerate)
            while (ino_to_agglomerate != endpoints_to_agglomerate[1] and maxit_in <20):
                agglomerate_elem.append(elem_to_agglomerate[ino_to_agglomerate])
                ino_to_agglomerate = (ino_to_agglomerate +1)%len(elem_to_agglomerate)
                maxit_in +=1
            # restart from second endpoint on elem_agglomerating
            ino_agglomerating = (endpoints_agglomerating[1] -1)%len(elem_agglomerating)

        ino_agglomerating = (ino_agglomerating +1)%len(elem_agglomerating)
        maxit_ex +=1


# =========
#    Old implementation
#    #select starting point
#    if ((common_points[0][0] + 1)%len(elem_agglomerating)==common_points[1][0]):
#        first_common_0 = common_points[0][0]
#        first_common_1 = common_points[0][1]
#    else:
#        first_common_0 = common_points[1][0]
#        first_common_1 = common_points[1][1]
#
#
#    for ino0 in range(len(elem_agglomerating)):
#        if (ino0 == first_common_0):
#            for k1 in range(len(elem_to_agglomerate) - 1):
#                ino1 = (k1 + first_common_1)%len(elem_to_agglomerate)
#                agglomerate_elem.append(elem_to_agglomerate[ino1])
#        else:
#            agglomerate_elem.append(elem_agglomerating[ino0])
#<<<<<<<

    # Adjust first_ie in intface data of elem to agglomerate

    for icut in range(len(intface_data_agglomerating)):
        first_ie = intface_data_agglomerating[icut][2]
        point_first_ie = elem_agglomerating[first_ie]
        for ino, point in enumerate(agglomerate_elem):
            if point == point_first_ie:
                intface_data_agglomerating[icut][2] = ino

    for icut in range(len(intface_data_to_agglomerate)):
        first_ie = intface_data_to_agglomerate[icut][2]
        point_first_ie = elem_to_agglomerate[first_ie]
        for ino, point in enumerate(agglomerate_elem):
            if point == point_first_ie:
                intface_data_to_agglomerate[icut][2] = ino

    return agglomerate_elem, intface_data_agglomerating+intface_data_to_agglomerate

def get_intface_data (mesh, iel):
    """
    Returns [cut, no_edges, first_ie] (data are used to update cuts field of agglomeratr mesh)
    Args:
        mesh (mema.mesh2D): mesh
        iel (int): element index
    """
    found = False
    icut = 0
    while (not found and icut <len(mesh.cuts)):
        couple = mesh.cuts [icut][0]
        for pos, iel_couple in enumerate(couple):
            if (iel==iel_couple):
                found = True
                # recalling that cut = [couple, starting_ie (couple), edge_to_intface]
                # intface_data = [cut, len(cut), first_ie_of_cut (local to agglomerated elem)]
                return [ [icut, len(mesh.cuts[icut][2]), mesh.cuts[icut][1][pos]] ]
        icut += 1

    return [] # return empty list

def get_final_agglomerating (agglomerations, iag, verbose=False):
    """
    Internal procedure of agglomerate:
    Same element can find itself to be agglomerated several times,
    Get final element that ends up agglomerating element with index iag

    Args:
        agglomerations (list(list(int))): list of agglomerations to perform
        iag (int): index in list of agglomerations
    """
    if verbose: print (agglomerations[iag][1],"-->")
    to_agglomerate_elements = [agglomerations [k][1] for k in range(len(agglomerations))]
    agglomerating = agglomerations[iag][0]
    if (agglomerating in to_agglomerate_elements):
        # get index of agglomerating as agglomerted idx
        idx = [k for k in range(len(agglomerations)) if agglomerations[k][1]==agglomerating][0]
        return get_final_agglomerating (agglomerations, idx, verbose)
    else:
        if (verbose): print (agglomerating)
        return agglomerating

def suppress_elements (elem2node, cuts, list_of_elems):
    """
    Suppress elements from mesh
    cuts field is updated accordingly
    Args:
        elem2node (list(list(int))): element to node connectivity
        cuts (list): cuts field (interface-related data)
        list_of_elems (list(int)): list of elements to suppress
    Returns:
        new_elem2node: new element to node connectivity
        cuts: new cuts field
    """

    new_elem2node = copy.deepcopy(elem2node[0:list_of_elems[0]])
    for k in range(len(list_of_elems)-1):
        new_elem2node += elem2node[list_of_elems[k] +1 :list_of_elems[k+1]]
    new_elem2node += elem2node[list_of_elems[-1]+1:]

    new_cuts = copy.deepcopy(cuts)
    for icut in range(len(cuts)):
        couple = cuts[icut][0]
        for pos, couple_elem in enumerate(couple):
            # get how many elements are suppressed before couple_elem
            suppressed_elems_before = sum( [1 for k in list_of_elems if k< couple_elem])
            new_cuts[icut][0][pos] -= suppressed_elems_before

    return new_elem2node, new_cuts

