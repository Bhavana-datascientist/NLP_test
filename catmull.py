import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial as sp_spatial
import pandas as pd
from collections import defaultdict
from vedo import *


def center_point(p1, p2):
    """ 
    returns a point in the center of the 
    segment ended by points p1 and p2
    """
    cp = []
    for i in range(3):
        cp.append((p1[i] + p2[i]) / 2)

    return cp


def sum_point(p1, p2):
    """ 
    adds points p1 and p2
    """
    sp = []
    for i in range(3):
        sp.append(p1[i] + p2[i])

    return sp


def div_point(p, d):
    """ 
    divide point p by d
    """
    sp = []
    for i in range(3):
        sp.append(p[i] / d)

    return sp


def mul_point(p, m):
    """ 
    multiply point p by m
    """
    sp = []
    for i in range(3):
        sp.append(p[i] * m)

    return sp


def get_face_points(input_points, input_faces):
    """
    From http://rosettacode.org/wiki/Catmull%E2%80%93Clark_subdivision_surface
 
    1. for each face, a face point is created which is the average of all the points of the face.
    """

    # 3 dimensional space

    NUM_DIMENSIONS = 3

    # face_points will have one point for each face

    face_points = []

    for curr_face in input_faces:
        face_point = [0.0, 0.0, 0.0]
        for curr_point_index in curr_face:
            curr_point = input_points[curr_point_index]
            # add curr_point to face_point
            # will divide later
            for i in range(NUM_DIMENSIONS):
                face_point[i] += curr_point[i]
        # divide by number of points for average
        num_points = len(curr_face)
        for i in range(NUM_DIMENSIONS):
            face_point[i] /= num_points
        face_points.append(face_point)

    return face_points


def get_edges_faces(input_points, input_faces):
    """
 
    Get list of edges and the one or two adjacent faces in a list.
    also get center point of edge
 
    Each edge would be [pointnum_1, pointnum_2, facenum_1, facenum_2, center]
 
    """

    # will have [pointnum_1, pointnum_2, facenum]

    edges = []

    # get edges from each face

    for facenum in range(len(input_faces)):
        face = input_faces[facenum]
        num_points = len(face)
        # loop over index into face
        for pointindex in range(num_points):
            # if not last point then edge is curr point and next point
            if pointindex < num_points - 1:
                pointnum_1 = face[pointindex]
                pointnum_2 = face[pointindex + 1]
            else:
                # for last point edge is curr point and first point
                pointnum_1 = face[pointindex]
                pointnum_2 = face[0]
            # order points in edge by lowest point number
            if pointnum_1 > pointnum_2:
                temp = pointnum_1
                pointnum_1 = pointnum_2
                pointnum_2 = temp
            edges.append([pointnum_1, pointnum_2, facenum])

    # sort edges by pointnum_1, pointnum_2, facenum

    edges = sorted(edges)

    num_edges = len(edges)
    eindex = 0
    merged_edges = []

    while eindex < num_edges:
        e1 = edges[eindex]
        # check if not last edge
        if eindex < num_edges - 1:
            e2 = edges[eindex + 1]
            if e1[0] == e2[0] and e1[1] == e2[1]:
                merged_edges.append([e1[0], e1[1], e1[2], e2[2]])
                eindex += 2
            else:
                merged_edges.append([e1[0], e1[1], e1[2], None])
                eindex += 1
        else:
            merged_edges.append([e1[0], e1[1], e1[2], None])
            eindex += 1

    # add edge centers

    edges_centers = []

    for me in merged_edges:
        p1 = input_points[me[0]]
        p2 = input_points[me[1]]
        cp = center_point(p1, p2)
        edges_centers.append(me + [cp])

    return edges_centers


def get_edge_points(input_points, edges_faces, face_points):
    """
    for each edge, an edge point is created which is the average 
    between the center of the edge and the center of the segment made
    with the face points of the two adjacent faces.
    """

    edge_points = []

    for edge in edges_faces:
        # get center of edge
        cp = edge[4]
        # get center of two facepoints
        fp1 = face_points[edge[2]]
        # if not two faces just use one facepoint
        # should not happen for solid like a cube
        if edge[3] == None:
            fp2 = fp1
        else:
            fp2 = face_points[edge[3]]
        cfp = center_point(fp1, fp2)
        # get average between center of edge and
        # center of facepoints
        edge_point = center_point(cp, cfp)
        edge_points.append(edge_point)

    return edge_points


def get_avg_face_points(input_points, input_faces, face_points):
    """
 
    for each point calculate
 
    the average of the face points of the faces the point belongs to (avg_face_points)
 
    create a list of lists of two numbers [facepoint_sum, num_points] by going through the
    points in all the faces.
 
    then create the avg_face_points list of point by dividing point_sum (x, y, z) by num_points
 
    """

    # initialize list with [[0.0, 0.0, 0.0], 0]

    num_points = len(input_points)
    temp_points = []
    for pointnum in range(num_points):
        temp_points.append([[0.0, 0.0, 0.0], 0])

    # loop through faces updating temp_points

    for facenum in range(len(input_faces)):
        fp = face_points[facenum]
        for pointnum in input_faces[facenum]:
            tp = temp_points[pointnum][0]
            temp_points[pointnum][0] = sum_point(tp, fp)
            temp_points[pointnum][1] += 1

    # divide to create avg_face_points

    avg_face_points = []

    for tp in temp_points:
        afp = div_point(tp[0], tp[1])
        avg_face_points.append(afp)

    return avg_face_points


def get_avg_mid_edges(input_points, edges_faces):
    """
 
    the average of the centers of edges the point belongs to (avg_mid_edges)
 
    create list with entry for each point 
    each entry has two elements. one is a point that is the sum of the centers of the edges
    and the other is the number of edges. after going through all edges divide by
    number of edges.
 
    """

    # initialize list with [[0.0, 0.0, 0.0], 0]

    num_points = len(input_points)

    temp_points = []

    for pointnum in range(num_points):
        temp_points.append([[0.0, 0.0, 0.0], 0])

    # go through edges_faces using center updating each point

    for edge in edges_faces:
        cp = edge[4]
        for pointnum in [edge[0], edge[1]]:
            tp = temp_points[pointnum][0]
            temp_points[pointnum][0] = sum_point(tp, cp)
            temp_points[pointnum][1] += 1

    # divide out number of points to get average

    avg_mid_edges = []

    for tp in temp_points:
        ame = div_point(tp[0], tp[1])
        avg_mid_edges.append(ame)

    return avg_mid_edges


def get_points_faces(input_points, input_faces):
    # initialize list with 0

    num_points = len(input_points)

    points_faces = []

    for pointnum in range(num_points):
        points_faces.append(0)

    # loop through faces updating points_faces

    for facenum in range(len(input_faces)):
        for pointnum in input_faces[facenum]:
            points_faces[pointnum] += 1

    return points_faces


def get_new_points(input_points, points_faces, avg_face_points, avg_mid_edges):
    """
 
    m1 = (n - 3) / n
    m2 = 1 / n
    m3 = 2 / n
    new_coords = (m1 * old_coords)
               + (m2 * avg_face_points)
               + (m3 * avg_mid_edges)
 
    """

    new_points = []

    for pointnum in range(len(input_points)):
        n = points_faces[pointnum]
        m1 = (n - 3) / n
        m2 = 1 / n
        m3 = 2 / n
        old_coords = input_points[pointnum]
        p1 = mul_point(old_coords, m1)
        afp = avg_face_points[pointnum]
        p2 = mul_point(afp, m2)
        ame = avg_mid_edges[pointnum]
        p3 = mul_point(ame, m3)
        p4 = sum_point(p1, p2)
        new_coords = sum_point(p4, p3)

        new_points.append(new_coords)

    return new_points


def switch_nums(point_nums):
    """
    Returns tuple of point numbers
    sorted least to most
    """
    if point_nums[0] < point_nums[1]:
        return point_nums
    else:
        return (point_nums[1], point_nums[0])


def cmc_subdiv(input_points, input_faces, len_faces=0):
    # 1. for each face, a face point is created which is the average of all the points of the face.
    # each entry in the returned list is a point (x, y, z).

    face_points = get_face_points(input_points, input_faces)
    # get list of edges with 1 or 2 adjacent faces

    edges_faces = get_edges_faces(input_points, input_faces)

    # get edge points, a list of points

    edge_points = get_edge_points(input_points, edges_faces, face_points)

    # the average of the face points of the faces the point belongs to (avg_face_points)                

    avg_face_points = get_avg_face_points(input_points, input_faces, face_points)

    # the average of the centers of edges the point belongs to (avg_mid_edges)

    avg_mid_edges = get_avg_mid_edges(input_points, edges_faces)

    # how many faces a point belongs to

    points_faces = get_points_faces(input_points, input_faces)

    new_points = get_new_points(input_points, points_faces, avg_face_points, avg_mid_edges)

    """
 
    Then each face is replaced by new faces made with the new points,
 
    for a triangle face (a,b,c):
       (a, edge_point ab, face_point abc, edge_point ca)
       (b, edge_point bc, face_point abc, edge_point ab)
       (c, edge_point ca, face_point abc, edge_point bc)
 
    for a quad face (a,b,c,d):
       (a, edge_point ab, face_point abcd, edge_point da)
       (b, edge_point bc, face_point abcd, edge_point ab)
       (c, edge_point cd, face_point abcd, edge_point bc)
       (d, edge_point da, face_point abcd, edge_point cd)
 
    face_points is a list indexed by face number so that is
    easy to get.
 
    edge_points is a list indexed by the edge number
    which is an index into edges_faces.
 
    need to add face_points and edge points to 
    new_points and get index into each.
 
    then create two new structures
 
    face_point_nums - list indexes by facenum
    whose value is the index into new_points
 
    edge_point num - dictionary with key (pointnum_1, pointnum_2)
    and value is index into new_points
 
    """

    # add face points to new_points

    face_point_nums = []

    # point num after next append to new_points
    next_pointnum = len(new_points)

    for face_point in face_points:
        new_points.append(face_point)
        face_point_nums.append(next_pointnum)
        next_pointnum += 1

    # add edge points to new_points

    edge_point_nums = dict()

    for edgenum in range(len(edges_faces)):
        pointnum_1 = edges_faces[edgenum][0] + len_faces
        pointnum_2 = edges_faces[edgenum][1] + len_faces
        edge_point = edge_points[edgenum]
        new_points.append(edge_point)
        edge_point_nums[(pointnum_1, pointnum_2)] = next_pointnum
        next_pointnum += 1

    # new_points now has the points to output. Need new
    # faces

    """
 
    just doing this case for now:
 
    for a quad face (a,b,c,d):
       (a, edge_point ab, face_point abcd, edge_point da)
       (b, edge_point bc, face_point abcd, edge_point ab)
       (c, edge_point cd, face_point abcd, edge_point bc)
       (d, edge_point da, face_point abcd, edge_point cd)
 
    new_faces will be a list of lists where the elements are like this:
 
    [pointnum_1, pointnum_2, pointnum_3, pointnum_4]
 
    """

    new_faces = []

    for oldfacenum in range(len(input_faces)):
        oldface = input_faces[oldfacenum]
        # 4 point face
        if len(oldface) == 4:
            a = oldface[0] + len_faces
            b = oldface[1] + len_faces
            c = oldface[2] + len_faces
            d = oldface[3] + len_faces
            face_point_abcd = face_point_nums[oldfacenum] + len_faces
            edge_point_ab = edge_point_nums[switch_nums((a, b))] + len_faces
            edge_point_da = edge_point_nums[switch_nums((d, a))] + len_faces
            edge_point_bc = edge_point_nums[switch_nums((b, c))] + len_faces
            edge_point_cd = edge_point_nums[switch_nums((c, d))] + len_faces
            new_faces.append((a, edge_point_ab, face_point_abcd, edge_point_da))
            new_faces.append((b, edge_point_bc, face_point_abcd, edge_point_ab))
            new_faces.append((c, edge_point_cd, face_point_abcd, edge_point_bc))
            new_faces.append((d, edge_point_da, face_point_abcd, edge_point_cd))
        if len(oldface) == 3:
            a = oldface[0] + len_faces
            b = oldface[1] + len_faces
            c = oldface[2] + len_faces
            face_point_abc = face_point_nums[oldfacenum] + len_faces
            edge_point_ab = edge_point_nums[switch_nums((a, b))] + len_faces
            edge_point_bc = edge_point_nums[switch_nums((b, c))] + len_faces
            edge_point_ca = edge_point_nums[switch_nums((c, a))] + len_faces
            new_faces.append((a, edge_point_ab, face_point_abc, edge_point_ca))
            new_faces.append((b, edge_point_bc, face_point_abc, edge_point_ab))
            new_faces.append((c, edge_point_ca, face_point_abc, edge_point_bc))

    return np.array(new_points), np.array(new_faces)


def graph_output(output_points, output_faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    """
 
    Plot each face
 
    """

    for facenum in range(len(output_faces)):
        curr_face = output_faces[facenum]
        xcurr = []
        ycurr = []
        zcurr = []
        for pointnum in range(len(curr_face)):
            xcurr.append(output_points[curr_face[pointnum]][0])
            ycurr.append(output_points[curr_face[pointnum]][1])
            zcurr.append(output_points[curr_face[pointnum]][2])
        xcurr.append(output_points[curr_face[0]][0])
        ycurr.append(output_points[curr_face[0]][1])
        zcurr.append(output_points[curr_face[0]][2])

        ax.plot(xcurr, ycurr, zcurr, color='b')

    plt.show()


def get_triangles(output_faces):
    a = []
    for y in output_faces:
        for x in [y]:
            a.append([x[0], x[3], x[1]])
            a.append([x[0], x[3], x[2]])
            a.append([x[1], x[3], x[2]])
            a.append([x[0], x[2], x[1]])
    return a


# Then we create our new normal array:
def vertices_normals(output_points, tri_list):
    utils.record_statement("Calculating the vertices normals starting.....")
    faces = np.array(tri_list)
    vertices = np.array(output_points)
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each
    # triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
    # normals = normalize_v3(n)
    normals = normalize_v3(vertices)
    utils.record_statement("Calculating the vertices normals ended.....")
    return normals


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def geometry_model(uni_data, tetra=False, alpha=10):
    dimensions = {True: 3, False: 2}
    dim = dimensions[tetra]
    utils.record_statement("Geometry model constuction started.....")
    points = uni_data[["x", "y", "z"]].to_numpy()
    # hull = sp_spatial.ConvexHull(points,incremental=True)
    hull = sp_spatial.Delaunay(points[:, :dim], furthest_site=False)
    indices = hull.simplices
    # faces = points[indices, :]
    points_int = np.array(points, dtype='int')
    faces_int = np.array(indices, dtype='int')
    iterations = 1
    output_points, output_faces = points_int, faces_int
    for i in range(iterations):
        # output_points = itemgetter(*sorted(np.unique(output_faces)))(output_points)
        output_points, output_faces = cmc_subdiv(output_points, output_faces, len_faces=len(np.unique(output_faces)))
        points = np.insert(points, 3, 1, axis=1)
        output_points = np.insert(output_points, 3, 0, axis=1)
        output_points = np.concatenate((points, output_points), axis=0)
        output_points = np.unique(output_points, axis=0)

    tri_hull = sp_spatial.Delaunay(output_points[:, :dim], incremental=True)
    indices = tri_hull.simplices.tolist()

    tri_df = pd.DataFrame(output_points, columns=["x", "y", "z", "mapped"], dtype=np.float).round(3)
    normals = vertices_normals(output_points[:, :3], indices)  # output_faces)
    tri_df['normals'] = pd.DataFrame(normals).apply(lambda x: list(x.values), axis=1)
    utils.record_statement("Geometry model constuction Ended.....")
    return tri_df, indices  # tri_list


def convex(arr_points):
    con_hull = sp_spatial.ConvexHull(arr_points)
    output_faces = np.array(con_hull.simplices)
    return arr_points, output_faces


def alphahull(arr_points):
    vertices, output_faces = alpha_shape_3D(arr_points, 10)
    return arr_points, output_faces


def qhull(arr_points):
    tetra = sp_spatial.Delaunay(arr_points)
    output_faces = np.array(tetra.simplices)
    return arr_points, output_faces


def mls(data_points):
    pts0 = Points(data_points, r=3)
    pts1 = smoothMLS2D(pts0, f=0.8)
    reco = recoSurface(pts1, bins=50)
    return reco.points(), reco.faces()


def geometry_construct(uni_data, geo_type=alphahull):
    arr_points = uni_data[["x", "y", "z"]].to_numpy()
    try:
        utils.record_statement("Geometric data construction started .......")
        output_arr_points, output_faces = geo_type(arr_points)

        normals = normalize_v3(np.array(output_arr_points))  # output_faces)
        normals -= normals.mean(axis=0)
        utils.record_statement("Geometric data construction ended .......")

        uni_data['normals'] = pd.DataFrame(normals).apply(lambda x: list(x.values), axis=1)
        return uni_data, output_faces.tolist()
    except Exception as e:
        print(str(e))
        pass


def alpha_shape_3D(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos_df - Dataframe with X Y Z coordinate points.
        alpha - alpha value.
    return
        outer surface vertex indices, edge indices, and triangle indices
    """
    # points = pos_df[["x", "y", "z"]].to_numpy()
    tetra = sp_spatial.Delaunay(points)
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(points, tetra.vertices, axis=0)
    normsq = np.sum(tetrapos ** 2, axis=2)[:, :, None]
    ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))
    a = np.linalg.det(np.concatenate((tetrapos, ones), axis=2))
    Dx = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [1, 2]], ones), axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 2]], ones), axis=2))
    Dz = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 1]], ones), axis=2))
    c = np.linalg.det(np.concatenate((normsq, tetrapos), axis=2))
    r = np.sqrt(Dx ** 2 + Dy ** 2 + Dz ** 2 - 4 * a * c) / (2 * np.abs(a))

    tetras = tetra.simplices
    # Find tetrahedrals
    if alpha > 0:
        tetras = tetra.vertices[r < alpha, :]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:, TriComb].reshape(-1, 3)
    Triangles = np.sort(Triangles, axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles: TrianglesDict[tuple(tri)] += 1
    Triangles = np.array([tri for tri in TrianglesDict if TrianglesDict[tri] == 1])
    # edges
    EdgeComb = np.array([(0, 1), (0, 2), (1, 2)])

    return points, Triangles  # Vertices,Edges,Triangles
