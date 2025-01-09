from . import load_mesh
import numpy as np
from sympy import symbols, Eq, solve



def is_proper(value):
    """
    Checks if value is real and in interval(0,1].

    Parameters:
    - value: numerical value to check

    Returns:
    True or False
    """
    return (value.is_real and abs(value)<=1 and value>0)

def rotate_90_clockwise(vec):
    """
    Rotate a vector 90 degrees clockwise
    Args:
        vec (np.array): vector in R^2
    Returns:
        (np.arrya): rotated vector in R^2
    """
    rot = np.array([[0, 1],[-1, 0]])
    return np.dot(rot, vec)

def calc_intersection(x1, x2, y1, y2, rho):
    """
    Check if segment on a plane intersects with ball of radius rho.

    Parameters:
    - x1: x coord of point1
    - x2: x coord of point2
    - y1: y coord of point1
    - y2: y coord of point2
    - rho: disk radius

    Returns:
    - True/False
    - intersection position
    """
    # Define variables
    x, y, t= symbols('x y t')
    # Equation of the circle centered at the origin
    circle_equation = Eq(x**2 + y**2, rho**2)
    # Parametric equations of the line segment
    segment_x = x1 + (x2 - x1) * t
    segment_y = y1 + (y2 - y1) * t
    # Substitute parametric equations into the circle equation
    circle_intersection = circle_equation.subs({x: segment_x, y: segment_y})
    # Solve the resulting quadratic equation for t
    intersection_points = solve(circle_intersection, t)
    #print("Curvilinear abscissa: ", intersection_points)
    if (len(intersection_points)==0):
        return [False, np.array([0.0,0.0])]
       # Evaluate the parametric equations at the intersection points
    intersection_coordinates = \
       [(segment_x.subs(t, point), segment_y.subs(t, point)) for point in intersection_points]
    #print ("Intersection points: ", intersection_coordinates)
    if (is_proper(intersection_points[0])):
        #print ("it's the first")
        point = intersection_coordinates[0]
        return [True, np.array([point[0], point[1]])]
    elif (is_proper(intersection_points[1])):
        #print ("it's the second")
        point = intersection_coordinates[1]
        return [True, np.array([point[0], point[1]])]
    else:
        return [False, np.array([0.0,0.0])]

def calc_intersection_segments(A1, B1, A2, B2):
    def on_segment(p, q, r):
        """Check if point q lies on line segment pr"""
        if q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and \
           q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]):
            return True
        return False

    def orientation(p, q, r):
        """Find orientation of ordered triplet (p, q, r)
        0 -> p, q and r are collinear
        1 -> Clockwise
        2 -> Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        elif val > 0:
            return 1
        else:
            return 2

    def do_intersect(p1, q1, p2, q2):
        """Check if line segments p1q1 and p2q2 intersect"""
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, q1):
            return True

        if o2 == 0 and on_segment(p1, q2, q1):
            return True

        if o3 == 0 and on_segment(p2, p1, q2):
            return True

        if o4 == 0 and on_segment(p2, q1, q2):
            return True

        return False

    def line_intersection(p1, q1, p2, q2):
        """Calculate the intersection point of two lines"""
        dy1, dx1, prod1 = q1[1] - p1[1], q1[0] - p1[0], q1[0] * p1[1] - q1[1] * p1[0]
        dy2, dx2, prod2 = q2[1] - p2[1], q2[0] - p2[0], q2[0] * p2[1] - q2[1] * p2[0]
        determinant = dx1 * dy2 - dx2 * dy1
        if determinant == 0:
            return [False, np.array([0.0,0.0])] # Lines are parallel
        x = (dx2 * prod1  - dx1 * prod2) / determinant
        y = (dy2 * prod1  - dy1 * prod2) / determinant
        return [True, np.array([x,y])]

    if do_intersect(A1, B1, A2, B2):
        return line_intersection(A1, B1, A2, B2)
    else:
        return [False, np.array([0.0,0.0])]


def check_if_in_triangle(p1, p2, p3, q):
    """
    Check if point q is inside the triangle formed by points p1, p2, and p3.

    Parameters:
    p1, p2, p3: Tuples representing the vertices of the triangle (x, y).
    q: Tuple representing the point to check (x, y).

    Returns:
    bool: True if q is inside the triangle, False otherwise.
    """
    # Extract the coordinates
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x, y = q

    # Compute vectors
    v0 = (x3 - x1, y3 - y1)
    v1 = (x2 - x1, y2 - y1)
    v2 = (x - x1, y - y1)

    # Compute dot products
    dot00 = v0[0] * v0[0] + v0[1] * v0[1]
    dot01 = v0[0] * v1[0] + v0[1] * v1[1]
    dot02 = v0[0] * v2[0] + v0[1] * v2[1]
    dot11 = v1[0] * v1[0] + v1[1] * v1[1]
    dot12 = v1[0] * v2[0] + v1[1] * v2[1]

    # Compute barycentric coordinates
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    # Check if point is in triangle
    return (u >= 0) and (v >= 0) and (u + v < 1)

# check if face is cut
def is_cut(mesh,iel,rho):
    """
    checks if face is intersected by disk with radius rho
    """
    r_min = 1e6
    r_max = 0
    for ino in range(3):
        r = np.linalg.norm(mesh.coords[mesh.elem2node[iel,ino]])
        if (r<=r_min):
            r_min = r
        if (r>=r_max):
            r_max = r
    if (r_min<=rho and rho<=r_max):
        return True

