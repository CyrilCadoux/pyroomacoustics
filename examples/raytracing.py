'''
In this example, we compute the path followed by a ray emitted in an L-shape 2D room
'''

from __future__ import print_function

import scipy
import numpy as np
import matplotlib.pyplot as plt
import time
import pyroomacoustics as pra


#==================== ROOM SETUP ====================
room_ll = [-1,-1]
room_ur = [1,1]
src_pos = [0,0]
mic_pos = [0.5, 0.1]

max_order = 10

# Store the corners of the room floor in an array
pol = 3 * np.array([[0,0], [0,0.75], [2,1], [2,0.5], [1,0.5], [1,0]]).T

# Create the room from its corners
room = pra.Room.from_corners(pol, fs=16000, max_order=max_order, absorption=0.1)

# Add a source somewhere in the room
room.add_source([1.5, 1.2])

#==================== FUNCTIONS ====================


def get_max_distance(room) :
    '''
    Computes the maximum distance that a ray could cover without hitting anything.
    Every ray will be seen as a segment of this length+1, so that it is sure that it hits at least one wall.
    This allows us to use the wall.intersection function to compute the hitting point

    :arg room: the room that is studied
    :returns: a double corresponding to the max distance
    '''

    def get_extreme_xy(room, top_right=True):
        '''
        Combines the coordinates of all wall corners to yield a points that is placed in the extreme
        top right OR bottom left position

        :param room: The room being studied
        :param top_right: Boolean that controls the output.
        :return: The extreme top_right OR bottom left point with format [x y]
        '''
        # [[x_min_wall_0, y_min_wall_0]
        # [x_min_wall_1, y_min_wall_1]...]

        if top_right:
            largest_xy = np.array([np.ndarray.max(w.corners, 1) for w in room.walls])
            return np.ndarray.max(largest_xy, 0)


        smallest_xy = np.array([np.ndarray.min(w.corners, 1) for w in room.walls])
        return np.ndarray.min(smallest_xy,0)

    return scipy.spatial.distance.euclidean(get_extreme_xy(room, False), get_extreme_xy(room)) +1

def compute_end_point(start, length, alpha):
    '''
    Computes the end point of a segment, given its starting point, its angle and its length
    :param start: a 2 dim array defining the starting position
    :param length: the length of the segment
    :param alpha: the angle (rad) of the segment with  respect to the vector [x = 1, y=0] (ie horizontal, pointing to the right)
    :return: a 2 dim array containing the end point of the segment
    '''
    return [start[0] + length*np.cos(alpha), start[1]+length*np.sin(alpha)]

def same_wall(w1,w2):
    '''
    Returns True if both walls are the same
    :param w1: the first wall
    :param w2: the second wall
    :return: True if they are the same, False otherwise
    '''

    c1 = np.array(w1.corners)
    c2 = np.array(w2.corners)

    return sum(sum(abs(c1-c2))) == 0

def next_hit_position(start, end, room, previous_wall):
    '''
    Finds the next wall that will be hit by the ray (represented as a segment here) and outputs the hitting point.
    For non-shoebox rooms, there may be several walls intersected by the ray. In this case we compute the intersection points
    for all those walls and only keep the closest point to the start.
    :param start: a 2 dim array representing the starting point of the ray
    :param end: a 2 dim array representing the end point of the ray. Recall that thanks to get_max_distance, we are sure that there is at least one wall between start and end.
    :param room: the room in which the ray propagates
    :param previous_wall : a wall object representing the last wall that the ray has hit. It is None before the first hit.
    :return: an array with two elements
                - a 2 dim array representing the place where the ray hits the next wall
                - the wall that is going to be hit
    '''

    intersected_walls = []

    # We collect the walls that are intersected and that are not the previous wall
    for w in room.walls:

        different_than_previous = previous_wall is not None and not same_wall(previous_wall, w);
        it_intersects = w.intersects(start, end)[0]

        # Candidate walls for first hit
        if it_intersects and (previous_wall is None or different_than_previous) :
            intersected_walls = intersected_walls + [w]


    # If no wall has been intersected
    if len(intersected_walls) == 0:
        raise ValueError("No wall has been intersected")

    # If only 1 wall is intersected
    if len(intersected_walls) == 1:
        return intersected_walls[0].intersection(start, end)[0], intersected_walls[0]

    # If one reaches this points, it means that several walls have been intersected (non shoebox room)
    intersection_points = [w.intersection(start, end)[0] for w in intersected_walls]
    dist_from_start = np.array([scipy.spatial.distance.euclidean(start, p) for p in intersection_points])

    print("dist_from_start",dist_from_start)


    # Returns the closest point to 'start', ie. the one corresponding to the correct wall
    correct_wall = np.argmin(dist_from_start)
    return intersection_points[correct_wall], intersected_walls[correct_wall]

def get_quadrant(vect):
    '''
    Outputs the quadrant that the vector in parameter belongs to
    :param vect: a 2D vector
    :return: an integer:
                - 1 if the vector (starting from (0,0)) belongs to the first quandrant ([0, pi/2])
                - 2 if the vector (starting from (0,0)) belongs to the second quandrant ([pi/2, pi])
                - 3 if the vector (starting from (0,0)) belongs to the third quandrant ([pi, 3pi/2])
                - 4 if the vector (starting from (0,0)) belongs to the last quandrant ([3pi/2, 2pi])
    '''

    if (vect[0] >= 0):
        if(vect[1] >= 0):
            return 1
        return 4

    if (vect[1] >= 0):
        return 2
    return 3

def equivalent(angle):
    '''
    Returns the equivalent value of the angle in the range [-pi;pi]
    :param angle: an angle in radians
    :return: An equivalent value in [-pi;pi]
    '''

    pi = np.pi

    while(angle > pi):
        angle -= 2*pi

    while(angle < -pi):
        angle += 2*pi

    return angle

def normalize(vector):
    '''
    Returns the unit vector of the vector.
    :param vector: an N dim array representing a vector
    :return: the same vector but with a magnitude of 1
    '''
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    '''
    Returns the angle in radians between vectors 'v1' and 'v2'

    :param v1: an N dim array representing the first vector
    :param v2: an N dim array representing the first vector
    :return: the angle formed by the two vectors. WARNING : the angle is not signed, hence it belongs to [0,pi]
    '''
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def compute_new_angle(start, hit_point, wall_normal, alpha):
    '''
    Computes the new directional angle of the ray when the latter hits a wall
    :param start: a 2 dim array that represents the point that originated the ray before the hit. This point is either the previous hit point, or the source of the sound (for the first iteration).
    :param hit_point: a 2 dim array that represents the intersection between the ray and the wall
    :param wall_normal: a 2 dim array that represents the normal vector of the wall
    :param alpha: The angle of the incident vector with respect to the [1,0] vector
    :return: a new angle (rad) that will give its direction to the ray
    '''


    # The reference vector to compute the angles
    ref_vec = np.array([1,0])
    incident = np.array(hit_point)-np.array(start)

    # We get the quadrant of both vectors
    qi = get_quadrant(incident)
    qn = get_quadrant(wall_normal)

    '''
    =================
    Tricky part here :
    
    There is the extreme case where the normal is purely vertical or horizontal.
        (1) If the normal is vertical, the wall is horizontal and thus we return -alpha
        (2) If the normal is horizontal, the wall is vertical and thus we return pi-alpha
    
    Otherwise, here are the cases where we should work with the inverted version of the wall_normal, since the given normal points to the 'wrong' side of the wall :
        (1) the angle between the reverse incident ray and the normal should be less than pi/2
    
    

    =================
    '''


    # When normal is vertical, ie the wall is horizontal
    if np.dot(np.array([1,0]), wall_normal) == 0:
        return equivalent(-alpha)

    # When normal is horizontal, ie the wall is vertical
    if np.dot(np.array([0,1]), wall_normal) == 0:
        return equivalent(np.pi-alpha)

    reversed_incident= (-1)*incident

    if angle_between(reversed_incident, wall_normal) > np.pi/2:
        wall_normal = (-1) * wall_normal
        qn = get_quadrant(wall_normal)

    # Here we must be careful since angle_between() only yields positive angles
    beta = angle_between(reversed_incident, wall_normal)
    n_alpha = angle_between(ref_vec, wall_normal)

    print("incident", incident)
    print("ref_vec", ref_vec)
    print("normal", wall_normal)
    print("beta", beta * 180 / np.pi)
    print("n_alpha", n_alpha * 180 / np.pi)

    if (qi == 1 and qn == 2): result = equivalent(n_alpha - beta)
    elif (qi == 1 and qn == 3): result = equivalent(-n_alpha + beta)
    elif (qi == 1 and qn == 4): result = equivalent(-n_alpha + beta)

    elif (qi == 2 and qn == 1): result = equivalent(n_alpha + beta)
    elif (qi == 2 and qn == 3): result = equivalent(-n_alpha - beta)
    elif (qi == 2 and qn == 4): result = equivalent(-n_alpha - beta)

    elif (qi == 3 and qn == 1): result = equivalent(n_alpha + beta)
    elif (qi == 3 and qn == 2): result = equivalent(n_alpha + beta)
    elif (qi == 3 and qn == 4): result = equivalent(-n_alpha - beta)

    elif (qi == 4 and qn == 1): result = equivalent(n_alpha - beta)
    elif (qi == 4 and qn == 2): result = equivalent(n_alpha - beta)
    else: result = equivalent(-n_alpha + beta)

    print("----")
    print("qi", qi)
    print("qn", qn)
    return result



#==================== RAY TRACING ====================

#Setup parameters
RAY_SEGMENT_LENGTH = get_max_distance(room)
angle = equivalent(0.18+np.pi/4) # The angle (rad) of the ray with respect to the vector [x = 1, y=0] (ie horizontal, pointing to the right)
start = room.sources[0].position
end = compute_end_point(start, RAY_SEGMENT_LENGTH, angle)
wall = None

room.plot(img_order=6)

iter = 0

while( iter < 10):


    hit_point, wall = next_hit_position(start, end, room, wall)
    plt.plot([hit_point[0],start[0]], [hit_point[1], start[1]], 'ro-')
    print("\n============")
    print("angle", angle*180/np.pi)
    print("start", start)
    print("end", end)
    print("hit_point", hit_point)


    # We use all our information to compute the reflected angle
    angle = compute_new_angle(start,hit_point,wall.normal,angle)

    print("----")
    print("new_angle", angle*180/np.pi)

    # We can now update start and end since we achieved to compute the angle with the previous info
    start = hit_point.copy()
    print("new_start", start)
    end = compute_end_point(start, RAY_SEGMENT_LENGTH, angle)
    print("new_end", end)
    print()

    iter += 1



plt.show()


