import numpy as np
import time as timer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)
import networkx as nx

# The system is as follows
def x1_calc (x, t, input_func):
    return x[0] + input_func*np.cos(x[2])*time_step

def x2_calc (x, t, input_func):
    return x[1] + input_func*np.sin(x[2])*time_step

def x3_calc (x, t, input_func2):
    return x[2] + input_func2*time_step

# ================== first measurement model....
def y1_calc (x):
    return x[0]

def y2_calc (x):
    return x[1]

def y3_calc (x):
    return x[2]

# ================== second measurement model....
def y1_calc_m2 (x):
    return np.sqrt(np.square(x[0])+np.square(x[1]))

def y2_calc_m2 (x):
    return np.tan(x[2])

def read_pgm(pgmf):
    with open( pgmf, 'rb' ) as f:
        """Return a raster of integers from a PGM as a list of lists."""
        header =  f.readline()
        print( header[0], header[1] )
        assert header == b'P5\n'
        while True:
            l = f.readline()
            if not l[0] == 35:   # skip any header comment lines
                break
        (width, height) = [int(i) for i in l.split()]
        depth = int(f.readline())
        assert depth <= 255

        raster = []
        for y in range(height):
            row = []
            for y in range(width):
                row.append(ord(f.read(1)))
            raster.append(row)

    return raster
# =============================================================
# Probabalistic roadmap specific functions....

def dist_obstacle (x):  # inefficient: loops through ALL obstacle points
    _dist = 10  #temporary placeholding value
    x_val = 0.0
    y_val = 0.0
    for row_index, row in enumerate(r_formatted):
        for col_index, elem in enumerate(row):
            if elem == 0:  # this means there is an obstacle there
                obstacle_x = col_index * dx
                #obstacle_y = row_index*dy
                obstacle_y = row_index * dy
                #print("obstacle @: ", (obstacle_x, obstacle_y))
                dist_temp = np.sqrt(np.square(x[0]-obstacle_x)+np.square(x[1]-obstacle_y))
                if dist_temp < _dist:
                    _dist = dist_temp
                    x_val = obstacle_x
                    y_val = obstacle_y
    return _dist, (x_val, y_val)  # return distance and point

def controller_carrot (state1, theta,  state2):  # where x = x[0], x[1] = x, y
    v_input = 1.0
    w_input = 0.0
    theta_ = theta
    turn_amplification = 1.0
    angle = np.arctan2((state2[1]-state1[1]), (state2[0]-state1[0]))
    if (angle - theta) > 0.0:  # turn ccw
        w_input = abs(angle - theta) * turn_amplification
        v_input = 0.1
    elif (angle - theta) < 0.0:  # turn cw
        w_input = -abs(angle - theta) * turn_amplification
        v_input = 0.1
    return v_input, w_input

def check_path_found (edges, start_pose, goal_pose):  # DEPRECATED
    print(">>>>> Check if paths are found...")
    init_list = np.array([0.0, 0.0])
    point_list = np.array([0.0, 0.0])
    print("start pose and goal pose: ", start_pose, ", ", goal_pose)
    init_list = np.vstack((init_list, [start_pose[0], start_pose[1]]))
    bool = False
    edge_found_count = 1  # temporary...
    # if we keep finding new routes to new points using edges, keep looking!
    while edge_found_count > 0:
        for edge in edges:  # find next set of points
            edge_found_count = 0
            for init_point in init_list[1:]:
                if edge[0].item() == init_point[0].item() and edge[1].item() == init_point[1].item():
                    point_list = np.vstack((point_list, [edge[2], edge[3]]))
                    edge_found_count = edge_found_count + 1
        if edge_found_count == 0:
            print("no paths from: ", init_point)
            return False
        # check if one of the edges has led us to the goal!
        for point in point_list[1:]:  # check is any found point = goal
            if point[0] == goal_pose[0] and point[1] == goal_pose[1]:
                # we reached the goal pose!
                print("Goal Reached!!!!")
                return True
        # if we went through the point list, and none of them are the goal...
        # move a layer forward in terms of depth...
        init_list = point_list
        point_list = np.array([0.0, 0.0])
        print("move a layer forward in terms of node depth...")
        # if edges were found, the point list is NOT empty, loop this process
        # until we reach the goal!
        # if No edges were found, the point list IS empty -> return False
    return bool

k_closest = 5  # arbitrarily set
# K closest nodes from a single node, that are valid
def k_closest_nodes_2 (nodes_x, nodes_y, x, y):
    # the most recent node will be at the end of the nodes array
    #print(">>>> K CLOSEST CODE ")
    k_ = k_closest
    points_found = 0
    #print("point in question: ", (x, y))
    x_point = x
    y_point = y
    nodes_x_ = nodes_x
    nodes_y_ = nodes_y
    #print("nodes to search through: ", len(nodes_x_))
    x_closest = 0.0
    y_closest = 0.0
    closest_nodes = np.array([0.0, 0.0])
    if len(nodes_x_) < k_:  # if there arnt even K nodes yet
        k_ = len(nodes_x_)
    # while we have more points to find and we still have nodes
    while points_found < k_ and len(nodes_x_) > 0:
        distance_ = 10000.0  # simply an initial placeholder
        for index in range(len(nodes_x_)):  # remember we already chopped the zero'th term and last term
            dist_temp = np.sqrt(np.square(x_point - nodes_x_[index]) + np.square(y_point - nodes_y_[index]))
            #print("Index and distance: ", index, dist_temp)
            if dist_temp < distance_:
                distance_ = dist_temp
                x_closest = nodes_x_[index]
                y_closest = nodes_y_[index]
                closest_index = index
        # Now we have the closest node!
        # lets check if the edge formed is valid
        boo = bresenham_collisions_single(np.array([x_point, y_point, x_closest, y_closest]))
        if boo:
            closest_nodes = np.vstack((closest_nodes, [x_closest, y_closest]))
            # now remove this node from the array... (by index)
            #print("delete node with index: ", closest_index)
            nodes_x_ = np.delete(nodes_x_, closest_index)
            nodes_y_ = np.delete(nodes_y_, closest_index)
            points_found = points_found + 1
        else:
            # that edge is not valid! Do not add, just delete
            nodes_x_ = np.delete(nodes_x_, closest_index)
            nodes_y_ = np.delete(nodes_y_, closest_index)

    return closest_nodes[1:]  # returns v-stacked array of closest nodes


# distance if less than, then consider it a collision
dist_tol = 0.1  # from an obstacle
dist_tol_fine = 0.05
def bresenham_collisions_single (edge):  # no collision returns False
    #print(">>> Collision checker with Bresenhams (single)")
    # this applies Bresenham's line algorithm to discretize the 'edge'
    dx_ = 0.0
    dy_ = 0.0
    #print("edge to check: ", edge)
    x1 = edge[0]
    y1 = edge[1]
    x2 = edge[2]
    y2 = edge[3]
    if abs(y2 - y1) <= abs(x2 - x1):
        if x2>x1:  # RIGHT HAND SIDE
            dx_ = 0.05  # dx becomes the increment
            dy_ = ((y2-y1)/(x2-x1))*dx_
        elif x2<x1:  # LEFT HAND SIDE
            dx_ = -0.05  # dx becomes the increment
            dy_ = ((y2-y1)/(x2-x1))*dx_
    else:
        if y2>y1:  # UPWARDS
            dy_ = 0.05  # other wise dy becomes the increment
            dx_ = ((x2-x1)/(y2-y1))*dy_
        elif y2<y1:  # DOWNWARDS
            dy_ = -0.05  # other wise dy becomes the increment
            dx_ = ((x2-x1)/(y2-y1))*dy_
    # now move along the line and ensure that there are no collisions!
    while abs(x2 - x1) + abs(y2 - y1) > dist_tol:
        #print("point to check: ", (x1, y1))
        #print("approaching... : ", (x2, y2))
        dist_, (dist_x, dist_y) = dist_obstacle(np.array([x1, y1]))
        #print("==> distance from obstacle: ", dist_)
        #print("==> distance between points: ", abs(x2 - x1) + abs(y2 - y1))
        if dist_ <= dist_tol:
            #print("edge not valid: collision detected")
            return True
        else:
            x1_temp = x1 + dx_
            y1_temp = y1 + dy_
            x1 = x1_temp
            y1 = y1_temp
            #print("moving along edge: ", (x1, y1))
    if abs(x2 - x1) + abs(y2 - y1) <= dist_tol:
        #print("Made it without a collision! The edge is valid!")
        return False

def graph_to_nodes (graph_object):
    nodes_array = np.array([0.0, 0.0])
    nodes_list = list(graph_object.nodes())  # ['5.6,7.8', '2.1,4.9', ...]
    for element in nodes_list:
        temp_elem = element.split(',')
        x_p_val = float(temp_elem[0])
        y_p_val = float(temp_elem[1])
        nodes_array = np.vstack((nodes_array, [x_p_val, y_p_val]))
    return nodes_array[1:]

def add_to_closest_graph(x_interp_, y_interp_, x_closest, y_closest,
                         G_i_1, G_1_i, G_1_2, G_2_1, G_2_3, G_3_2, G_3_4, G_4_3,
                         bool_pos1, bool_pos2, bool_pos3, bool_pos4):
    # search the graphs for the node, we know it exists!
    # only search a graph if its path hasnt been found yet
    # add the new node and new edge connecting to it
    x_interp = round(x_interp_, 2)
    y_interp = round(y_interp_, 2)
    if not bool_pos1:
        G_i_1_array = graph_to_nodes(G_i_1)
        #print("G_i_1: ", list(G_i_1.nodes()))
        #print("array is: ", G_i_1_array.shape[0])
        #print("first x_val: ", G_i_1_array[0][0])
        for i in range(G_i_1_array.shape[0]):
            if G_i_1_array[i][0] == x_closest and G_i_1_array[i][1] == y_closest:
                # add node and edge!
                string_point = '{},{}'.format(x_closest, y_closest)
                string_interp = '{},{}'.format(x_interp, y_interp)
                G_i_1.add_node(string_interp)
                G_i_1.add_edge(string_point, string_interp)
                print("... point added to G_i_1")
                return
        G_1_i_array = graph_to_nodes(G_1_i)
        for i in range(G_1_i_array.shape[0]):
            if G_1_i_array[i][0] == x_closest and G_1_i_array[i][1] == y_closest:
                # add node and edge!
                string_point = '{},{}'.format(x_closest, y_closest)
                string_interp = '{},{}'.format(x_interp, y_interp)
                G_1_i.add_node(string_interp)
                G_1_i.add_edge(string_point, string_interp)
                print("... point added to G_1_i")
                return
    if not bool_pos2:
        G_1_2_array = graph_to_nodes(G_1_2)
        for i in range(G_1_2_array.shape[0]):
            if G_1_2_array[i][0] == x_closest and G_1_2_array[i][1] == y_closest:
                # add node and edge!
                string_point = '{},{}'.format(x_closest, y_closest)
                string_interp = '{},{}'.format(x_interp, y_interp)
                G_1_2.add_node(string_interp)
                G_1_2.add_edge(string_point, string_interp)
                print("... point added to G_1_2")
                return
        G_2_1_array = graph_to_nodes(G_2_1)
        for i in range(G_2_1_array.shape[0]):
            if G_2_1_array[i][0] == x_closest and G_2_1_array[i][1] == y_closest:
                # add node and edge!
                string_point = '{},{}'.format(x_closest, y_closest)
                string_interp = '{},{}'.format(x_interp, y_interp)
                G_2_1.add_node(string_interp)
                G_2_1.add_edge(string_point, string_interp)
                print("... point added to G_2_1")
                return
    if not bool_pos3:
        G_2_3_array = graph_to_nodes(G_2_3)
        for i in range(G_2_3_array.shape[0]):
            if G_2_3_array[i][0] == x_closest and G_2_3_array[i][1] == y_closest:
                # add node and edge!
                string_point = '{},{}'.format(x_closest, y_closest)
                string_interp = '{},{}'.format(x_interp, y_interp)
                G_2_3.add_node(string_interp)
                G_2_3.add_edge(string_point, string_interp)
                print("... point added to G_i_1")
                return
        G_3_2_array = graph_to_nodes(G_3_2)
        for i in range(G_3_2_array.shape[0]):
            if G_3_2_array[i][0] == x_closest and G_3_2_array[i][1] == y_closest:
                # add node and edge!
                string_point = '{},{}'.format(x_closest, y_closest)
                string_interp = '{},{}'.format(x_interp, y_interp)
                G_3_2.add_node(string_interp)
                G_3_2.add_edge(string_point, string_interp)
                print("... point added to G_3_2")
                return
    if not bool_pos4:
        G_3_4_array = graph_to_nodes(G_3_4)
        for i in range(G_3_4_array.shape[0]):
            if G_3_4_array[i][0] == x_closest and G_3_4_array[i][1] == y_closest:
                # add node and edge!
                string_point = '{},{}'.format(x_closest, y_closest)
                string_interp = '{},{}'.format(x_interp, y_interp)
                G_3_4.add_node(string_interp)
                G_3_4.add_edge(string_point, string_interp)
                print("... point added to G_3_4")
                return
        G_4_3_array = graph_to_nodes(G_4_3)
        for i in range(G_4_3_array.shape[0]):
            if G_4_3_array[i][0] == x_closest and G_4_3_array[i][1] == y_closest:
                # add node and edge!
                string_point = '{},{}'.format(x_closest, y_closest)
                string_interp = '{},{}'.format(x_interp, y_interp)
                G_4_3.add_node(string_interp)
                G_4_3.add_edge(string_point, string_interp)
                print("... point added to G_4_3")
                return

# =========================================

# =================================================

baseline = 0.45
time_step = 0.1
dx = dy = 0.1

# this r = MAP; which is 100X100
r = read_pgm( 'sim_map.pgm' )
r_formatted = r.copy()
print("rows:", len(r[0][:]))  # note that Y=rows are flipped
print("cols:", len(r[:][0]))

# re-mapping to something more intuitive
for row_index, row in enumerate(r):  # row is for Y (inverted)
    for col_index, elem in enumerate(row):  # col is for X
        if elem == 0:  # this means there is an obstacle there
            r_formatted[99 - row_index][col_index] = 0.0
        else:
            r_formatted[99 - row_index][col_index] = 1.0

loop_count = 0.0
x = np.array([5*dx, 95*dy, 0.0])  # this is the starting pose
# store values (plot path later)
state_array_x = np.array([0.0])
state_array_y = np.array([0.0])
state_array_theta = np.array([0.0])

# ===========================================
object_clearance_tol = 0.1
node_clearance_tol = 0.1
d_tree = 0.5
graph_connect_tol = 0.5

# as long as we have not yet formed paths....
bool_pos1 = False
bool_pos2 = False
bool_pos3 = False
bool_pos4 = False

# lets also set up some real time plotting
fig1 = plt.figure()
fig1.canvas.draw()
plt.show(block=False)

# now to set up the poses to vist on the ROUTE
# NOTE: the goal orientation angle (optional) is ignored
init_pos = np.array([0.5, 9.5])
pos1 = np.array([7.0, 1.5])
pos2 = np.array([9.0, 5.0])
pos3 = np.array([3.0, 9.5])
pos4 = np.array([0.5, 5.0])
# let us create and populate out graph structure(s)
# there will be many graph structures:
# 1) init pos to pos1
# 2) pos1 to init pos
# 3) pos1 to pos2
# 4) pos2 to pos1
# 5) pos2 to pos3
# 6) pos3 to pos2
# 7) pos3 to pos4
# 8) pos4 to pos3
print("============== NetworkX: creating graph structures ....")
G_i_1 = nx.Graph()
string = '{},{}'.format(init_pos[0], init_pos[1])
G_i_1.add_node(string, pos=(init_pos[0], init_pos[1]))
# string = '{},{}'.format(init_pos[0], init_pos[0])
# G_i_1.add_node(string, pos=(init_pos[0], init_pos[0]))
# string = '{},{}'.format(init_pos[1], init_pos[1])
# G_i_1.add_node(string, pos=(init_pos[1], init_pos[1]))

G_1_i = nx.Graph()
string = '{},{}'.format(pos1[0], pos1[1])
G_1_i.add_node(string, pos=(pos1[0], pos1[1]))

G_1_2 = nx.Graph()
string = '{},{}'.format(pos1[0], pos1[1])
G_1_2.add_node(string, pos=(pos1[0], pos1[1]))

G_2_1 = nx.Graph()
string = '{},{}'.format(pos2[0], pos2[1])
G_2_1.add_node(string, pos=(pos2[0], pos2[1]))

G_2_3 = nx.Graph()
string = '{},{}'.format(pos2[0], pos2[1])
G_2_3.add_node(string, pos=(pos2[0], pos2[1]))

G_3_2 = nx.Graph()
string = '{},{}'.format(pos3[0], pos3[1])
G_3_2.add_node(string, pos=(pos3[0], pos3[1]))

G_3_4 = nx.Graph()
string = '{},{}'.format(pos3[0], pos3[1])
G_3_4.add_node(string, pos=(pos3[0], pos3[1]))

G_4_3 = nx.Graph()
string = '{},{}'.format(pos4[0], pos4[1])
G_4_3.add_node(string, pos=(pos4[0], pos4[1]))

# The technique used is single query for each path individually
# now solving our RRT paths in a loop
while loop_count <= 1000:  # break in code included
    print("Begin loop >>>>>>>>>>>>>>")
    # first extract all nodes from graphs....
    all_nodes_array = np.array([0.0, 0.0])
    # if we found the path already no need to consider those graphs
    if not bool_pos1:
        nodes_array_1 = graph_to_nodes(G_i_1)
        nodes_array_2 = graph_to_nodes(G_1_i)
        all_nodes_array = np.vstack((all_nodes_array, nodes_array_1))
        all_nodes_array = np.vstack((all_nodes_array, nodes_array_2))
    if not bool_pos2:
        nodes_array_3 = graph_to_nodes(G_1_2)
        nodes_array_4 = graph_to_nodes(G_2_1)
        all_nodes_array = np.vstack((all_nodes_array, nodes_array_3))
        all_nodes_array = np.vstack((all_nodes_array, nodes_array_4))
    if not bool_pos3:
        nodes_array_5 = graph_to_nodes(G_2_3)
        nodes_array_6 = graph_to_nodes(G_3_2)
        all_nodes_array = np.vstack((all_nodes_array, nodes_array_5))
        all_nodes_array = np.vstack((all_nodes_array, nodes_array_6))
    if not bool_pos4:
        nodes_array_7 = graph_to_nodes(G_3_4)
        nodes_array_8 = graph_to_nodes(G_4_3)
        all_nodes_array = np.vstack((all_nodes_array, nodes_array_7))
        all_nodes_array = np.vstack((all_nodes_array, nodes_array_8))

    all_nodes_array_final = all_nodes_array[1:]
    print("All nodes: ", all_nodes_array_final)
    print("shape of all nodes: ", all_nodes_array_final.shape)
    # now add a random point and see if it collides
    obj_distance = 0.0
    node_distance = 1000.0
    interp_distance = 1000.0
    collision = True
    # keep trying if random point isL
    # 1) too close to another node
    # 2) too close/ colliding with an object (point or path)
    #    after moving d_tree distance away from closest node on any tree
    while obj_distance < object_clearance_tol or node_distance < node_clearance_tol or interp_distance > d_tree or collision:
        print(">>>>>>> A new random point")
        # as long as x_rand is close to colliding (<0.1 from obstacle), keep trying...
        # rand() returns random numb. from 0 to 1
        rand_x = round(abs(np.random.rand())*10.0, 2)
        rand_y = round(abs(np.random.rand())*10.0, 2)
        x_rand = np.array([rand_x, rand_y])
        print("random point: ", x_rand)
        node_distance = 1000.0
        # now ensure that we are a distance 0.1 from all other nodes
        closest_node_index = 0
        print("all nodes array shape: ", all_nodes_array_final.shape)
        for i in range(all_nodes_array_final.shape[0]):  # solves the closest node
            node_distance_temp = np.sqrt(np.square(x_rand[0]-all_nodes_array_final[i][0])+np.square(x_rand[1]-all_nodes_array_final[i][1]))
            if node_distance_temp < node_distance:
                node_distance = node_distance_temp
                closest_node_index = i
        print("random point distance from closest node: ", node_distance)
        print("and this closest node has index: ", closest_node_index, " and coord's: ", (all_nodes_array_final[closest_node_index][0], all_nodes_array_final[closest_node_index][1]))
        # if the node distance if greater than d_tree, interpolate between the points
        # done by moving rand point progressively closer until dist <= d_tree !!
        x_interp = rand_x
        y_interp = rand_y
        x_closest = all_nodes_array_final[closest_node_index][0]
        y_closest = all_nodes_array_final[closest_node_index][1]
        if x_interp == x_closest:
            print("try another random point... (equal x values)")
            continue  # to avoid error when dividing by zero (just try another random point)
        gradient = (y_interp - y_closest) / (x_interp - x_closest)
        if abs(y_interp - y_closest) <= abs(x_interp - x_closest):
            if x_interp > x_closest:
                dx_grad = -0.01
                dy_grad = dx_grad * gradient
            else:
                dx_grad = 0.01
                dy_grad = dx_grad * gradient
        else:
            if y_interp > y_closest:
                dy_grad = -0.01
                dx_grad = dy_grad * (1 / gradient)
            else:
                dy_grad = 0.01
                dx_grad = dy_grad * (1 / gradient)
        interp_distance = np.sqrt(np.square(x_interp - all_nodes_array_final[closest_node_index][0])
                                  + np.square(y_interp - all_nodes_array_final[closest_node_index][1]))
        print("the initial interpolated distance: ", interp_distance)
        print("the interpolated point: ", (x_interp, y_interp))
        while interp_distance > d_tree:
            x_interp = x_interp + dx_grad
            y_interp = y_interp + dy_grad
            interp_distance = np.sqrt(np.square(x_interp - all_nodes_array_final[closest_node_index][0])
                                      + np.square(y_interp - all_nodes_array_final[closest_node_index][1]))

        # get the final interpolated point
        print("final interpolated point: ", (x_interp, y_interp))
        print("distance from closest node: ", interp_distance)
        # now we can FINALLY check for collision and either make the collision boolean False
        # or keep it True (if there is a collision)
        collision = bresenham_collisions_single(np.array([x_interp, y_interp, x_closest, y_closest]))
        print("collision?: ", collision)
        # and the distance from an object?
        obj_distance, (x_dist, y_dist) = dist_obstacle(np.array([x_interp, y_interp]))
        print("interp point distance (from object): ", obj_distance)
        # ===============================================================
    # The point is now ready to be added to the closest graph
    # the point to be added is x_interp, y_interp
    # the closest point on the graph is x_closest, y_closest
    add_to_closest_graph(x_interp, y_interp, x_closest, y_closest,
                         G_i_1, G_1_i, G_1_2, G_2_1, G_2_3, G_3_2, G_3_4, G_4_3,
                         bool_pos1, bool_pos2, bool_pos3, bool_pos4)

    # now we try to connect the graphs
    # (if their nodes are less than a distance apart and there are no collisions)
    # first find the closest points between the 2 graphs
    graph_connect_tol = 0.5
    # ===================== for path 1
    if not bool_pos1:
        print("Attempting to connect graphs for PATH 1 ...")
        G_i_1_array = graph_to_nodes(G_i_1)
        G_1_i_array = graph_to_nodes(G_1_i)
        graph_dist = 1000.0  # just a temporary placeholder
        # find closest points between graphs
        for i in range(G_i_1_array.shape[0]):
            for j in range(G_1_i_array.shape[0]):
                graph_dist_temp = np.sqrt(np.square(G_i_1_array[i][0] - G_1_i_array[j][0]) + np.square(G_i_1_array[i][1] - G_1_i_array[j][1]))
                if graph_dist_temp < graph_dist:
                    graph_dist = graph_dist_temp
                    point_1_x = G_i_1_array[i][0]
                    point_1_y = G_i_1_array[i][1]
                    point_2_x = G_1_i_array[j][0]
                    point_2_y = G_1_i_array[j][1]
        print("distance btw graphs: ", graph_dist)
        # connect the closest points if they are reasonably close and do not have collisions between them
        if not bresenham_collisions_single(np.array([point_1_x, point_1_y, point_2_x, point_2_y])):
            print("there is no collision between the points on either graph!")
            if graph_dist < graph_connect_tol:
                print(" and they are close enough to connect! dist: ", graph_dist)
                string_1 = '{},{}'.format(point_1_x, point_1_y)
                string_2 = '{},{}'.format(point_2_x, point_2_y)
                G_i_1.add_edge(string_1, string_2)
                # the path is found!
                bool_pos1 = True
                # and merge these graphs!
                G_path_1 = nx.compose(G_i_1, G_1_i)
                # bool_pos1 = nx.has_path(G_path_1, init_string, pos1_string)
    # ===================== for path 2
    if not bool_pos2:
        print("Attempting to connect graphs for PATH 2 ...")
        G_1_2_array = graph_to_nodes(G_1_2)
        G_2_1_array = graph_to_nodes(G_2_1)
        graph_dist = 1000.0  # just a temporary placeholder
        # find closest points between graphs
        for i in range(G_1_2_array.shape[0]):
            for j in range(G_2_1_array.shape[0]):
                graph_dist_temp = np.sqrt(np.square(G_1_2_array[i][0] - G_2_1_array[j][0]) + np.square(G_1_2_array[i][1] - G_2_1_array[j][1]))
                if graph_dist_temp < graph_dist:
                    graph_dist = graph_dist_temp
                    point_1_x = G_1_2_array[i][0]
                    point_1_y = G_1_2_array[i][1]
                    point_2_x = G_2_1_array[j][0]
                    point_2_y = G_2_1_array[j][1]
        print("distance btw graphs: ", graph_dist)
        # connect the closest points if they are reasonably close and do not have collisions between them
        if not bresenham_collisions_single(np.array([point_1_x, point_1_y, point_2_x, point_2_y])):
            print("there is no collision between the points on either graph!")
            if graph_dist < graph_connect_tol:
                print(" and they are close enough to connect! dist: ", graph_dist)
                string_1 = '{},{}'.format(point_1_x, point_1_y)
                string_2 = '{},{}'.format(point_2_x, point_2_y)
                G_1_2.add_edge(string_1, string_2)
                # the path is found!
                bool_pos2 = True
                # and merge these graphs!
                G_path_2 = nx.compose(G_1_2, G_2_1)
    # ===================== for path 3
    if not bool_pos3:
        print("Attempting to connect graphs for PATH 3 ...")
        G_2_3_array = graph_to_nodes(G_2_3)
        G_3_2_array = graph_to_nodes(G_3_2)
        graph_dist = 1000.0  # just a temporary placeholder
        # find closest points between graphs
        for i in range(G_2_3_array.shape[0]):
            for j in range(G_3_2_array.shape[0]):
                graph_dist_temp = np.sqrt(np.square(G_2_3_array[i][0] - G_3_2_array[j][0]) + np.square(G_2_3_array[i][1] - G_3_2_array[j][1]))
                if graph_dist_temp < graph_dist:
                    graph_dist = graph_dist_temp
                    point_1_x = G_2_3_array[i][0]
                    point_1_y = G_2_3_array[i][1]
                    point_2_x = G_3_2_array[j][0]
                    point_2_y = G_3_2_array[j][1]
        print("distance btw graphs: ", graph_dist)
        # connect the closest points if they are reasonably close and do not have collisions between them
        if not bresenham_collisions_single(np.array([point_1_x, point_1_y, point_2_x, point_2_y])):
            print("there is no collision between the points on either graph!")
            if graph_dist < graph_connect_tol:
                print(" and they are close enough to connect! dist: ", graph_dist)
                string_1 = '{},{}'.format(point_1_x, point_1_y)
                string_2 = '{},{}'.format(point_2_x, point_2_y)
                G_2_3.add_edge(string_1, string_2)
                # the path is found!
                bool_pos3 = True
                # and merge these graphs!
                G_path_3 = nx.compose(G_2_3, G_3_2)
    # ===================== for path 4
    if not bool_pos4:
        print("Attempting to connect graphs for PATH 4 ...")
        G_3_4_array = graph_to_nodes(G_3_4)
        G_4_3_array = graph_to_nodes(G_4_3)
        graph_dist = 1000.0  # just a temporary placeholder
        # find closest points between graphs
        for i in range(G_3_4_array.shape[0]):
            for j in range(G_4_3_array.shape[0]):
                graph_dist_temp = np.sqrt(np.square(G_3_4_array[i][0] - G_4_3_array[j][0]) + np.square(G_3_4_array[i][1] - G_4_3_array[j][1]))
                if graph_dist_temp < graph_dist:
                    graph_dist = graph_dist_temp
                    point_1_x = G_3_4_array[i][0]
                    point_1_y = G_3_4_array[i][1]
                    point_2_x = G_4_3_array[j][0]
                    point_2_y = G_4_3_array[j][1]
        print("distance btw graphs: ", graph_dist)
        # connect the closest points if they are reasonably close and do not have collisions between them
        if not bresenham_collisions_single(np.array([point_1_x, point_1_y, point_2_x, point_2_y])):
            print("there is no collision between the points on either graph!")
            if graph_dist < graph_connect_tol:
                print(" and they are close enough to connect! dist: ", graph_dist)
                string_1 = '{},{}'.format(point_1_x, point_1_y)
                string_2 = '{},{}'.format(point_2_x, point_2_y)
                G_3_4.add_edge(string_1, string_2)
                # the path is found!
                bool_pos4 = True
                # and merge these graphs!
                G_path_4 = nx.compose(G_3_4, G_4_3)

    print("All graphs:")
    print("G_i_1 nodes: ", list(G_i_1.nodes()))
    print("G_i_1 edges: ", list(G_i_1.edges()))
    print("G_1_i nodes: ", list(G_1_i.nodes()))
    print("G_1_i edges: ", list(G_1_i.edges()))
    print("G_1_2 nodes: ", list(G_1_2.nodes()))
    print("G_1_2 edges: ", list(G_1_2.edges()))
    print("G_2_1 nodes: ", list(G_2_1.nodes()))
    print("G_2_1 edges: ", list(G_2_1.edges()))
    print("G_2_3 nodes: ", list(G_2_3.nodes()))
    print("G_2_3 edges: ", list(G_2_3.edges()))
    print("G_3_2 nodes: ", list(G_3_2.nodes()))
    print("G_3_2 edges: ", list(G_3_2.edges()))
    print("G_3_4 nodes: ", list(G_3_4.nodes()))
    print("G_3_4 edges: ", list(G_3_4.edges()))
    print("G_4_3 nodes: ", list(G_4_3.nodes()))
    print("G_4_3 edges: ", list(G_4_3.edges()))

    # ===============================================
    loop_count = loop_count + 1
    print(">>>>>>>>>> Loop Completed || Loop Count: ", loop_count)

    # plot found nodes and edges (real time)....
    print("now plotting...")
    G_i_1_array = graph_to_nodes(G_i_1)
    G_1_i_array = graph_to_nodes(G_1_i)
    G_1_2_array = graph_to_nodes(G_1_2)
    G_2_1_array = graph_to_nodes(G_2_1)
    G_2_3_array = graph_to_nodes(G_2_3)
    G_3_2_array = graph_to_nodes(G_3_2)
    G_3_4_array = graph_to_nodes(G_3_4)
    G_4_3_array = graph_to_nodes(G_4_3)
    final_graphs_array = np.vstack((G_i_1_array, G_1_i_array))
    final_graphs_array = np.vstack((final_graphs_array, G_1_2_array))
    final_graphs_array = np.vstack((final_graphs_array, G_2_1_array))
    final_graphs_array = np.vstack((final_graphs_array, G_2_3_array))
    final_graphs_array = np.vstack((final_graphs_array, G_3_2_array))
    final_graphs_array = np.vstack((final_graphs_array, G_3_4_array))
    final_graphs_array = np.vstack((final_graphs_array, G_4_3_array))

    for index in range(final_graphs_array.shape[0]):
        plt.scatter(final_graphs_array[index][0], final_graphs_array[index][1], alpha=0.8, edgecolors='none', s=30, color='blue')
    # also plot gridworld obstacles
    # plot the obstacles ie) gridworld
    for row_index, row in enumerate(r_formatted):  # row is for Y
        for col_index, elem in enumerate(row):  # col is for X
            if elem == 0.0:  # this means there is an obstacle there
                plt.scatter((col_index) * dx, (row_index) * dy, alpha=0.8, edgecolors='none', s=30, color='black')
    # lets draw the edges as well.....


    plt.title('RRT NODE/ EDGE MAP')
    plt.legend()
    axes = plt.gca()
    axes.set_xticks(np.arange(0, 10, dx))
    axes.set_yticks(np.arange(0, 10, dy))
    axes.set_xlim([0.0, 10])
    axes.set_ylim([0.0, 10])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    fig1.canvas.draw()

    timer.sleep(0.1)
    plt.clf()

    # using NETWORKX - GRAPH BASED PATH PLANNING
    # ===================================

    # see if the paths exist...
    init_string = '{},{}'.format(init_pos[0], init_pos[1])
    pos1_string = '{},{}'.format(pos1[0], pos1[1])
    pos2_string = '{},{}'.format(pos2[0], pos2[1])
    pos3_string = '{},{}'.format(pos3[0], pos3[1])
    pos4_string = '{},{}'.format(pos4[0], pos4[1])

    print("do we have paths?....")
    print("for pos1: ", bool_pos1)
    print("for pos2: ", bool_pos2)
    print("for pos3: ", bool_pos3)
    print("for pos4: ", bool_pos4)
    if bool_pos1:
        print("===> init to POS1 Solved")
        print("Shortest Path: ", nx.shortest_path(G_path_1, init_string, pos1_string))
    if bool_pos2:
        print("===> POS1 to POS2 Solved")
        print("Shortest Path: ", nx.shortest_path(G_path_2, pos1_string, pos2_string))
    if bool_pos3:
        print("===> POS2 to POS3 Solved")
        print("Shortest Path: ", nx.shortest_path(G_path_3, pos2_string, pos3_string))
    if bool_pos4:
        print("===> POS3 to POS4 Solved")
        print("Shortest Path: ", nx.shortest_path(G_path_4, pos3_string, pos4_string))
    if bool_pos1 and bool_pos2 and bool_pos3 and bool_pos4:
        print(">>>>>>>>>>>>>>>>>>>> ALL PATH(s) FOUND! <<<<<<<<<<<<<<<<")
        # save all data to .csv
        # .........................................
        break
    else:
        print("Paths NOT found...yet")


print(">>>>>>>>>>>>>> END of RRT SOLVER")
# =========================================================


# now lets plot the node/ edge roadmap!
fig2 = plt.figure()
for row_index, row in enumerate(r_formatted):  # row is for Y
    for col_index, elem in enumerate(row):  # col is for X
        if elem == 0.0:  # this means there is an obstacle there
            plt.scatter((col_index) * dx, (row_index)*dy, alpha=0.8, edgecolors='none', s=30, color='black')
# now to plot all the nodes
# note: range() starts at zero
for index in range(len(nodes_x[1:])):
    plt.scatter(nodes_x[index+1], nodes_y[index+1], alpha=0.8, edgecolors='none', s=30, color='blue')
plt.scatter((5)*dx, (95)*dy, alpha=1.0, edgecolors='none', s=50, color='green')
# and also include all the intermediary goal positions
plt.scatter(pos1[0], pos1[1], alpha=1.0, edgecolors='none', s=50, color='red')
plt.scatter(pos2[0], pos2[1], alpha=1.0, edgecolors='none', s=50, color='red')
plt.scatter(pos3[0], pos3[1], alpha=1.0, edgecolors='none', s=50, color='red')
plt.scatter(pos4[0], pos4[1], alpha=1.0, edgecolors='none', s=50, color='red')
# lets plot the edges as well (check)
for edge_p in edges[1:]:
    plt.plot([edge_p[0], edge_p[2]], [edge_p[1], edge_p[3]])
plt.title('Final NODE/ EDGE MAP')
plt.legend()
axes = plt.gca()
axes.set_xticks(np.arange(0, 10, dx))
axes.set_yticks(np.arange(0, 10, dy))
axes.set_xlim([0.0, 10])
axes.set_ylim([0.0, 10])
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
#plt.show()

# Now we simulate the robots motion
init_string = '[{},{}]'.format(init_pos[0], init_pos[1])
pos1_string = '[{},{}]'.format(pos1[0], pos1[1])
pos2_string = '[{},{}]'.format(pos2[0], pos2[1])
pos3_string = '[{},{}]'.format(pos3[0], pos3[1])
pos4_string = '[{},{}]'.format(pos4[0], pos4[1])
# first acquire all the paths
path_1 = nx.shortest_path(G, init_string, pos1_string)
path_2 = nx.shortest_path(G, pos1_string, pos2_string)
path_3 = nx.shortest_path(G, pos2_string, pos3_string)
path_4 = nx.shortest_path(G, pos3_string, pos4_string)
# now extracting the sequence of position values in each path
path_array = np.array([0.0, 0.0])
for element in path_1:  # element looks like: [3.0,9.5]
    x_p_val = float(element[1:4])
    y_p_val = float(element[5:8])
    path_array = np.vstack((path_array, [x_p_val, y_p_val]))
for element in path_2[1:]:  # chopped as path1 includes first point of path2
    x_p_val = float(element[1:4])
    y_p_val = float(element[5:8])
    path_array = np.vstack((path_array, [x_p_val, y_p_val]))
for element in path_3[1:]:
    x_p_val = float(element[1:4])
    y_p_val = float(element[5:8])
    path_array = np.vstack((path_array, [x_p_val, y_p_val]))
for element in path_4[1:]:
    x_p_val = float(element[1:4])
    y_p_val = float(element[5:8])
    path_array = np.vstack((path_array, [x_p_val, y_p_val]))
final_path_array = path_array[1:]
# =================================================================== SIMULATION

time = 0.0
theta = 0.0  # this is taken arbitrarily (as in Q1)
# now simulate the robots path
path_index = 0
# what is the max path index?....
max_path_index = final_path_array.shape[0]  # number of rows
print("max path index: ", max_path_index)
while path_index < max_path_index:
    # ================================
    state = final_path_array[path_index]
    print("Current state: ", state)
    # find control inputs to take up to the next state
    v_input, w_input = controller_carrot(state, theta, final_path_array[path_index + 1])
    print("chosen controller inputs (v, w): ", (v_input, w_input))
    robot_pos_new = np.array([x1_calc(x, time, v_input),
                              x2_calc(x, time, v_input),
                              x3_calc(x, time, w_input)])
    print("Next robot state: ", (robot_pos_new[0], robot_pos_new[1], robot_pos_new[2]))
    # store the data!
    state_array_x = np.append(state_array_x, x[0])
    state_array_y = np.append(state_array_y, x[1])
    state_array_theta = np.append(state_array_theta, x[2])
    # now increment time and carry forward the data
    x[0] = robot_pos_new[0]
    x[1] = robot_pos_new[1]
    x[2] = robot_pos_new[2]
    time = time + time_step

    # if we are within 0.5m of the next point this is sufficient!
    path_dist = np.sqrt(np.square(state[0] - final_path_array[index + 1][0]) + np.square(state[1] - final_path_array[index + 1][1]))
    print("distance from next point: ", path_dist)
    if path_dist < 0.5:
        path_index = path_index + 1
        print("are within 0.5m of next point! Increment path index, new path_index = ", path_index)

    print("Loop complete || Time: ", round(time, 1))
# ============================================================

# now plotting the trajectory of the robot

fig3, ax = plt.Figure()
for row_index, row in enumerate(r_formatted):  # row is for Y
    for col_index, elem in enumerate(row):  # col is for X
        if elem == 0.0:  # this means there is an obstacle there
            plt.scatter((col_index) * dx, (row_index)*dy, alpha=0.8, edgecolors='none', s=30, color='black')
for index in range(len(nodes_x[1:])):
    plt.scatter(nodes_x[index+1], nodes_y[index+1], alpha=0.8, edgecolors='none', s=30, color='blue')
plt.scatter((5)*dx, (95)*dy, alpha=1.0, edgecolors='none', s=50, color='green')
# and also include all the intermediary goal positions
plt.scatter(pos1[0], pos1[1], alpha=1.0, edgecolors='none', s=50, color='red')
plt.scatter(pos2[0], pos2[1], alpha=1.0, edgecolors='none', s=50, color='red')
plt.scatter(pos3[0], pos3[1], alpha=1.0, edgecolors='none', s=50, color='red')
plt.scatter(pos4[0], pos4[1], alpha=1.0, edgecolors='none', s=50, color='red')
# lets plot the edges as well (check)
for edge_p in edges[1:]:
    plt.plot([edge_p[0], edge_p[2]], [edge_p[1], edge_p[3]])
plt.scatter(state_array_x[1:], state_array_y[1:], alpha=1.0, edgecolors='none', s=50, color='blue', label='Final (PRM) Path')
plt.title('Final Robot (PRM) Path(s)')
plt.legend()
axes = plt.gca()
axes.set_xticks(np.arange(0, 10, dx))
axes.set_yticks(np.arange(0, 10, dy))
axes.set_xlim([0.0, 10])
axes.set_ylim([0.0, 10])
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()

print(">>>>>>> CODE COMPLETE <<<<<<<")
