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
    v_input = 0.1
    w_input = 0.0
    theta_ = theta
    turn_amplification = 1.0
    angle = np.arctan2((state2[1]-state1[1]), (state2[0]-state1[0]))
    # 1 degree to radians = 0.0174533
    if (angle - theta) > 0.0:  # turn ccw
        #w_input = abs(angle - theta) * turn_amplification
        w_input = 0.3
        v_input = 0.1
    elif (angle - theta) < -0.0:  # turn cw
        #w_input = -abs(angle - theta) * turn_amplification
        w_input = -0.3
        v_input = 0.1
    return v_input, w_input

k_closest = 10  # arbitrarily set
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
        # lets also make sure the distance is a max of k_dist (smaller triangles)
        if boo:
            # this node is good!
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
def bresenham_collisions_single (edge):
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
            print("edge not valid: collision detected")
            return False
        else:
            x1_temp = x1 + dx_
            y1_temp = y1 + dy_
            x1 = x1_temp
            y1 = y1_temp
            #print("moving along edge: ", (x1, y1))
    #if abs(x2 - x1) + abs(y2 - y1) <= dist_tol:
    print("Made it without a collision! The edge is valid!")
    return True
# =========================================
# Functions Defined.... moving on to main code body...
# =================================================

baseline = 0.45
time_step = 0.1
dx = dy = 0.1

# this r = MAP; which is 100X100

r = read_pgm( 'sim_map.pgm' )
r_formatted = read_pgm( 'sim_map.pgm' )
print("rows:", len(r[0][:]))  # note that Y=rows are flipped
print("cols:", len(r[:][0]))

# re-mapping to something more intuitive
for row_index, row in enumerate(r):  # row is for Y (inverted)
    for col_index, elem in enumerate(row):  # col is for X
        if elem == 0:  # this means there is an obstacle there
            r_formatted[99 - row_index][col_index] = 0.0
        else:
            r_formatted[99 - row_index][col_index] = 1.0

# now to set up the poses to vist on the ROUTE
# NOTE: the goal orientation angle (optional) is ignored
init_pos = np.array([0.5, 9.5])
pos1 = np.array([7.0, 1.5])
pos2 = np.array([9.0, 5.0])
pos3 = np.array([3.0, 9.5])
pos4 = np.array([0.5, 5.0])

# start the simulation......
loop_count = 0.0
x = np.array([5*dx, 95*dy, 0.0])  # this is the starting pose
# store values (plot path later)
state_array_x = np.array([0.0])
state_array_y = np.array([0.0])
state_array_theta = np.array([0.0])

# we will store a list of all our nodes
nodes_x = np.array([0.0])
nodes_y = np.array([0.0])
# add initial and final position(s) to nodes array
nodes_x = np.append(nodes_x, x[0])
nodes_y = np.append(nodes_y, x[1])
nodes_x = np.append(nodes_x, pos1[0])
nodes_y = np.append(nodes_y, pos1[1])
nodes_x = np.append(nodes_x, pos2[0])
nodes_y = np.append(nodes_y, pos2[1])
nodes_x = np.append(nodes_x, pos3[0])
nodes_y = np.append(nodes_y, pos3[1])
nodes_x = np.append(nodes_x, pos4[0])
nodes_y = np.append(nodes_y, pos4[1])
# initialize the edges array
edges = np.array([[0.0, 0.0, 0.0, 0.0]])  # [x1, y1, x2, y2]
# vstack for edges
object_clearance_tol = 0.1
node_clearance_tol = 0.5

bool_pos1 = False
bool_pos2 = False
bool_pos3 = False
bool_pos4 = False

# lets also set up some real time plotting
fig1 = plt.figure()
#fig1.canvas.draw()
#plt.show(block=False)
for row_index, row in enumerate(r_formatted):  # row is for Y
    for col_index, elem in enumerate(row):  # col is for X
        if elem == 0.0:  # this means there is an obstacle there
            plt.scatter((col_index) * dx, (row_index)*dy, alpha=0.8, edgecolors='none', s=30, color='black')

plt.show()

while loop_count <= 1000:  # break in code included
    print("Begin loop >>>>>>>>>>>>>>")
    # now add a random point and see if it collides
    obj_distance = 0.0
    node_distance = 1000.0
    # also make sure we are AT LEAST 0.5 units away from any other point
    while obj_distance < object_clearance_tol or node_distance < node_clearance_tol:  # as long as x_rand is close to colliding (<0.1 from obstacle), keep trying...
        # rand() returns random numb. from 0 to 1
        rand_x = round(abs(np.random.rand())*10.0, 2)
        rand_y = round(abs(np.random.rand())*10.0, 2)
        x_rand = np.array([rand_x, rand_y])
        print("random point: ", x_rand)
        obj_distance, (x_dist, y_dist) = dist_obstacle(x_rand)
        #print("random point distance (from object): ", obj_distance)
        node_distance = 1000.0
        for i in range(len(nodes_x[1:])):
            node_distance_temp = np.sqrt(np.square(x_rand[0]-nodes_x[i+1])+np.square(x_rand[1]-nodes_y[i+1]))
            if node_distance_temp < node_distance:
                node_distance = node_distance_temp
        #print("random point distance from closest node: ", node_distance)
    # now add it to nodes....
    nodes_x = np.append(nodes_x, x_rand[0])
    nodes_y = np.append(nodes_y, x_rand[1])
    print("point added to nodes: ", x_rand)
    # ============ now remove all edges and reform them all
    edges = np.array([[0.0, 0.0, 0.0, 0.0]])
    # find K closest points to x_rand from the nodes array
    # note: the last added node is the most recent....
    for i in range(len(nodes_x[1:])):
        # ===> FIND K CLOSEST THAT ARE VALID!
        closest_array = k_closest_nodes_2(nodes_x[1:], nodes_y[1:], nodes_x[i+1], nodes_y[i+1])
        #print("closest points found: ", closest_array[1:])
        # form new edges and check for any collisions
        new_edges = np.array([0.0, 0.0, 0.0, 0.0])
        for close_point in closest_array[1:]:
            new_edges = np.vstack((new_edges, [nodes_x[i+1], nodes_y[i+1], close_point[0], close_point[1]]))
            #new_edges = np.vstack((new_edges, [x_rand[0], x_rand[1], close_point[0], close_point[1]]))
        print("New edges to add: ", new_edges[1:])
        # keep only valid (no-collision) edges
        #new_edges_2 = bresenham_collisions(new_edges[1:])  # chopped here!
        new_edges_2 = new_edges[1:]
        # note: new edges 2 (collision free edges) is also already chopped
        # add new collision free edges to array of existing edges
        #print("new edges found: ", new_edges_2)
        # note that no new edges could also be found. So first check this
        if np.mean(new_edges_2) != 0.0:  # means non-zero = we found something!
            for edge_2 in new_edges_2:
                edges = np.vstack((edges, edge_2))
    print("number of nodes: ", len(nodes_x[1:]))
    print("number/ shape of edges: ", edges[1:].shape)
    loop_count = loop_count + 1
    print(">>>>>>>>>> Loop Completed || Loop Count: ", loop_count)
    # plot found nodes and edges (real time)....
    print("now plotting...")
    """
    for index in range(len(nodes_x[1:])):
        plt.scatter(nodes_x[index + 1], nodes_y[index + 1], alpha=0.8, edgecolors='none', s=30, color='blue')
    for edge_p in edges[1:]:
        plt.plot([edge_p[0], edge_p[2]], [edge_p[1], edge_p[3]])
    # also plot gridworld obstacles
    # plot the obstacles ie) gridworld
    for row_index, row in enumerate(r_formatted):  # row is for Y
        for col_index, elem in enumerate(row):  # col is for X
            if elem == 0.0:  # this means there is an obstacle there
                plt.scatter((col_index) * dx, (row_index) * dy, alpha=0.8, edgecolors='none', s=30, color='black')
    plt.title('NODE/ EDGE MAP')
    plt.legend()
    axes = plt.gca()
    axes.set_xticks(np.arange(0, 10, dx))
    axes.set_yticks(np.arange(0, 10, dy))
    axes.set_xlim([0.0, 10])
    axes.set_ylim([0.0, 10])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    #fig1.canvas.draw()
    #timer.sleep(0.1)
    #plt.clf()
    """

    # using NETWORKX - GRAPH BASED PATH PLANNING
    # ===================================
    print("============== NetworkX Path planning ....")
    G = nx.Graph()  # always re-initiating our graph
    for i in range(len(nodes_x[1:])):
        string = '{},{}'.format(nodes_x[i+1], nodes_y[i+1])
        print("string: ", string)
        G.add_node(string, pos=(nodes_x[i+1], nodes_y[i+1]))
    print("all nodes added!")
    print("list(G.nodes): ", list(G.nodes))

    for edge in edges[1:]:
        edge_x1 = edge[0]
        edge_y1 = edge[1]
        edge_x2 = edge[2]
        edge_y2 = edge[3]
        edge_distance = np.sqrt(np.square(edge_x1 - edge_x2) + np.square(edge_y1 - edge_y2))
        string1 = '{},{}'.format(edge_x1, edge_y1)
        string2 = '{},{}'.format(edge_x2, edge_y2)
        G.add_edge(string1, string2, weight=edge_distance)
    print("list(G.edges): ", list(G.edges))

    # see if the paths exist...
    init_string = '{},{}'.format(init_pos[0], init_pos[1])
    pos1_string = '{},{}'.format(pos1[0], pos1[1])
    pos2_string = '{},{}'.format(pos2[0], pos2[1])
    pos3_string = '{},{}'.format(pos3[0], pos3[1])
    pos4_string = '{},{}'.format(pos4[0], pos4[1])
    bool_pos1 = nx.has_path(G, init_string, pos1_string)
    bool_pos2 = nx.has_path(G, pos1_string, pos2_string)
    bool_pos3 = nx.has_path(G, pos2_string, pos3_string)
    bool_pos4 = nx.has_path(G, pos3_string, pos4_string)
    print("do we have paths?....")
    print("for pos1: ", bool_pos1)
    print("for pos2: ", bool_pos2)
    print("for pos3: ", bool_pos3)
    print("for pos4: ", bool_pos4)
    if bool_pos1:
        print("===> init to POS1 Solved")
        print("Shortest Path: ", nx.shortest_path(G, init_string, pos1_string))
    if bool_pos2:
        print("===> POS1 to POS2 Solved")
        print("Shortest Path: ", nx.shortest_path(G, pos1_string, pos2_string))
    if bool_pos3:
        print("===> POS2 to POS3 Solved")
        print("Shortest Path: ", nx.shortest_path(G, pos2_string, pos3_string))
    if bool_pos4:
        print("===> POS3 to POS4 Solved")
        print("Shortest Path: ", nx.shortest_path(G, pos3_string, pos4_string))
    if bool_pos1 and bool_pos2 and bool_pos3 and bool_pos4:
        print(">>>>>>>>>>>>>>>>>>>> ALL PATH(s) FOUND! <<<<<<<<<<<<<<<<")
        # save all data to .csv
        np.savetxt('Q2_nodes_x.csv', nodes_x, delimiter=',')
        np.savetxt('Q2_nodes_y.csv', nodes_y, delimiter=',')
        np.savetxt('Q2_edges.csv', edges, delimiter=',')
        break
    else:
        print("Paths NOT found...yet")


print(">>>>>>>>>>>>>> END of PROBABILISTIC ROADMAP SOLVER")
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
init_string = '{},{}'.format(init_pos[0], init_pos[1])
pos1_string = '{},{}'.format(pos1[0], pos1[1])
pos2_string = '{},{}'.format(pos2[0], pos2[1])
pos3_string = '{},{}'.format(pos3[0], pos3[1])
pos4_string = '{},{}'.format(pos4[0], pos4[1])
# first acquire all the paths
# path_1 = nx.shortest_path(G, init_string, pos1_string)
# path_2 = nx.shortest_path(G, pos1_string, pos2_string)
# path_3 = nx.shortest_path(G, pos2_string, pos3_string)
# path_4 = nx.shortest_path(G, pos3_string, pos4_string)
path_1 = nx.dijkstra_path(G, init_string, pos1_string)
path_2 = nx.dijkstra_path(G, pos1_string, pos2_string)
path_3 = nx.dijkstra_path(G, pos2_string, pos3_string)
path_4 = nx.dijkstra_path(G, pos3_string, pos4_string)
# now extracting the sequence of position values in each path
print("====> Extracting sequence of position values...")
path_array = np.array([0.0, 0.0])
for element in path_1:
    temp_elem = element.split(',')
    x_p_val = float(temp_elem[0])
    y_p_val = float(temp_elem[1])
    print("x and y values: ", (x_p_val, y_p_val))
    path_array = np.vstack((path_array, [x_p_val, y_p_val]))
for element in path_2[1:]:  # chopped as path1 includes first point of path2
    temp_elem = element.split(',')
    x_p_val = float(temp_elem[0])
    y_p_val = float(temp_elem[1])
    print("x and y values: ", (x_p_val, y_p_val))
    path_array = np.vstack((path_array, [x_p_val, y_p_val]))
for element in path_3[1:]:
    temp_elem = element.split(',')
    x_p_val = float(temp_elem[0])
    y_p_val = float(temp_elem[1])
    print("x and y values: ", (x_p_val, y_p_val))
    path_array = np.vstack((path_array, [x_p_val, y_p_val]))
for element in path_4[1:]:
    temp_elem = element.split(',')
    x_p_val = float(temp_elem[0])
    y_p_val = float(temp_elem[1])
    print("x and y values: ", (x_p_val, y_p_val))
    path_array = np.vstack((path_array, [x_p_val, y_p_val]))
final_path_array = path_array[1:]
np.savetxt('Q2_final_path.csv', final_path_array, delimiter=',')
print("Final path of robot saved!")
# =================================================================== SIMULATION

time = 0.0
theta = 0.0  # this is taken arbitrarily (as in Q1)
# now simulate the robots path
path_index = 0
# the initial state is...
x = np.array([final_path_array[path_index][0], final_path_array[path_index][1], theta])
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
    path_dist = np.sqrt(np.square(state[0] - final_path_array[path_index + 1][0]) + np.square(state[1] - final_path_array[path_index + 1][1]))
    print("distance from next point: ", path_dist)
    if path_dist < 0.5:
        path_index = path_index + 1
        print("are within 0.5m of next point! Increment path index, new path_index = ", path_index, ", goal = ", final_path_array[path_index])

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
