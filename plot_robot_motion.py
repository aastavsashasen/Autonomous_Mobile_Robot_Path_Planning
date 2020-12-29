import sys, os
import numpy as np
import time as timer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)
import networkx as nx
import pickle

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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

def controller_carrot (state1, theta,  state2):  # where x = x[0], x[1] = x, y

    v_input = 0.3
    w_input = 0.0
    theta_ = theta  # theta value goes till
    print("<<<<<<<<<<<<< Controller: ")
    print("state 1: ", state1)
    print("state 2: ", state2)
    print("theta before mapping: ", theta)
    if theta > np.pi:
        theta_temp = - theta
        theta = theta_temp
    if theta < - np.pi * (2):
        theta_temp = - theta
        theta = theta_temp
    angle = np.arctan2((state2[1] - state1[1]), (state2[0] - state1[0]))
    # angle>> [-pi, pi] = both 180 = problem?
    print("theta after mapping: ", theta)
    print("angle from arctan2: ", angle)
    print("resulting (angle - theta): ", (angle - theta))
    difference = (angle - theta)
    difference = (difference + np.pi) % (2 * np.pi) - np.pi
    # uneccessary if statement used for investigation ONLY
    # 1 degree to radians = 0.0174533
    if (angle >= 0 and theta >= 0):
        # w_input = abs(angle - theta) * turn_amplification
        print("1angle from arctan2: ", angle)
        w_input = 0.0 + 2.0 * difference
        v_input = 0.3
    elif (angle >= 0 and theta <= 0):  # > 0.05:  # turn ccw DONE
        # w_input = abs(angle - theta) * turn_amplification
        print("1angle from arctan2: ", angle)
        w_input = 0.0 + 2.0 * difference
        v_input = 0.3
    elif (angle <= 0 and theta <= 0):  # DONE
        # w_input = -abs(angle - theta) * turn_amplification
        print("2angle from arctan2: ", angle)
        w_input = -0.0 + 2.0 * difference
        v_input = 0.3
    elif (angle <= 0 and theta >= 0):  # turn cw DONE
        # w_input = -abs(angle - theta) * turn_amplification
        print("2angle from arctan2: ", angle)
        w_input = -0.0 + 2.0 * difference
        v_input = 0.3
    print("Final vel. input and ang. vel input: ", v_input, ", ", w_input)
    return v_input, w_input

def graph_to_nodes (graph_object):
    nodes_array = np.array([0.0, 0.0])
    nodes_list = list(graph_object.nodes())  # ['5.6,7.8', '2.1,4.9', ...]
    for element in nodes_list:
        temp_elem = element.split(',')
        x_p_val = float(temp_elem[0])
        y_p_val = float(temp_elem[1])
        nodes_array = np.vstack((nodes_array, [x_p_val, y_p_val]))
    return nodes_array[1:]

def graph_to_edges (graph_object):
    edges_array = np.array([0.0, 0.0, 0.0, 0.0])
    edge_list = list(graph_object.edges())  # [('5.6,7.8', '2.1,4.9'), ...]
    # if there are no edges just return an empty list
    if len(edge_list) == 0:
        return edge_list
    for element in edge_list:
        point1 = element[0]
        point2 = element[1]
        temp_elem_1 = point1.split(',')
        temp_elem_2 = point2.split(',')
        x_1 = float(temp_elem_1[0])
        y_1 = float(temp_elem_1[1])
        x_2 = float(temp_elem_2[0])
        y_2 = float(temp_elem_2[1])
        edges_array = np.vstack((edges_array, [x_1, y_1, x_2, y_2]))
    return edges_array[1:]

# =================================================

# Now we simulate the robots motion
# this r = MAP; which is 100X100
r = read_pgm( 'sim_map.pgm' )
r_formatted = read_pgm( 'sim_map.pgm' )

# re-mapping to something more intuitive
for row_index, row in enumerate(r):  # row is for Y (inverted)
    for col_index, elem in enumerate(row):  # col is for X
        if elem == 0:  # this means there is an obstacle there
            r_formatted[99 - row_index][col_index] = 0.0
        else:
            r_formatted[99 - row_index][col_index] = 1.0

init_pos = np.array([0.5, 9.5])
pos1 = np.array([7.0, 1.5])
pos2 = np.array([9.0, 5.0])
pos3 = np.array([3.0, 9.5])
pos4 = np.array([0.5, 5.0])


init_string = '{},{}'.format(init_pos[0], init_pos[1])
pos1_string = '{},{}'.format(pos1[0], pos1[1])
pos2_string = '{},{}'.format(pos2[0], pos2[1])
pos3_string = '{},{}'.format(pos3[0], pos3[1])
pos4_string = '{},{}'.format(pos4[0], pos4[1])
# first acquire all the paths
# loading all the arrays required:
file1 = open('q3_data/G_path_1.txt', 'rb')
G_path_1 = pickle.load(file1)
file1.close()
file2 = open('q3_data/G_path_2.txt', 'rb')
G_path_2 = pickle.load(file2)
file2.close()
file3 = open('q3_data/G_path_3.txt', 'rb')
G_path_3 = pickle.load(file3)
file3.close()
file4 = open('q3_data/G_path_4.txt', 'rb')
G_path_4 = pickle.load(file4)
file4.close()

path_1 = nx.shortest_path(G_path_1, init_string, pos1_string)
path_2 = nx.shortest_path(G_path_2, pos1_string, pos2_string)
path_3 = nx.shortest_path(G_path_3, pos2_string, pos3_string)
path_4 = nx.shortest_path(G_path_4, pos3_string, pos4_string)

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

# now extracting the nodes and edges
G_1_array = graph_to_nodes(G_path_1)
G_2_array = graph_to_nodes(G_path_2)
G_3_array = graph_to_nodes(G_path_3)
G_4_array = graph_to_nodes(G_path_4)
final_graphs_array_ = np.array([0.0, 0.0])
final_graphs_array_ = np.vstack((final_graphs_array_, G_1_array))
final_graphs_array_ = np.vstack((final_graphs_array_, G_2_array))
final_graphs_array_ = np.vstack((final_graphs_array_, G_3_array))
final_graphs_array_ = np.vstack((final_graphs_array_, G_4_array))
final_graphs_array = final_graphs_array_[1:]

# and for edges.....
final_graphs_array_e_ = np.array([0.0, 0.0, 0.0, 0.0])
G_1_array_e = graph_to_edges(G_path_1)
G_2_array_e = graph_to_edges(G_path_2)
G_3_array_e = graph_to_edges(G_path_3)
G_4_array_e = graph_to_edges(G_path_4)


if len(G_1_array_e) != 0:
    final_graphs_array_e_ = np.vstack((final_graphs_array_e_, G_1_array_e))
if len(G_2_array_e) != 0:
    final_graphs_array_e_ = np.vstack((final_graphs_array_e_, G_2_array_e))
if len(G_3_array_e) != 0:
    final_graphs_array_e_ = np.vstack((final_graphs_array_e_, G_3_array_e))
if len(G_4_array_e) != 0:
    final_graphs_array_e_ = np.vstack((final_graphs_array_e_, G_4_array_e))

final_graphs_array_e = final_graphs_array_e_[1:]

# now populating our relevant arrays...
#nodes_x = np.loadtxt('q1_data_backup/Q2_nodes_x.csv', delimiter=',')
#nodes_y = np.loadtxt('q1_data_backup/Q2_nodes_y.csv', delimiter=',')
#edges = np.loadtxt('q1_data_backup/Q2_edges.csv', delimiter=',')
#final_path_array = np.loadtxt('q1_data_backup/Q2_final_path.csv', delimiter=',')
nodes_x_ = np.array([0.0])
nodes_y_ = np.array([0.0])
for element in final_graphs_array:
    nodes_x_ = np.append(nodes_x_, final_graphs_array[0])
    nodes_y_ = np.append(nodes_y_, final_graphs_array[1])
nodes_x = nodes_x_[1:]
nodes_y = nodes_y_[1:]

edges = final_graphs_array_e


print("nodes_x: ", len(nodes_x))
print("nodes_y: ", len(nodes_y))
print("edges: ", edges.shape)
print("final path array: ", final_path_array.shape)

# now plotting our final map...
dx = dy = 0.1

# fig1 = plt.figure()
# for row_index, row in enumerate(r_formatted):  # row is for Y
#     for col_index, elem in enumerate(row):  # col is for X
#         if elem == 0.0:  # this means there is an obstacle there
#             plt.scatter((col_index) * dx, (row_index)*dy, alpha=0.8, edgecolors='none', s=30, color='black')
# # now to plot all the nodes
# # note: range() starts at zero
# for index in range(len(nodes_x[1:])):
#     plt.scatter(nodes_x[index+1], nodes_y[index+1], alpha=0.8, edgecolors='none', s=30, color='blue')
# plt.scatter((5)*dx, (95)*dy, alpha=1.0, edgecolors='none', s=50, color='green')
# # and also include all the intermediary goal positions
# plt.scatter(pos1[0], pos1[1], alpha=1.0, edgecolors='none', s=50, color='red')
# plt.scatter(pos2[0], pos2[1], alpha=1.0, edgecolors='none', s=50, color='red')
# plt.scatter(pos3[0], pos3[1], alpha=1.0, edgecolors='none', s=50, color='red')
# plt.scatter(pos4[0], pos4[1], alpha=1.0, edgecolors='none', s=50, color='red')
# # lets plot the edges as well (check)
# for edge_p in edges[1:]:
#     plt.plot([edge_p[0], edge_p[2]], [edge_p[1], edge_p[3]])
# plt.title('Final NODE/ EDGE MAP')
# plt.legend()
# axes = plt.gca()
# axes.set_xticks(np.arange(0, 10, dx*10.0))
# axes.set_yticks(np.arange(0, 10, dy*10.0))
# axes.set_xlim([0.0, 10])
# axes.set_ylim([0.0, 10])
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid()
# plt.show()

# =================================================================== SIMULATION

fig1 = plt.figure()
fig1.canvas.draw()
plt.show(block=False)

time = 0.0
time_step = 0.1
theta = 0.0  # this is taken arbitrarily (as in Q1)
# now simulate the robots path
path_index = 0
# the initial state is...
x = np.array([final_path_array[path_index][0], final_path_array[path_index][1], theta])
# what is the max path index?....
max_path_index = final_path_array.shape[0]  # number of rows
print("max path index: ", max_path_index)
# record the robots progress
state_array_x = np.array([0.0])
state_array_y = np.array([0.0])
state_array_theta = np.array([0.0])

# plot the solved PRM node/edge Map
for element in final_path_array:
    plt.scatter(element[0], element[1], alpha=0.8, edgecolors='none', s=30, color='blue')
for edge_p in edges[1:]:
    plt.plot([edge_p[0], edge_p[2]], [edge_p[1], edge_p[3]])
# also plot gridworld obstacles
# plot the obstacles ie) gridworld
for row_index, row in enumerate(r_formatted):  # row is for Y
    for col_index, elem in enumerate(row):  # col is for X
        if elem == 0.0:  # this means there is an obstacle there
            plt.scatter((col_index) * dx, (row_index) * dy, alpha=0.8, edgecolors='none', s=30, color='black')
plt.scatter((5)*dx, (95)*dy, alpha=1.0, edgecolors='none', s=50, color='green', label='Robot Path')
# and also include all the intermediary goal positions
plt.scatter(pos1[0], pos1[1], alpha=1.0, edgecolors='none', s=50, color='red')
plt.scatter(pos2[0], pos2[1], alpha=1.0, edgecolors='none', s=50, color='red')
plt.scatter(pos3[0], pos3[1], alpha=1.0, edgecolors='none', s=50, color='red')
plt.scatter(pos4[0], pos4[1], alpha=1.0, edgecolors='none', s=50, color='red')

count1 = 0
count2 = 0
count3 = 0
count4 = 0
print_count = 0
while path_index < (max_path_index - 1):
    # ================================
    state = final_path_array[path_index]
    at_pos1 = (abs(x[0] - 7.0) + abs(x[1] - 1.5)) < 0.5
    at_pos2 = (abs(x[0] - 9.0) + abs(x[1] - 5.0)) < 0.5
    at_pos3 = (abs(x[0] - 3.0) + abs(x[1] - 9.5)) < 0.5
    at_pos4 = (abs(x[0] - 0.5) + abs(x[1] - 5.0)) < 0.5
    if at_pos1 and count1 == 0 and count2 == 0 and count3 == 0 and count4 == 0:
        enablePrint()
        count1 = 1
        print("at pos 1: ", (x[0], x[1]))
        print("Waiting for input")
        input('')
        blockPrint()
    if at_pos2 and count1 == 1 and count2 == 0 and count3 == 0 and count4 == 0:
        enablePrint()
        count2 = 1
        print("at pos 2: ", (x[0], x[1]))
        print("Waiting for input")
        input('')
        blockPrint()
    if at_pos3 and count1 == 1 and count2 == 1 and count3 == 0 and count4 == 0:
        enablePrint()
        count3 = 1
        print("at pos 3: ", (x[0], x[1]))
        print("Waiting for input")
        input('')
        blockPrint()
    if at_pos4 and count1 == 1 and count2 == 1 and count3 == 1 and count4 == 0:
        enablePrint()
        count4 = 1
        print("at pos 4: ", (x[0], x[1]))
        print("Waiting for input")
        input('')
        blockPrint()

    print("Current start point: ", state)
    # find control inputs to take up to the next state
    print("!!! current state:", (x[0], x[1], x[2]))
    v_input, w_input = controller_carrot(np.array([x[0], x[1]]), x[2], final_path_array[path_index + 1])
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
    # if path_index > 2:
    #     q1 = ax.quiver(x[0], x[1], final_path_array[path_index + 1][0] - x[0], final_path_array[path_index + 1][1] - x[1], angles='xy', color='blue', scale=1, scale_units='xy')  # goal
    #     q2 = ax.quiver(x[0], x[1], np.cos(x[2]), np.sin(x[2]), angles='xy', color='red', scale=1, scale_units='xy')
    # #p1 = plt.quiverkey(q1, 1, 16.5, 50, "50 m/s", coordinates='data', color='blue')
    #p2 = plt.quiverkey(q2, 1, 16.5, 50, "50 m/s", coordinates='data', color='red')
    path_dist = np.sqrt(np.square(x[0] - final_path_array[path_index + 1][0]) + np.square(x[1] - final_path_array[path_index + 1][1]))
    print("distance from next point: ", path_dist)
    plt.scatter(x[0], x[1], alpha=1.0, edgecolors='none', s=30, color='green')
    if path_dist < 0.4:  # safer margin than neccessary
        path_index = path_index + 1
        print("are within 0.1m of next point! Increment path index, new path_index = ", path_index, ", goal = ", final_path_array[path_index])
    # ===========================

    print("real time plotting ....")
    # for i in range(len(state_array_x[1:])):
    #     plt.scatter(state_array_x[i + 1], state_array_y[i + 1], alpha=1.0, edgecolors='none', s=40, color='red')


    plt.title('NODE/ EDGE MAP')
    plt.legend()
    axes = plt.gca()
    axes.set_xticks(np.arange(0, 10, dx*10.0))
    axes.set_yticks(np.arange(0, 10, dy*10.0))
    axes.set_xlim([0.0, 10])
    axes.set_ylim([0.0, 10])
    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.grid()
    fig1.canvas.draw()
    timer.sleep(0.000001)
    #plt.clf()
    # if path_index > 2:
    #     q1.remove()
    #     q2.remove()
    print_count = print_count + 1
    print("Loop complete || Time: ", round(time, 1))
# ============================================================

# now plotting the trajectory of the robot

fig3 = plt.figure()
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
axes.set_xticks(np.arange(0, 10, dx*10.0))
axes.set_yticks(np.arange(0, 10, dy*10.0))
axes.set_xlim([0.0, 10])
axes.set_ylim([0.0, 10])
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()

print(">>>>>>>> TRAJECTORY SIMULATION COMPLETE! <<<<<<<<<<<<<<<<<")
