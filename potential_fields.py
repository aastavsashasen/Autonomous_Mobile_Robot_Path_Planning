import numpy as np
import time as timer
import autograd
from autograd import jacobian
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import _tkinter
import scipy
from scipy.stats import norm
matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)

# The inputs are dependant on the algorithm! = Path planning => [v, w]

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
# Potential Fields specific functions....

K_a = 10.0  # original = 2.0
def f_a (x, x_goal):  # returns scalar
    return K_a * np.sqrt(np.square(x[0] - x_goal[0]) + np.square(x[1] - x_goal[1]))

def dist_obstacle (x):  # inefficient: loops through ALL obstacle points
    _dist = 10
    x_val = 0.0
    y_val = 0.0
    for row_index, row in enumerate(r):
        for col_index, elem in enumerate(row):
            if elem == 0:  # this means there is an obstacle there
                obstacle_x = col_index*dx
                #obstacle_y = row_index*dy
                obstacle_y = (99 - row_index) * dy
                #print("obstacle @: ", (obstacle_x, obstacle_y))
                dist_temp = np.sqrt(np.square(x[0]-obstacle_x)+np.square(x[1]-obstacle_y))
                #dist_temp = abs(x[0] - obstacle_x) + abs(x[1] - obstacle_y)
                if dist_temp < _dist:
                    _dist = dist_temp
                    x_val = obstacle_x
                    y_val = obstacle_y

    return _dist, (x_val, y_val)  # return distance and point

d_0 = 2.0  # original = 2.0
K_r = 2.0  # original = 2.0
max_repel_val = 300.0  # original = 18.0
def f_r (x):
    distance, (_x, _y) = dist_obstacle(x)
    if distance > d_0:
        return 0.0
    elif distance == 0.0:
        return max_repel_val
    else:
        #return K_r * ((1/distance)-(1/d_0))
        return K_r * np.square((1/distance)-(1/d_0))

K_vel = 1.0
def velocity_from_gradient (x_tbh, x_goal):
    x = np.array([x_tbh[0], x_tbh[1]])
    distance, (_x, _y) = dist_obstacle(x)
    print("--------------- VELOCITY FROM GRAD, point: ", x_tbh)
    print("distance to object: ", distance)
    print("point on object: ", (_x, _y))
    f_1_temp = 2*K_a*(x - x_goal)
    f_1 = f_1_temp/np.linalg.norm(f_1_temp)
    print("value of f1 (normalized): ", f_1)
    # ------------------------
    if distance <= d_0:
        f_2_temp = 2*K_r*((1/d_0)-(1/distance))*((x - np.array([_x, _y]))/(np.power(distance, 3)))
        f_2 = f_2_temp/np.linalg.norm(f_2_temp)
        print("WE ARE CLOSE TO AN OBJECT!!!")
    else:
        print("We are NOT close to an object...")
        f_2 = np.array([0.0, 0.0])
    print("value of f2 (normalized): ", f_2)
    #final_vel = -K_vel*(f_1 - f_2)
    final_vel = -K_vel * (f_1 + f_2)
    print("and then the final velocity: ", final_vel)
    magnitude = np.sqrt(np.square(final_vel[0]) + np.square(final_vel[1]))
    normalized_final_vel = np.array([final_vel[0]/magnitude, final_vel[1]/magnitude])
    return normalized_final_vel

def controller (x, vel_grad):  # where x = x[0], x[1] = x, y
    v_input = np.sqrt(np.square(vel_grad[0])+np.square(vel_grad[1]))*0.8
    w_input = 0.0
    turn_amplification = 1.5
    angle = np.arctan2(vel_grad[1], vel_grad[0])
    print("==========> Vel. Grad. Vector ANGLE: ", angle)
    if (angle - x[2]) > 0.0:  # turn ccw
        w_input = abs(angle - x[2]) * turn_amplification
        print("w_input) turn CCW: ", w_input)
    elif (angle - x[2]) < 0.0:  # turn cw
        w_input = -abs(angle - x[2]) * turn_amplification
        print("w_input) turn CW: ", w_input)
    return v_input, w_input


# =====================================================================================

baseline = 0.45
time_step = 0.1
dx = dy = 0.1

# this r = MAP; which is 100X100
r = read_pgm( 'sim_map.pgm' )
print("rows:", len(r[0][:]))  # note that Y=rows are flipped
print("cols:", len(r[:][0]))

fig1 = plt.figure()
for row_index, row in enumerate(r):
    for col_index, elem in enumerate(row):
        if elem == 0:  # this means there is an obstacle there
            #print("zero element!: ", elem)
            plt.scatter((col_index) * dx, (99 - row_index)*dy, alpha=0.8, edgecolors='none', s=30, color='black')
        else:
            dist, (x_p, y_p) = dist_obstacle(np.array([col_index*dx, (99-row_index)*dy]))
            #print("Non-Zero element!: ", elem)
            plt.scatter((col_index)*dx, (99 - row_index)*dy, alpha=0.8, edgecolors='none', s=20*dist*10, color='blue')
            plt.scatter(x_p, y_p, alpha=0.8, edgecolors='none', s=20 * dist * 10, color='yellow')
plt.scatter((5)*dx, (95)*dy, alpha=1.0, edgecolors='none', s=50, color='green')
plt.scatter((95)*dx, (5)*dy, alpha=1.0, edgecolors='none', s=50, color='red')
plt.title('MAP')
plt.legend()
axes = plt.gca()
axes.set_xticks(np.arange(0, len(r[:][0])*dx, dx))
axes.set_yticks(np.arange(0, len(r[0][:])*dy, dy))
axes.set_xlim([0.0, len(r[:][0])*dx])
axes.set_ylim([0.0, len(r[0][:])*dy])
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
#plt.show()

# Plot a 3D potential function to analyse the effect of
# and optimize algorithmic hyperparameters
# form x,y and z arrays ...
fig2 = plt.figure()
ax = fig2.gca(projection='3d')

# Make data.
X = np.arange(0, 10, 0.1)
Y = np.arange(0, 10, 0.1)
X_mesh, Y_mesh = np.meshgrid(X, Y)
zs = np.zeros((100, 100))
for x_i, x_val in enumerate(X):  # cols = X
    for y_i, y_val in enumerate(Y):  # rows = Y
        #print("index and values and f_a: ", (x_i, y_i), (x_val, y_val), f_a(x=np.array([x_val, y_val]), x_goal=np.array([9.5, 0.5])))
        #zs[y_i, len(Y) - 1 - x_i] = f_a(x=np.array([x_val, y_val]), x_goal=np.array([95*dx, 5*dy])) + f_r(x=np.array([x_val, y_val]))
        #zs[len(Y) - 1 - y_i, x_i] = 0*f_a(x=np.array([x_val, y_val]), x_goal=np.array([95 * dx, 5 * dy])) + f_r(x=np.array([x_val, y_val]))
        ### SINCE MATPLOTLIB IS BROKEN FOR SOME REASON SWITH THE AXIS!
        zs[y_i, x_i] = f_a(x=np.array([x_val, y_val]), x_goal=np.array([9.5, 0.5])) + \
                       f_r(x=np.array([x_val, y_val]))
print("zs shape: ", zs.shape)
Z = zs
#Z = zs.reshape(X.shape)
# Plot the surface.
#surf = ax.plot_surface(X_mesh, Y_mesh, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
surf = ax.plot_surface(X_mesh, Y_mesh, Z, cmap=cm.coolwarm)
# Add a color bar which maps values to colors.
fig2.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Pot. Field. Value')

# WHAT ABOUT SMOOTHER INTERPOLATED DATA?
# fig3 = plt.figure()
# ax = fig3.gca(projection='3d')
# X_ = np.arange(0, 10, 0.01)
# Y_ = np.arange(0, 10, 0.01)
# X_new, Y_new = np.meshgrid(X_, Y_)
# tck = interpolate.bisplrep(X, Y, Z, s=0)
# Z_new = interpolate.bisplev(X_new[:, 0], Y_new[0, :], tck)
# surf = ax.plot_surface(X_new, Y_new, Z_new, cmap=cm.coolwarm, antialiased=True)
# fig3.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Pot. Field. Value (Interpolated)')

#plt.show()

# start the simulation......
time = 0.0
theta = 0.0
x = np.array([5*dx, 95*dy, theta])  # this is the starting pose
x_goal_ = np.array([95*dx, 5*dy])
# store values (plot path later)
state_array_x = np.array([0.0])
state_array_y = np.array([0.0])
state_array_theta = np.array([0.0])
# and for visualizing the suggested direction (vel_gradient)
U = np.array([0.0])
V = np.array([0.0])
distance_from_goal = 10.0

# keep looping as long as we are far from the goal, or time limit not reached
while time <= 100 and distance_from_goal > 0.1:
    # now calc the velocity magnitude and direction to move in...
    print("Begin loop >>>>>>>>>>>>>>")
    print("Current state: ", x)
    vel_gradient = velocity_from_gradient(x, x_goal_)  # vel = [x, y] np vector
    print("Velocity from gradient: ", vel_gradient)
    v_input, w_input = controller(x, vel_gradient)
    print("chosen controller inputs (v, w): ", (v_input, w_input))
    robot_pos_new = np.array([x1_calc(x, time, v_input),
                              x2_calc(x, time, v_input),
                              x3_calc(x, time, w_input)])
    print("Next robot state: ", (robot_pos_new[0], robot_pos_new[1], robot_pos_new[2]))
    # store the data!
    state_array_x = np.append(state_array_x, x[0])
    state_array_y = np.append(state_array_y, x[1])
    state_array_theta = np.append(state_array_theta, x[2])
    # and a little extra data: vel_gradient vectors
    U = np.append(U, vel_gradient[0])
    V = np.append(V, vel_gradient[1])
    # now increment time and carry forward the data
    x[0] = robot_pos_new[0]
    x[1] = robot_pos_new[1]
    x[2] = robot_pos_new[2]
    time = time + time_step
    distance_from_goal = np.sqrt(np.square(x[0]-9.5)+np.square(x[1]-0.5))
    print("Distance from goal: ", distance_from_goal)
    print("Loop complete || Time: ", round(time, 1))
# ============================================================
# now plotting the trajectory of the robot
widths = np.ones((len(state_array_x[1:])))*0.0001
fig3, ax = plt.subplots()
for row_index, row in enumerate(r):
    for col_index, elem in enumerate(row):
        if elem == 0:  # this means there is an obstacle there
            plt.scatter((col_index) * dx, (99 - row_index)*dy, alpha=0.8, edgecolors='none', s=30, color='black')
        else:
            dist, (x_p, y_p) = dist_obstacle(np.array([col_index*dx, (99-row_index)*dy]))
            plt.scatter(x_p, y_p, alpha=0.8, edgecolors='none', s=20 * dist * 10, color='yellow')
plt.scatter((5)*dx, (99 - 5)*dy, alpha=1.0, edgecolors='none', s=50, color='green')
plt.scatter((95)*dx, (99 - 95)*dy, alpha=1.0, edgecolors='none', s=50, color='red')
plt.scatter(state_array_x[1:], state_array_y[1:], alpha=1.0, edgecolors='none', s=50, color='blue', label='Final (PF) Path')
q = ax.quiver(state_array_x[1:], state_array_y[1:], U[1:], V[1:], linewidths=widths)
plt.quiverkey(q, X=0.3, Y=1.1, U=10, label="Anti-Gradient", labelpos='E')
plt.title('Resulting Path')
plt.legend()
axes = plt.gca()
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()

print(">>>>>>> CODE COMPLETE <<<<<<<")
