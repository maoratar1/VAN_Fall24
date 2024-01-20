import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

TOLERANCE = 10**(-5)
RAD_TO_DEG = 180 / np.pi
DEG_TO_RAD = np.pi / 180

empty = np.array([[0, 0, 0.],
          [0, 0, 0],
          [0, 0, 0]])


def euler_to_rot_mat(yaw, pitch, roll):
    # Rz = yaw | Ry = pitch | Rx = roll
    Rx = np.array([[1., 0., 0.],
                   [0., np.cos(roll), np.sin(roll)],
                   [0., -np.sin(roll), np.cos(roll)]])

    Ry = np.array([[np.cos(pitch), 0., -np.sin(pitch)],
                   [0., 1., 0.],
                   [np.sin(pitch), 0., np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), np.sin(yaw), 0.],
                   [-np.sin(yaw), np.cos(yaw), 0.],
                   [0., 0., 1.]])

    return np.matmul(Rz, np.matmul(Ry, Rx))


def rot_mat_to_euler_angles(rot_mat):
    tol = 10 ** (-4)
    # psi = yaw | theta = pitch | phi = roll
    yaw1, yaw2, pitch1, pitch2, roll1, roll2 = 0, 0, 0, 0, 0, 0
    if rot_mat[2, 0] != 1 or rot_mat[2, 0] != -1:
        pitch1 = np.arcsin(rot_mat[2, 0])
        pitch2 = -np.pi - pitch1

        roll1 = np.arcsin(-rot_mat[2, 1] / np.cos(pitch1))
        if np.abs(np.cos(pitch1) * np.cos(roll1) - rot_mat[2, 2]) > tol:
            roll1 = -np.pi - roll1

        roll2 = np.arcsin(-rot_mat[2, 1] / np.cos(pitch2))
        if np.abs(np.cos(pitch2) * np.cos(roll2) - rot_mat[2, 2]) > tol:
            roll2 = -np.pi - roll2

        yaw1 = np.arcsin(-rot_mat[1, 0] / np.cos(pitch1))
        if np.abs(np.cos(pitch1) * np.cos(yaw1) - rot_mat[0, 0]) > tol:
            yaw1 = - np.pi - yaw1

        yaw2 = np.arcsin(-rot_mat[1, 0] / np.cos(pitch2))
        if np.abs(np.cos(pitch2) * np.cos(yaw2) - rot_mat[0, 0]) > tol:
            yaw2 = - np.pi - yaw2
        # yaw2 = np.sign(yaw1) * np.pi - yaw1

    elif rot_mat[2, 0] == 1:
        pitch1 = np.pi / 2
        pitch2 = np.pi / 2

        roll1 = np.arctan(rot_mat[0, 1] / rot_mat[1, 1])
        roll2 = np.pi - roll1

        yaw1 = 0
        yaw2 = 0
    elif rot_mat[2, 0] == -1:
        pitch1 = 3 * np.pi / 2
        pitch2 = 3 * np.pi / 2

        roll1 = -np.arctan(rot_mat[0, 1] / rot_mat[1, 1])
        roll2 = np.pi - roll1

        yaw1 = 0
        yaw2 = 0

    # yaw_ret, pitch_ret, roll_ret = 0, 0, 0
    # if yaw1 > 0 and

    return convert_to_range(yaw1), convert_to_range(yaw2), \
           convert_to_range(pitch1), convert_to_range(pitch2), \
           convert_to_range(roll1), convert_to_range(roll2)


def q1b():
    yaw = np.pi / 7.  # psi
    pitch = np.pi / 5.  # theta
    roll = np.pi / 4.  # phi

    print("Rotation Matrix for yaw {} pitch {} roll {}\n".format(yaw, pitch, roll))
    print(euler_to_rot_mat(yaw, pitch, roll))


def convert_to_range(angle):
    if angle < - np.pi:
        return 2 * np.pi + angle
    elif angle > np.pi:
        return angle - 2 * np.pi
    else:
        return angle


def q1d():
    R = np.array([[0.813797681, -0.440969611, 0.378522306],
                  [0.46984631, 0.882564119, 0.0180283112],
                  [-0.342020143, 0.163175911, 0.92541657]])
    yaw1, yaw2, pitch1, pitch2, roll1, roll2 = rot_mat_to_euler_angles(R)
    print("Quesion 1d: ")
    print("Two possible angle values: ")
    print(RAD_TO_DEG * roll1, RAD_TO_DEG * pitch1, RAD_TO_DEG * yaw1)
    print(RAD_TO_DEG * roll2, RAD_TO_DEG * pitch2, RAD_TO_DEG * yaw2)

    print("Rot1:")
    res1 = euler_to_rot_mat(yaw1, pitch1, roll1)
    print("Test1: ", np.allclose(R, res1, atol=10**(-5)))
    print("Rot2: ")
    res2 = euler_to_rot_mat(yaw2, pitch2, roll2)
    print("Test2: ", np.allclose(R, res2, atol=10**(-5)))


def q2():
    R_global2cam = np.array([[0.5363, -0.8440, 0.],
                             [0.8440, 0.5363, 0],
                             [0, 0, 1]])
    t_cam2global = np.array([-451.2459, 257.0322, 400]).reshape(3, 1)
    t_global2cam = - R_global2cam @ t_cam2global

    l_global = np.array([450, 400, 50]).reshape(3, 1)

    l_cam = R_global2cam @ l_global + t_global2cam
    # l_cam = R_global2cam @ l_global - R_global2cam @ t_cam2global
    # l_cam = R_global2cam @ (l_global - t_cam2global)

    print("l_cam is: ")
    print(l_cam)

    print()
    print("Check with inverse on l_cam:")
    print("Test inv: ", np.allclose(np.linalg.inv(R_global2cam) @ l_cam + t_cam2global, l_global, atol=TOLERANCE))

    return


def create_composed_T_arr(T, arr_len=10):
    arr = [None] * arr_len
    arr[0] = np.eye(3)
    for i in range(1, arr_len):
        arr[i] = T @ arr[i - 1]

    return arr


def get_locations_from_T(Ts_lst):
    return np.array([T[0:2, 2] for T in Ts_lst])


def get_headings(Ts_lst):
    return np.array([T[0:2, 1] for T in Ts_lst])


def draw_locs_and_headings(robot_locations, robot_headings, xlim_bot, xlim_top, ylim_bot, ylim_top, plt_heading=True, plt_loc=True, num=None,):
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.xlim(xlim_bot, xlim_top)
    plt.ylim(ylim_bot, ylim_top)
    num_robots = num if num is not None else len(robot_locations)
    robot_locations = robot_locations[:num_robots, :]
    robot_headings = robot_headings[:num_robots, :]
    # Plot robot locations
    if plt_loc:
        ax.scatter(robot_locations[:, 0], robot_locations[:, 1], label='Robot Location', color='blue')

    arrow_len = 0.5
    # Plot heading vectors
    if plt_heading:
        for i, (x, y) in enumerate(robot_locations):
            dx, dy = robot_headings[i] * arrow_len
            ax.arrow(x, y, dx, dy, width=0.0005, head_width=0.003, head_length=0.1, fc='red', ec='red')

    # Set labels and legend
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Robot Location with Headings')

    ax.legend()

    # Show the plot
    fig.savefig("./robots_loc")


def draw_robot_pose(ax, robot_locations, robot_headings, type, clr):
    ax.scatter(robot_locations[:, 0], robot_locations[:, 1], label=f'Robot {type} Location', color=clr)

    arrow_len = 0.5
    # Plot heading vectors
    for i, (x, y) in enumerate(robot_locations):
        dx, dy = robot_headings[i] * arrow_len
        ax.arrow(x, y, dx, dy, width=0.0005, head_width=0.003, head_length=0.1, fc='red', ec='red')


def draw_robot_pose_actual_and_commanded(commanded_loc, commanded_headings, actual_loc, actual_headings, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    # plt.xlim(xlim_bot, xlim_top)
    # plt.ylim(ylim_bot, ylim_top)

    draw_robot_pose(ax, commanded_loc, commanded_headings, "Commanded", 'green')
    draw_robot_pose(ax, actual_loc, actual_headings, "Actual", 'blue')
    # Set labels and legend
    ax.set_xlabel('X_axis (Meters)')
    ax.set_ylabel('Y-axis (Meters)')
    ax.set_title(f'Robot Location with Headings\n{title}')

    ax.legend()

    # Show the plot
    fig.savefig("./robots_loc")


def q3b():
    commanded_T = np.array([[1, 0, 0.],
                            [0, 1, 1],
                            [0, 0, 1]])
    actual_angle = 1 * DEG_TO_RAD
    actual_T = np.array([[np.cos(actual_angle), -np.sin(actual_angle), 0.],
                         [np.sin(actual_angle), np.cos(actual_angle),  1.1],
                         [0,            0,       1]])

    commanded_T_lst = create_composed_T_arr(commanded_T)
    actual_T_lst = create_composed_T_arr(actual_T)

    commanded_loc = get_locations_from_T(commanded_T_lst)
    actual_loc = get_locations_from_T(actual_T_lst)

    commanded_headings = get_headings(commanded_T_lst)
    actual_headings = get_headings(actual_T_lst)

    draw_locs_and_headings(commanded_loc, commanded_headings, -1, 1, -0.5, 13, plt_heading=True, plt_loc=True, num=None)
    draw_locs_and_headings(actual_loc, actual_headings, -3, 3, -0.5, 13, plt_heading=True, plt_loc=True, num=None)

    loc_diff = actual_loc[-1] - commanded_loc[-1]
    heading_err = np.arccos(np.dot(actual_headings[-1, :], commanded_headings[-1, :]))
    x_err = loc_diff[0]
    y_err = loc_diff[1]

    err_title = "Dead Reckoning: x error: {:.3f} | y error : {:.3f} | Heading error: {:.3f} (Deg: {:.2f})".format(x_err, y_err, heading_err, heading_err * RAD_TO_DEG)
    draw_robot_pose_actual_and_commanded(commanded_loc, commanded_headings, actual_loc, actual_headings, err_title)