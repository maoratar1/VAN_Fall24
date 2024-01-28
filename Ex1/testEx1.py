import Ex1
import numpy as np


def check_inv(yaw, pitch, roll):
    yaw1, yaw2, pitch1, pitch2, roll1, roll2 = Ex1.rot_mat_to_euler_angles(Ex1.euler_to_rot_mat(yaw, pitch, roll))

    print("Check Inversion: yaw {}, {} | pitch {}, {} | roll {}, {}".format(yaw1, yaw2,
                                                                            pitch1, pitch2,
                                                                            roll1, roll2))

    print("Check Inversion Diff: yaw {}, {} | pitch {}, {} | roll {}, {}".format(yaw - yaw1, yaw - yaw2,
                                                                                 pitch - pitch1, pitch - pitch2,
                                                                                 roll - roll1, roll - roll2))


def check_rot():
    yaw = 0  # psi
    pitch = np.pi / 3.  # theta
    roll = 0.  # phi

    print("Rotation Matrix for yaw {} pitch {} roll {}".format(yaw, pitch, roll))
    print(Ex1.euler_to_rot_mat(yaw, pitch, roll))

    print()
    print("Rotation Matrix for yaw {} pitch {} roll {}".format(np.pi - yaw, np.pi - pitch, np.pi - roll))
    print(Ex1.euler_to_rot_mat(yaw, np.pi - pitch, roll))


def check_angles():
    rot_mat = np.array([[0.5, -0.1464, 0.8536],
                        [0.5, 0.8536, -0.1464],
                        [-0.7071, 0.5, 0.5]])
    yaw1, yaw2, pitch1, pitch2, roll1, roll2 = Ex1.rot_mat_to_euler_angles(rot_mat)
    print("Check Inversion: yaw {}, {} | pitch {}, {} | roll {}, {}".format(yaw1, yaw2,
                                                                            pitch1, pitch2,
                                                                            roll1, roll2))
    print("Original mat:")
    print(rot_mat)
    print("mat from 1: ")
    res1 = Ex1.euler_to_rot_mat(yaw1, pitch1, roll1)
    print(res1)
    print("Test: ", np.allclose(rot_mat, res1, atol=10**(-4)))
    print()
    print("mat from 2: ")
    res2 = Ex1.euler_to_rot_mat(yaw2, pitch2, roll2)
    print(res2)
    print("Test: ", np.allclose(rot_mat, res2, atol=10**(-4)))


if __name__ == '__main__':
    yaw = np.pi / 7.  # psi
    pitch = np.pi / 5.  # theta
    roll = np.pi / 4.  # phi

    # yaw = 0  # psi
    # pitch = np.pi / 3.  # theta
    # roll = 0.  # phi
    # check_rot()
    # check_inv(yaw, pitch, roll)

    # check_inv(yaw, pitch, roll)

    check_angles()
