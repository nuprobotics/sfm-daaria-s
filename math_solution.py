import numpy as np


def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
):
    """
    :param camera_matrix: first and second camera matrix, np.ndarray 3x3
    :param camera_position1: first camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation1: first camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param camera_position2: second camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation2: second camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param image_points1: points in the first image, np.ndarray Nx2
    :param image_points2: points in the second image, np.ndarray Nx2
    :return: triangulated points, np.ndarray Nx3
    """

    pos_cam1 = camera_position1.reshape((3, 1))
    pos_cam2 = camera_position2.reshape((3, 1))

    rot_to_cam1 = camera_rotation1.T
    rot_to_cam2 = camera_rotation2.T

    trans_cam1 = -rot_to_cam1 @ pos_cam1
    trans_cam2 = -rot_to_cam2 @ pos_cam2

    ext_matrix1 = np.hstack((rot_to_cam1, trans_cam1))
    ext_matrix2 = np.hstack((rot_to_cam2, trans_cam2))

    proj_matrix1 = camera_matrix @ ext_matrix1
    proj_matrix2 = camera_matrix @ ext_matrix2

    points_3D = []
    for pt1, pt2 in zip(image_points1, image_points2):
        system_matrix = np.vstack([
            pt1[0] * proj_matrix1[2, :] - proj_matrix1[0, :],
            pt1[1] * proj_matrix1[2, :] - proj_matrix1[1, :],
            pt2[0] * proj_matrix2[2, :] - proj_matrix2[0, :],
            pt2[1] * proj_matrix2[2, :] - proj_matrix2[1, :]
        ])

        _, _, vh = np.linalg.svd(system_matrix)
        X_homogeneous = vh[-1]
        X_homogeneous /= X_homogeneous[3]

        points_3D.append(X_homogeneous[:3])

    return np.array(points_3D)