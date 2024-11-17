import numpy as np
import cv2
import typing
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import yaml


def get_matches(image1, image2) -> typing.Tuple[
    typing.Sequence[cv2.KeyPoint], typing.Sequence[cv2.KeyPoint], typing.Sequence[cv2.DMatch]]:
    # Initialize SIFT and convert images to grayscale
    sift = cv2.SIFT_create()
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors
    kp1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    kp2, descriptors2 = sift.detectAndCompute(img2_gray, None)

    # Use BFMatcher with k-NN
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_1_to_2: typing.Sequence[typing.Sequence[cv2.DMatch]] = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches_2_to_1: typing.Sequence[typing.Sequence[cv2.DMatch]] = bf.knnMatch(descriptors2, descriptors1, k=2)

    # Implement k-ratio test with k=0.75
    k_ratio = 0.75
    good_matches_1_to_2 = [
        m for m, n in matches_1_to_2 if m.distance < k_ratio * n.distance
    ]
    good_matches_2_to_1 = [
        m for m, n in matches_2_to_1 if m.distance < k_ratio * n.distance
    ]

    # Perform left-right consistency check
    matches = []
    for match1 in good_matches_1_to_2:
        for match2 in good_matches_2_to_1:
            if match1.queryIdx == match2.trainIdx and match1.trainIdx == match2.queryIdx:
                matches.append(match1)
                break

    return kp1, kp2, matches


def get_second_camera_position(kp1, kp2, matches, camera_matrix):
    coordinates1 = np.array([kp1[match.queryIdx].pt for match in matches])
    coordinates2 = np.array([kp2[match.trainIdx].pt for match in matches])
    E, mask = cv2.findEssentialMat(coordinates1, coordinates2, camera_matrix)
    _, R, t, mask = cv2.recoverPose(E, coordinates1, coordinates2, camera_matrix)
    return R, t, E


# Task 3
def triangulation(
        camera_matrix: np.ndarray,
        camera1_translation_vector: np.ndarray,
        camera1_rotation_matrix: np.ndarray,
        camera2_translation_vector: np.ndarray,
        camera2_rotation_matrix: np.ndarray,
        kp1: typing.Sequence[cv2.KeyPoint],
        kp2: typing.Sequence[cv2.KeyPoint],
        matches: typing.Sequence[cv2.DMatch]
) -> np.ndarray:
    # Compute the projection matrix for Camera 1
    camera1_projection_matrix = camera_matrix @ np.hstack((
        camera1_rotation_matrix,
        camera1_translation_vector
    ))  # P1 = K[R|t]

    # Compute the projection matrix for Camera 2
    camera2_projection_matrix = camera_matrix @ np.hstack((
        camera2_rotation_matrix,
        camera2_translation_vector
    ))  # P2 = K[R|t]

    # Extract matched keypoints from both images
    points1 = np.array([kp1[match.queryIdx].pt for match in matches], dtype=np.float32).T
    points2 = np.array([kp2[match.trainIdx].pt for match in matches], dtype=np.float32).T

    # Perform triangulation
    points_4d = cv2.triangulatePoints(
        camera1_projection_matrix,
        camera2_projection_matrix,
        points1,
        points2
    )

    # Convert homogeneous coordinates to 3D points
    points_3d = (points_4d[:3] / points_4d[3]).T  # Divide by w to get [X, Y, Z]

    return points_3d


# Task 4
def resection(
        image1,
        image2,
        camera_matrix,
        matches,
        points_3d
):
    kps_image1, kps_image2, refined_matches = get_matches(image1, image2)
    point_map = {match.queryIdx: points_3d[i] for i, match in enumerate(matches)}
    object_points = []
    image_points = []
    for m in refined_matches:
        point_idx = m.queryIdx
        if point_idx in point_map:
            object_points.append(point_map[point_idx])
            image_points.append(kps_image2[m.trainIdx].pt)
    object_points = np.array(object_points)
    image_points = np.array(image_points)
    _, rotation_vec, translation_vec, _ = cv2.solvePnPRansac(
        object_points, image_points, camera_matrix, np.zeros(5)
    )

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    return rotation_matrix, translation_vec


def convert_to_world_frame(translation_vector, rotation_matrix):
    world_rotation_matrix = rotation_matrix.T
    world_position = -world_rotation_matrix @ translation_vector

    return world_position, world_rotation_matrix


def visualisation(
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        camera_position3: np.ndarray,
        camera_rotation3: np.ndarray,
):
    def plot_camera(ax, position, direction, label):
        color_scatter = 'blue' if label != 'Camera 3' else 'green'
        # print(position)
        ax.scatter(position[0][0], position[1][0], position[2][0], color=color_scatter, s=100)
        color_quiver = 'red' if label != 'Camera 3' else 'magenta'

        ax.quiver(position[0][0], position[1][0], position[2][0], direction[0], direction[1], direction[2],
                  length=1, color=color_quiver, arrow_length_ratio=0.2)
        ax.text(position[0][0], position[1][0], position[2][0], label, color='black')

    camera_positions = [camera_position1, camera_position2, camera_position3]
    camera_directions = [camera_rotation1[:, 2], camera_rotation2[:, 2], camera_rotation3[:, 2]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_camera(ax, camera_positions[0], camera_directions[0], 'Camera 1')
    plot_camera(ax, camera_positions[1], camera_directions[1], 'Camera 2')
    plot_camera(ax, camera_positions[2], camera_directions[2], 'Camera 3')

    initial_elev = 0
    initial_azim = 270

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=initial_elev, azim=initial_azim)

    ax.set_xlim([-1.50, 2.0])
    ax.set_ylim([-.50, 3.0])
    ax.set_zlim([-.50, 3.0])

    ax_elev_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
    elev_slider = Slider(ax_elev_slider, 'Elev', 0, 360, valinit=initial_elev)

    ax_azim_slider = plt.axes([0.1, 0.05, 0.65, 0.03])
    azim_slider = Slider(ax_azim_slider, 'Azim', 0, 360, valinit=initial_azim)

    def update(val):
        elev = elev_slider.val
        azim = azim_slider.val
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()

    elev_slider.on_changed(update)
    azim_slider.on_changed(update)

    plt.show()


def main():
    image1 = cv2.imread('./images/image0.jpg')
    image2 = cv2.imread('./images/image1.jpg')
    image3 = cv2.imread('./images/image2.jpg')
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    camera_matrix = np.array(config["camera_matrix"], dtype=np.float32, order='C')

    key_points1, key_points2, matches_1_to_2 = get_matches(image1, image2)
    R2, t2, E = get_second_camera_position(key_points1, key_points2, matches_1_to_2, camera_matrix)
    triangulated_points = triangulation(
        camera_matrix,
        np.array([0, 0, 0]).reshape((3, 1)),
        np.eye(3),
        t2,
        R2,
        key_points1,
        key_points2,
        matches_1_to_2
    )

    R3, t3 = resection(image1, image3, camera_matrix, matches_1_to_2, triangulated_points)
    camera_position1, camera_rotation1 = convert_to_world_frame(np.array([0, 0, 0]).reshape((3, 1)), np.eye(3))
    camera_position2, camera_rotation2 = convert_to_world_frame(t2, R2)
    camera_position3, camera_rotation3 = convert_to_world_frame(t3, R3)
    visualisation(
        camera_position1,
        camera_rotation1,
        camera_position2,
        camera_rotation2,
        camera_position3,
        camera_rotation3
    )


if __name__ == "__main__":
    main()
