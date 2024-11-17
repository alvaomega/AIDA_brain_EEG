import cv2
import numpy as np
import math
from helpers import relative, relativeT

def gaze(frame, points):
    """
    The gaze function gets an image and face landmarks from mediapipe framework.
    The function draws the gaze direction into the frame and calculates the length of the gaze line based on the gaze angle.
    """

    # 2D image points.
    image_points = np.array([
        relative(points.landmark[4], frame.shape),    # Nose tip
        relative(points.landmark[152], frame.shape),  # Chin
        relative(points.landmark[263], frame.shape),  # Left eye left corner
        relative(points.landmark[33], frame.shape),   # Right eye right corner
        relative(points.landmark[287], frame.shape),  # Left Mouth corner
        relative(points.landmark[57], frame.shape)    # Right mouth corner
    ], dtype="double")

    # 2D image points with Z=0 (for 3D affine estimation).
    image_points1 = np.array([
        relativeT(points.landmark[4], frame.shape),    # Nose tip
        relativeT(points.landmark[152], frame.shape),  # Chin
        relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
        relativeT(points.landmark[33], frame.shape),   # Right eye, right corner
        relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
        relativeT(points.landmark[57], frame.shape)    # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0, -63.6, -12.5),          # Chin
        (-43.3, 32.7, -26),         # Left eye, left corner
        (43.3, 32.7, -26),          # Right eye, right corner
        (-28.9, -28.9, -24.1),      # Left Mouth corner
        (28.9, -28.9, -24.1)        # Right mouth corner
    ])

    # Eye ball center positions in 3D model coordinates.
    Eye_ball_center_right = np.array([-29.05, 32.7, -39.5])
    Eye_ball_center_left = np.array([29.05, 32.7, -39.5])

    # Camera internals.
    focal_length = frame.shape[1]  # Approximate focal length.
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion.

    # Solve PnP to find rotation and translation vectors.
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # 2D pupil location.
    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)

    # Estimate the affine transformation from image to world coordinates.
    retval, transformation, inliers = cv2.estimateAffine3D(image_points1, model_points)

    if transformation is not None:
        # Project pupil image point into 3D world point.
        pupil_world_cord = transformation @ np.array([left_pupil[0], left_pupil[1], 0, 1]).T

        # Calculate the gaze direction vector.
        gaze_direction = pupil_world_cord[:3] - Eye_ball_center_left

        # Normalize the gaze direction vector.
        gaze_direction_norm = gaze_direction / np.linalg.norm(gaze_direction)

        # Define an arbitrary scale for the gaze vector (e.g., looking 100 units ahead).
        gaze_length = 100.0
        gaze_target_world = Eye_ball_center_left + gaze_direction_norm * gaze_length

        # Calculate the gaze angle (theta) relative to the optical axis (assumed to be along Z-axis).
        theta = math.acos(gaze_direction_norm[2])  # Angle between gaze direction and Z-axis.

        # Project the gaze target point onto the image plane.
        gaze_target_world_point = gaze_target_world.reshape(-1, 3)
        eye_pupil2D, _ = cv2.projectPoints(
            gaze_target_world_point, rotation_vector, translation_vector, camera_matrix, dist_coeffs
        )

        # Project the eye position onto the image plane.
        eye_position_world_point = Eye_ball_center_left.reshape(-1, 3)
        eye_position2D, _ = cv2.projectPoints(
            eye_position_world_point, rotation_vector, translation_vector, camera_matrix, dist_coeffs
        )

        # Get the image points.
        p1 = (int(eye_position2D[0][0][0]), int(eye_position2D[0][0][1]))
        p2 = (int(eye_pupil2D[0][0][0]), int(eye_pupil2D[0][0][1]))

        # Draw the gaze line on the frame.
        cv2.line(frame, p1, p2, (0, 0, 255), 2)

        # Calculate the length of the line in pixels.
        line_length_pixels = np.linalg.norm(np.array(p2) - np.array(p1))

        # Alternatively, calculate the length based on the gaze angle.
        # Since the gaze angle is relative to the optical axis, and the focal length is in pixels:
        line_length_angle = focal_length * math.tan(theta)

        # Print or store the length if needed.
        print(f"Gaze line length (pixels): {line_length_pixels}")
        print(f"Gaze line length based on angle (pixels): {line_length_angle}")

        # Display the line length on the frame.
        cv2.putText(frame, f"Length: {line_length_pixels:.2f}px", (p1[0], p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Add logic to determine if the person is not looking at the screen.
        if line_length_pixels > 75:
            person_is_looking_at_screen = False
            cv2.putText(frame, "Person is not looking at screen", (p1[0], p1[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            person_is_looking_at_screen = True

    else:
        line_length_pixels = 100
        person_is_looking_at_screen = False  # Or False, depending on your desired behavior when transformation fails
        print(f"Gaze line not visible set: {line_length_pixels}")

    # Return the modified frame and the logical variable.
    return frame, person_is_looking_at_screen