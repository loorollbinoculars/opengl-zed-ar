"""
    This sample demonstrates how to capture a live depth map
    with the ZED SDK and project it out as a deformed plane.
"""
import numpy as np
import sys
from simple_viewer import GLViewer
import pyzed.sl as sl
import argparse


def setup_zed_camera():
    init_params = sl.InitParameters(depth_mode=sl.DEPTH_MODE.NEURAL_PLUS,
                                    coordinate_units=sl.UNIT.METER,
                                    coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
                                    camera_resolution=sl.RESOLUTION.HD1080,
                                    )
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    pos_params = sl.PositionalTrackingParameters()
    pos_params.set_floor_as_origin = True
    zed.enable_positional_tracking(pos_params)
    zed.grab()
    floor = sl.Plane()
    reset_plane = sl.Transform()
    zed.find_floor_plane(
        floor, reset_plane)
    zed.reset_positional_tracking(reset_plane)

    return zed


def build_projection_matrix(zed_camera_params: sl.CameraParameters):
    zed_camera_params.cx
    zed_camera_params.cy
    zed_camera_params.fx
    zed_camera_params.fy
    zed_camera_params.image_size.width
    zed_camera_params.image_size.height
    # Build the projection matrix based on the camera parameters
    projection_matrix = np.zeros((4, 4), dtype=np.float32)
    projection_matrix[0, 0] = 2 * zed_camera_params.fx / \
        zed_camera_params.image_size.width
    projection_matrix[1, 1] = 2 * zed_camera_params.fy / \
        zed_camera_params.image_size.height
    projection_matrix[0, 2] = (2 * zed_camera_params.cx) / \
        zed_camera_params.image_size.width - 1
    projection_matrix[1, 2] = (2 * zed_camera_params.cy) / \
        zed_camera_params.image_size.height - 1
    projection_matrix[2, 2] = -1
    projection_matrix[2, 3] = -1
    projection_matrix[3, 2] = -zed_camera_params.z_far / \
        (zed_camera_params.z_far - zed_camera_params.z_near)
    projection_matrix[3, 3] = 0
    return projection_matrix


def main():
    print("Running Depth Projection sample")
    cam_pose = sl.Pose()
    res = sl.Resolution()
    runtime_params = sl.RuntimeParameters()
    runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    res.width = 1920
    res.height = 1080
    zed = setup_zed_camera()
    v_fov = zed.get_camera_information(
    ).camera_configuration.calibration_parameters.left_cam.v_fov
    # Create OpenGL viewer
    viewer = GLViewer(v_fov)
    viewer.init(1, sys.argv)
    image = sl.Mat()
    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C1, sl.MEM.CPU)

    while viewer.is_available():
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(
                point_cloud, sl.MEASURE.DEPTH, sl.MEM.CPU, res)
            zed.get_position(cam_pose, sl.REFERENCE_FRAME.WORLD)
            # Print camera position in world coordinates
            zed.retrieve_image(
                image, sl.VIEW.LEFT, sl.MEM.CPU, res)
            viewer.updateData(
                cam_pose.pose_data(), image, point_cloud)
    viewer.exit()
    zed.close()


if __name__ == "__main__":
    main()
