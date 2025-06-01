"""
    This sample demonstrates how to capture a live depth map
    with the ZED SDK and project it out as a deformed plane.
"""
import numpy as np
import sys
import ogl_viewer.viewer as gl
from viewer import GLViewer
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
    print(floor.get_center())
    zed.find_floor_plane(
        floor, reset_plane)
    print(floor.get_center())

    return zed


def main():
    print("Running Depth Projection sample")
    cam_pose = sl.Pose()
    view_matrix = sl.Transform()
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
    viewer.init(1, sys.argv, res)
    image = sl.Mat()
    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C3, sl.MEM.CPU)

    while viewer.is_available():
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(
                point_cloud, sl.MEASURE.XYZ, sl.MEM.CPU, res)
            zed.get_position(cam_pose, sl.REFERENCE_FRAME.WORLD)
            print(zed.get_positional_tracking_status().tracking_fusion_status)
            print(cam_pose.pose_data().m[:3, 3])
            zed.retrieve_image(
                image, sl.VIEW.LEFT, sl.MEM.CPU, res)
            viewer.updateData(
                point_cloud, cam_pose.pose_data(), image)
    viewer.exit()
    zed.close()


if __name__ == "__main__":
    main()
