"""
    This sample demonstrates how to capture a live depth map
    with the ZED SDK and project it out as a deformed plane.
"""
import sys
from simple_viewer import GLViewer
import pyzed.sl as sl


def setup_zed_camera():
    init_params = sl.InitParameters(depth_mode=sl.DEPTH_MODE.NEURAL_PLUS,
                                    coordinate_units=sl.UNIT.METER,
                                    coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
                                    camera_resolution=sl.RESOLUTION.HD1080,
                                    )
    zed = sl.Camera()
    init_params.depth_maximum_distance = 10.0
    init_params.depth_minimum_distance = 0.2
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


def main():
    print("Running Depth Projection sample")
    cam_pose = sl.Pose()
    res = sl.Resolution()
    runtime_params = sl.RuntimeParameters()
    runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    runtime_params.enable_fill_mode = True
    res.width = 1920
    res.height = 1080
    zed = setup_zed_camera()
    v_fov = zed.get_camera_information(
    ).camera_configuration.calibration_parameters.left_cam.v_fov
    # Create OpenGL viewer
    viewer = GLViewer(v_fov)
    viewer.init(1, sys.argv)
    image = sl.Mat()
    depth_map = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C1, sl.MEM.CPU)

    while viewer.is_available():
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(
                depth_map, sl.MEASURE.DEPTH, sl.MEM.CPU, res)
            zed.get_position(cam_pose, sl.REFERENCE_FRAME.WORLD)
            # Print camera position in world coordinates
            zed.retrieve_image(
                image, sl.VIEW.LEFT, sl.MEM.CPU, res)
            viewer.updateData(
                cam_pose.pose_data(), image, depth_map)

    viewer.exit()
    zed.close()


if __name__ == "__main__":
    main()
