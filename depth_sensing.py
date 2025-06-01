########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    This sample demonstrates how to capture a live 3D point cloud   
    with the ZED SDK and display the result in an OpenGL window.    
"""

import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import argparse


def main(opt):
    print("Running Depth Sensing sample ... Press 'Esc' to quit\nPress 's' to save the point cloud")

    init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.NEURAL_LIGHT,
                             coordinate_units=sl.UNIT.METER,
                             coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
                             camera_resolution=sl.RESOLUTION.HD1080,)
    zed = sl.Camera()
    pos_params = sl.PositionalTrackingParameters()
    status = zed.open(init)
    pos_params.set_floor_as_origin = True
    zed.enable_positional_tracking(pos_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    res = sl.Resolution()
    res.width = 1920
    res.height = 1080
    v_fov = zed.get_camera_information(
    ).camera_configuration.calibration_parameters.left_cam.v_fov
    # Create OpenGL viewer
    viewer = gl.GLViewer(v_fov)
    viewer.init(1, sys.argv, res)

    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image = sl.Mat()
    while viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(
                point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, res)
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, res)
            viewer.updateData(point_cloud, image)
            if (viewer.save_data == True):
                point_cloud_to_save = sl.Mat()
                zed.retrieve_measure(point_cloud_to_save,
                                     sl.MEASURE.XYZRGBA, sl.MEM.CPU)
                err = point_cloud_to_save.write('Pointcloud.ply')
                if (err == sl.ERROR_CODE.SUCCESS):
                    print("Current .ply file saving succeed")
                else:
                    print("Current .ply file failed")
                viewer.save_data = False
    viewer.exit()
    zed.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str,
                        help='Path to an .svo file, if you want to replay it', default='')
    parser.add_argument('--ip_address', type=str,
                        help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default='')
    parser.add_argument('--resolution', type=str,
                        help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default='')
    opt = parser.parse_args()
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main(opt)
