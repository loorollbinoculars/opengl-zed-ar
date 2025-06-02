"""Basic GLUT viewer for ZED data."""

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import math
from threading import Lock
import numpy as np

import pyzed.sl as sl
from objects import FullScreenQuad, Cube

M_PI = 3.1415926


class GLViewer:
    def __init__(self, camera_v_fov):
        self.available = False
        self.mutex = Lock()
        self.camera = CameraGL(camera_v_fov)
        self.background = FullScreenQuad(
            sl.Resolution(1920, 1080), camera_v_fov)
        self.draw_background = True
        self.draw_cubes = True
        self.cubes = [Cube(False, 0.1, self.camera.proj, [i, j])
                      for i in range(-3, 3) for j in range(-3, 3)]

    def init(self, _argc, _argv):  # _params = sl.CameraParameters
        glutInit(_argc, _argv)
        wnd_w = int(glutGet(GLUT_SCREEN_WIDTH)*0.9)
        wnd_h = int(glutGet(GLUT_SCREEN_HEIGHT) * 0.9)
        glutInitWindowSize(wnd_w, wnd_h)
        glutInitWindowPosition(int(wnd_w*0.05), int(wnd_h*0.05))

        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutCreateWindow(b"ZED Depth Sensing")
        glViewport(0, 0, wnd_w, wnd_h)

        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_CONTINUE_EXECUTION)

        glEnable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        self.bckgrnd_clr = np.array([1/255., 1/255., 1/255.])
        # Register GLUT callback functions
        glutDisplayFunc(self.draw_callback)
        glutIdleFunc(self.idle)
        glutCloseFunc(self.exit)
        glutKeyboardFunc(self.keyPressedCallback)

        glutReshapeFunc(self.on_resize)
        self.available = True

        # Initialise objects:
        self.background.init()
        for cube in self.cubes:
            cube.init()

    def is_available(self):
        if self.available:
            glutMainLoopEvent()
        return self.available

    def keyPressedCallback(self, key, x, y):
        if ord(key) == 27:
            self.exit()
        if ord(key) == ord(' '):
            self.draw_background = not self.draw_background
        if ord(key) == ord('c'):
            self.draw_cubes = not self.draw_cubes

    def updateData(self, extrinsic_matrix: sl.Transform, mat, depth):
        self.mutex.acquire()
        self.background.update(mat, depth)
        extrinsic_matrix.inverse()  # TODO this should be done one level up.
        self.camera.viewMatrix = extrinsic_matrix.m
        self.mutex.release()

    def idle(self):
        if self.available:
            glutPostRedisplay()

    def exit(self):
        if self.available:
            self.available = False

    def on_resize(self, Width, Height):
        glViewport(0, 0, Width, Height)
        self.camera.proj = self.camera.build_perspective(
            self.camera.fov, Height / Width, self.camera.znear, self.camera.zfar)

    def draw_callback(self):
        if self.available:

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(
                self.bckgrnd_clr[0], self.bckgrnd_clr[1], self.bckgrnd_clr[2], 1.)

            with self.mutex:
                self.background.draw() if self.draw_background else None
                if self.draw_cubes:
                    for cube in self.cubes:
                        cube.viewMatrix = self.camera.viewMatrix
                        cube.draw()
            glutSwapBuffers()
            glutPostRedisplay()


class CameraGL:
    """Simple class to hold the projection and view matrices for the scene."""

    def __init__(self, camera_v_fov):
        self.znear = 0.001
        self.zfar = 10.0
        self.fov = camera_v_fov
        self.viewMatrix = np.eye(4)  # View matrix in practice
        self.projectionMatrix = sl.Matrix4f()
        self.vpMatrix = sl.Matrix4f()
        self.projectionMatrix.set_identity()
        # TODO: don't hardcode these.
        self.proj = self.build_perspective(camera_v_fov,  16/9, 0.2, 10.0)

    def update(self, new_extrinsic):
        """Update the view-projection matrix based on the current extrinsic matrix."""
        self.viewMatrix = new_extrinsic

    @staticmethod
    def build_perspective(fov_y, aspect, z_near, z_far):
        """Return a 4x4 perspective projection matrix as np.float32."""
        f = 1.0 / math.tan(math.radians(fov_y) * 0.5)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (z_far + z_near) / (z_near - z_far)
        proj[2, 3] = (2 * z_far * z_near) / (z_near - z_far)
        proj[3, 2] = -1.0
        return proj
