from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from pyglm.glm import perspective

import ctypes
import sys
import math
from threading import Lock
import numpy as np
import array

import pyzed.sl as sl
from simple3Dobject import Simple3DObject
from shader import Shader


M_PI = 3.1415926

VERTEX_SHADER = """
# version 330 core
layout(location = 0) in vec3 in_Vertex;
layout(location = 1) in vec4 in_Color;
uniform mat4 u_mvpMatrix;
out vec4 b_color;
void main() {
    b_color = in_Color;
    gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);
}
"""

FRAGMENT_SHADER = """
# version 330 core
in vec4 b_color;
layout(location = 0) out vec4 out_Color;
void main() {
   out_Color = b_color;
}
"""

POINTCLOUD_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec4 in_VertexRGBA;
uniform mat4 u_mvpMatrix;
out vec4 b_color;
void main() {
    uint vertexColor = floatBitsToUint(in_VertexRGBA.w); // Extract RGB bits (4 x 1 byte colors) into a uint (32-bit unsigned integer)
    vec3 clr_int = vec3((vertexColor & uint(0x000000FF)), (vertexColor & uint(0x0000FF00)) >> 8, (vertexColor & uint(0x00FF0000)) >> 16);
    b_color = vec4(clr_int.r / 255.0f, clr_int.g / 255.0f, clr_int.b / 255.0f, 1.f);
    gl_Position = u_mvpMatrix * vec4(in_VertexRGBA.xyz, 1);
}
"""

POINTCLOUD_FRAGMENT_SHADER = """
#version 330 core
in vec4 b_color;
layout(location = 0) out vec4 out_Color;
void main() {
   out_Color = b_color;
}
"""
SCREEN_VERT = """
#version 330 core
layout(location = 0) in vec2 in_Pos;   // −1..+1 clip-space
layout(location = 1) in vec2 in_UV;    // 0..1 texture coords
out vec2 uv;
void main() {
    uv = in_UV;
    uv.y = 1.0 - uv.y;  // flip the y coordinate for OpenGL
    uv.x = 1.0 - uv.x;  // flip the x coordinate for OpenGL
    gl_Position = vec4(in_Pos, 0.0, 1.0);   // already in clip-space
}
"""

SCREEN_FRAG = """
#version 330 core
in vec2 uv;
uniform sampler2D u_tex;
out vec4 out_Color;
void main() {
    out_Color = texture(u_tex, uv);
}
"""


class GLViewer:
    def __init__(self, camera_v_fov):
        self.available = False
        self.mutex = Lock()
        self.camera = CameraGL(camera_v_fov)
        self.wheelPosition = 0.
        self.mouse_button = [False, False]
        self.mouseCurrentPosition = [0., 0.]
        self.previousMouseMotion = [0., 0.]
        self.mouseMotion = [0., 0.]
        self.zedModel = Simple3DObject(True)
        self.cube = Simple3DObject(True)

        self.point_cloud = Simple3DObject(False, 4)
        self.save_data = False

    def init(self, _argc, _argv, res):  # _params = sl.CameraParameters
        glutInit(_argc, _argv)
        wnd_w = int(glutGet(GLUT_SCREEN_WIDTH)*0.9)
        wnd_h = int(glutGet(GLUT_SCREEN_HEIGHT) * 0.9)
        glutInitWindowSize(wnd_w, wnd_h)
        glutInitWindowPosition(int(wnd_w*0.05), int(wnd_h*0.05))

        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutCreateWindow("ZED Depth Sensing")
        glViewport(0, 0, wnd_w, wnd_h)

        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_CONTINUE_EXECUTION)

        glEnable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Compile and create the shader for 3D objects
        self.shader_image = Shader(VERTEX_SHADER, FRAGMENT_SHADER)
        self.shader_image_MVP = glGetUniformLocation(
            self.shader_image.get_program_id(), "u_mvpMatrix")

        self.shader_pc = Shader(POINTCLOUD_VERTEX_SHADER,
                                POINTCLOUD_FRAGMENT_SHADER)
        self.shader_pc_MVP = glGetUniformLocation(
            self.shader_pc.get_program_id(), "u_mvpMatrix")

        self.bckgrnd_clr = np.array([1/255., 1/255., 1/255.])

        # Create the camera model
        Z_ = -0.15
        FOV = self.camera.vfov
        Y_ = Z_ * math.tan(FOV * M_PI / 180. / 2.)
        X_ = Y_ * 16./9.

        A = np.array([0, 0, 0])
        B = np.array([X_, Y_, Z_])
        C = np.array([-X_, Y_, Z_])
        D = np.array([-X_, -Y_, Z_])
        E = np.array([X_, -Y_, Z_])

        lime_clr = np.array([217 / 255, 255/255, 66/255])

        self.zedModel.add_line(A, B, lime_clr)
        self.zedModel.add_line(A, C, lime_clr)
        self.zedModel.add_line(A, D, lime_clr)
        self.zedModel.add_line(A, E, lime_clr)

        self.zedModel.add_line(B, C, lime_clr)
        self.zedModel.add_line(C, D, lime_clr)
        self.zedModel.add_line(D, E, lime_clr)
        self.zedModel.add_line(E, B, lime_clr)

        self.zedModel.set_drawing_type(GL_LINES)
        self.zedModel.push_to_GPU()

        cube_lines = [
            [1, 1, 1], [1, 1, -1], [1, -1, -1], [1, -1, 1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, -1], [-1, -1, 1]
        ]
        for idx, line in enumerate(cube_lines):
            self.cube.add_line(
                line, cube_lines[(idx + 1) % len(cube_lines)], np.array([0, 0, 0]))
        self.cube.set_drawing_type(GL_LINES)
        self.cube.push_to_GPU()
        self.point_cloud.init(res)
        self.point_cloud.set_drawing_type(GL_POINTS)
        # Register GLUT callback functions
        glutDisplayFunc(self.draw_callback)
        glutIdleFunc(self.idle)
        glutKeyboardFunc(self.keyPressedCallback)
        glutCloseFunc(self.close_func)
        glutMouseFunc(self.on_mouse)
        glutMotionFunc(self.on_mousemove)
        glutReshapeFunc(self.on_resize)

        quad = np.array([
            -1, -1,   0, 0,
            1, -1,   1, 0,
            1,  1,   1, 1,
            -1, -1,   0, 0,
            1,  1,   1, 1,
            -1,  1,   0, 1
        ], dtype=np.float32)

        self.quadVAO = glGenVertexArrays(1)
        self.quadVBO = glGenBuffers(1)

        glBindVertexArray(self.quadVAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.quadVBO)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        # positions
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                              4*4, ctypes.c_void_p(0))
        # uvs
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                              4*4, ctypes.c_void_p(8))
        glBindVertexArray(0)

        self.shader_bg = Shader(SCREEN_VERT, SCREEN_FRAG)
        self.tex_loc = glGetUniformLocation(
            self.shader_bg.get_program_id(), "u_tex")

        #  --------- allocate empty texture once ----------
        self.rgb_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.rgb_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                     res.width, res.height,            # same size as ZED image
                     0, GL_BGRA, GL_UNSIGNED_BYTE, None)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glBindTexture(GL_TEXTURE_2D, 0)
        self.available = True

    def is_available(self):
        if self.available:
            glutMainLoopEvent()
        return self.available

    def updateData(self, pc, mat):
        self.mutex.acquire()
        glBindTexture(GL_TEXTURE_2D, self.rgb_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0,
                        0, 0, mat.get_width(), mat.get_height(),
                        GL_BGRA, GL_UNSIGNED_BYTE,
                        ctypes.c_void_p(mat.get_pointer()))   # zero-copy upload
        glBindTexture(GL_TEXTURE_2D, 0)
        self.point_cloud.setPoints(pc)
        self.mutex.release()

    def idle(self):
        if self.available:
            glutPostRedisplay()

    def exit(self):
        if self.available:
            self.available = False

    def close_func(self):
        if self.available:
            self.available = False

    def keyPressedCallback(self, key, x, y):
        if ord(key) == 27:
            self.close_func()
        if (ord(key) == 83 or ord(key) == 115):
            self.save_data = True

    def on_mouse(self, *args, **kwargs):
        (key, Up, x, y) = args
        if key == 0:
            self.mouse_button[0] = (Up == 0)
        elif key == 2:
            self.mouse_button[1] = (Up == 0)
        elif (key == 3):
            self.wheelPosition = self.wheelPosition + 1
        elif (key == 4):
            self.wheelPosition = self.wheelPosition - 1

        self.mouseCurrentPosition = [x, y]
        self.previousMouseMotion = [x, y]

    def on_mousemove(self, *args, **kwargs):
        (x, y) = args
        self.mouseMotion[0] = x - self.previousMouseMotion[0]
        self.mouseMotion[1] = y - self.previousMouseMotion[1]
        self.previousMouseMotion = [x, y]
        glutPostRedisplay()

    def on_resize(self, Width, Height):
        glViewport(0, 0, Width, Height)
        self.camera.setProjection(Height / Width)

    def draw_callback(self):
        if self.available:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(
                self.bckgrnd_clr[0], self.bckgrnd_clr[1], self.bckgrnd_clr[2], 1.)

            glDisable(GL_DEPTH_TEST)

            glUseProgram(self.shader_bg.get_program_id())
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.rgb_tex)
            glUniform1i(self.tex_loc, 0)          # texture unit 0

            glBindVertexArray(self.quadVAO)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glBindVertexArray(0)

            glBindTexture(GL_TEXTURE_2D, 0)
            glUseProgram(0)

            glEnable(GL_DEPTH_TEST)               # restore for 3-D stuff

            self.mutex.acquire()
            self.update()
            self.draw()
            self.mutex.release()

            glutSwapBuffers()
            glutPostRedisplay()

    def update(self):
        if (self.mouse_button[0]):
            r = sl.Rotation()
            vert = self.camera.vertical_
            tmp = vert.get()
            vert.init_vector(tmp[0] * 1., tmp[1] * 1., tmp[2] * 1.)
            r.init_angle_translation(self.mouseMotion[0] * 0.02, vert)
            self.camera.rotate(r)

            r.init_angle_translation(
                self.mouseMotion[1] * 0.02, self.camera.right_)
            self.camera.rotate(r)

        if (self.mouse_button[1]):
            t = sl.Translation()
            tmp = self.camera.right_.get()
            scale = self.mouseMotion[0] * -0.05
            t.init_vector(tmp[0] * scale, tmp[1] * scale, tmp[2] * scale)
            self.camera.translate(t)

            tmp = self.camera.up_.get()
            scale = self.mouseMotion[1] * 0.05
            t.init_vector(tmp[0] * scale, tmp[1] * scale, tmp[2] * scale)
            self.camera.translate(t)

        if (self.wheelPosition != 0):
            t = sl.Translation()
            tmp = self.camera.forward_.get()
            scale = self.wheelPosition * -0.065
            t.init_vector(tmp[0] * scale, tmp[1] * scale, tmp[2] * scale)
            self.camera.translate(t)

        self.camera.update()

        self.mouseMotion = [0., 0.]
        self.wheelPosition = 0

    def draw(self):
        vpMatrix = self.camera.getViewProjectionMatrix()
        glUseProgram(self.shader_image.get_program_id())
        glUniformMatrix4fv(self.shader_image_MVP, 1, GL_TRUE,
                           (GLfloat * len(vpMatrix))(*vpMatrix))
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.zedModel.draw()
        self.cube.draw()
        glUseProgram(0)

        glUseProgram(self.shader_pc.get_program_id())
        glUniformMatrix4fv(self.shader_pc_MVP, 1, GL_TRUE,
                           (GLfloat * len(vpMatrix))(*vpMatrix))
        glPointSize(1.)
        self.point_cloud.draw()
        glUseProgram(0)


class CameraGL:
    def __init__(self, camera_v_fov):
        self.ORIGINAL_FORWARD = sl.Translation()
        self.ORIGINAL_FORWARD.init_vector(0, 0, 1)
        self.ORIGINAL_UP = sl.Translation()
        self.ORIGINAL_UP.init_vector(0, 1, 0)
        self.ORIGINAL_RIGHT = sl.Translation()
        self.ORIGINAL_RIGHT.init_vector(1, 0, 0)
        self.znear = 0.5
        self.zfar = 100.
        self.vfov = camera_v_fov
        self.orientation_ = sl.Orientation()
        self.position_ = sl.Translation()
        self.forward_ = sl.Translation()
        self.up_ = sl.Translation()
        self.right_ = sl.Translation()
        self.vertical_ = sl.Translation()
        self.vpMatrix_ = sl.Matrix4f()
        self.offset_ = sl.Translation()
        self.offset_.init_vector(0, 0, 5)
        self.projection_ = sl.Matrix4f()
        self.projection_.set_identity()
        self.setProjection(1.78)

        self.position_.init_vector(0., 0., 0.)
        tmp = sl.Translation()
        tmp.init_vector(0, 0, -.1)
        tmp2 = sl.Translation()
        tmp2.init_vector(0, 1, 0)
        self.setDirection(tmp, tmp2)

    def update(self):
        dot_ = sl.Translation.dot_translation(self.vertical_, self.up_)
        if (dot_ < 0.):
            tmp = self.vertical_.get()
            self.vertical_.init_vector(
                tmp[0] * -1., tmp[1] * -1., tmp[2] * -1.)
        transformation = sl.Transform()

        tmp_position = self.position_.get()
        tmp = (self.offset_ * self.orientation_).get()
        new_position = sl.Translation()
        new_position.init_vector(
            tmp_position[0] + tmp[0], tmp_position[1] + tmp[1], tmp_position[2] + tmp[2])
        transformation.init_orientation_translation(
            self.orientation_, new_position)
        transformation.inverse()
        self.vpMatrix_ = self.projection_ * transformation

    def setProjection(self, im_ratio):
        fov_x = self.vfov * 3.1416 / 180.
        fov_y = self.vfov * im_ratio * 3.1416 / 180.

        self.projection_[(0, 0)] = 1. / math.tan(fov_x * .5)
        self.projection_[(1, 1)] = 1. / math.tan(fov_y * .5)
        self.projection_[(2, 2)] = -(self.zfar + self.znear) / \
            (self.zfar - self.znear)
        self.projection_[(3, 2)] = -1.
        self.projection_[(2, 3)] = -(2. * self.zfar *
                                     self.znear) / (self.zfar - self.znear)
        self.projection_[(3, 3)] = 0.

    def getViewProjectionMatrix(self):
        tmp = self.vpMatrix_.m
        vpMat = array.array('f')
        for row in tmp:
            for v in row:
                vpMat.append(v)
        return vpMat

    def setDirection(self, dir, vert):
        dir.normalize()
        tmp = dir.get()
        dir.init_vector(tmp[0] * -1., tmp[1] * -1., tmp[2] * -1.)
        self.orientation_.init_translation(self.ORIGINAL_FORWARD, dir)
        self.updateVectors()
        self.vertical_ = vert
        if (sl.Translation.dot_translation(self.vertical_, self.up_) < 0.):
            tmp = sl.Rotation()
            tmp.init_angle_translation(3.14, self.ORIGINAL_FORWARD)
            self.rotate(tmp)

    def translate(self, t):
        ref = self.position_.get()
        tmp = t.get()
        self.position_.init_vector(
            ref[0] + tmp[0], ref[1] + tmp[1], ref[2] + tmp[2])

    def setPosition(self, p):
        self.position_ = p

    def rotate(self, r):
        tmp = sl.Orientation()
        tmp.init_rotation(r)
        self.orientation_ = tmp * self.orientation_
        self.updateVectors()

    def setRotation(self, r):
        self.orientation_.init_rotation(r)
        self.updateVectors()

    def updateVectors(self):
        self.forward_ = self.ORIGINAL_FORWARD * self.orientation_
        self.up_ = self.ORIGINAL_UP * self.orientation_
        right = self.ORIGINAL_RIGHT
        tmp = right.get()
        right.init_vector(tmp[0] * -1., tmp[1] * -1., tmp[2] * -1.)
        self.right_ = right * self.orientation_
