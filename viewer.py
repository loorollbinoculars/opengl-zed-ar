from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import ctypes
import sys
import math
from threading import Lock
import numpy as np
import array

import pyzed.sl as sl
from simple3Dobject import Simple3DObject
from shader import Shader
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

DEPTH_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_Vertex;  // vertex position
uniform mat4 u_mvpMatrix; // Model View Projection Matrix. Of which View is updated every frame and M is identity
out vec4 b_color;
void main() {
    b_color = vec4(1.0,0.0,0.0, 1.0);
    gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);
}

"""

DEPTH_FRAGMENT_SHADER = """
#version 330 core
in vec4 b_color;
layout(location = 0) out vec4 out_Color;
void main() {
   out_Color = b_color;
}
"""


class GLViewer:
    def __init__(self, camera_v_fov):
        self.available = False
        self.mutex = Lock()
        self.camera = CameraGL(camera_v_fov)
        self.point_cloud = Simple3DObject(False, 3, 0)

    def init(self, _argc, _argv, res):  # _params = sl.CameraParameters
        glutInit(_argc, _argv)
        wnd_w = int(glutGet(GLUT_SCREEN_WIDTH)*0.9)
        wnd_h = int(glutGet(GLUT_SCREEN_HEIGHT) * 0.9)
        glutInitWindowSize(wnd_w, wnd_h)
        glutInitWindowPosition(int(wnd_w*0.05), int(wnd_h*0.05))
        print('getting here')
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

        self.shader_depth_map = Shader(DEPTH_VERTEX_SHADER,
                                       DEPTH_FRAGMENT_SHADER)
        self.shader_depth_map_MVP = glGetUniformLocation(
            self.shader_depth_map.get_program_id(), "u_mvpMatrix")

        self.bckgrnd_clr = np.array([1/255., 1/255., 1/255.])

        self.point_cloud.init(res)
        self.point_cloud.set_drawing_type(GL_QUAD_STRIP)

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

        # Register GLUT callback functions
        glutDisplayFunc(self.draw_callback)
        glutIdleFunc(self.idle)
        glutCloseFunc(self.close_func)
        glutKeyboardFunc(self.keyPressedCallback)

        glutReshapeFunc(self.on_resize)
        self.available = True

    def is_available(self):
        if self.available:
            glutMainLoopEvent()
        return self.available

    def keyPressedCallback(self, key, x, y):
        if ord(key) == 27:
            self.close_func()

    def updateData(self, point_cloud, extrinsic_matrix, image):
        self.mutex.acquire()
        self.point_cloud.setPoints(point_cloud)
        glBindTexture(GL_TEXTURE_2D, self.rgb_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0,
                        0, 0, image.get_width(), image.get_height(),
                        GL_BGRA, GL_UNSIGNED_BYTE,
                        ctypes.c_void_p(image.get_pointer()))   # zero-copy upload
        glBindTexture(GL_TEXTURE_2D, 0)
        self.camera.extrinsic = extrinsic_matrix  # MIGHT NEED TO BE TRANSPOSED!!
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
        self.camera.update()

    def draw(self):
        vpMatrix = self.camera.getViewProjectionMatrix()
        glUseProgram(self.shader_image.get_program_id())
        glUniformMatrix4fv(self.shader_image_MVP, 1, GL_TRUE,
                           (GLfloat * len(vpMatrix))(*vpMatrix))
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glUseProgram(0)

        glUseProgram(self.shader_depth_map.get_program_id())
        glUniformMatrix4fv(self.shader_depth_map_MVP, 1, GL_TRUE,
                           (GLfloat * len(vpMatrix))(*vpMatrix))
        glPointSize(1.)
        glUseProgram(0)


class CameraGL:
    def __init__(self, camera_v_fov):
        self.znear = 0.5
        self.zfar = 100.
        self.vfov = camera_v_fov
        self.orientation_ = sl.Orientation()
        self.position_ = sl.Translation()
        self.extrinsic = sl.Transform()  # View matrix in practice
        self.vpMatrix_ = sl.Matrix4f()
        self.offset_ = sl.Translation()
        self.offset_.init_vector(0, 0, 5)
        self.projection_ = sl.Matrix4f()
        self.projection_.set_identity()
        self.setProjection(1.77778)

        self.position_.init_vector(0., 0., 0.)

    def update(self):
        self.vpMatrix_ = self.projection_ * self.extrinsic

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

    def setPosition(self, p):
        self.position_ = p

    def setRotation(self, r):
        self.orientation_.init_rotation(r)
