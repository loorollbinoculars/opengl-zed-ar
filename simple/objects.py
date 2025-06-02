from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import array
import pyzed.sl as sl
import numpy as np
from shader import Shader
import math


# TODO:
# - Add a simple 3D object class that can be used to draw the XYZ axis and the origin of the world.
# - Figure out if you can just use the depth map, scale it to [0, 1], and use its values as the Z coordinate of the Screen vertex shader.!!!!


SCREEN_VERT = """
#version 330 core
layout(location = 0) in vec2 in_Pos;   // -1..+1 clip-space
layout(location = 1) in vec2 in_UV;    // 0..1 texture coords
out vec2 uv;
void main() {
    uv = in_UV;
    uv.y = 1.0 - uv.y;  // flip the y coordinate for OpenGL
    gl_Position = vec4(in_Pos, 1.0, 1.0);   // already in clip-space, at the back
}
"""

SCREEN_FRAG = """
#version 330 core
in vec2 uv;
uniform sampler2D u_tex;
uniform sampler2D u_depthTex;  // For depth rendering
out vec4 out_Color;
uniform float u_near;              // 0.10, 0.05, …
uniform float u_far;               // 10.0
uniform float u_fovY;              // radians
uniform float u_aspect;            // width / height


float eyeZ_to_depth(float z, float n, float f)
{
    return (1.0/z - 1.0/n) / (1.0/f - 1.0/n);
}

void main() {
    /* 1. view-ray direction ----------------------------------- */
    vec2 ndc = vec2(uv.x * 2.0 - 1.0,
                    (1.0 - uv.y) * 2.0 - 1.0);      // flip Y

    vec2 tanHalfFov = vec2(u_aspect, 1.0) * tan(u_fovY * 0.5);
    vec3 dir        = normalize(vec3(ndc * tanHalfFov, -1.0));

    /* 2. linear depth in metres from the texture -------------- */
    float d_radial  = texture(u_depthTex, uv).r;

    /* 3. convert to eye-space z                                */
    float z_eye     = d_radial * (-dir.z);          // positive metres

    /* 4. finally: eye-space z → depth-buffer value ------------ */
    float z_buffer  = clamp(eyeZ_to_depth(z_eye, u_near, u_far), 0.0, 1.0);

    out_Color = texture(u_tex, uv);
    gl_FragDepth = z_buffer;  // Scale depth to [0, 1] range);
}
"""

TRIANGLE_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_Vertex;
uniform mat4 u_mvpMatrix;
void main() {
    gl_Position = vec4(in_Vertex, 1);
}
"""
RED_FRAGMENT_SHADER = """
#version 330 core
out vec4 out_Color;
void main() {
    out_Color = vec4(107/255.0,194/255.0,184/255.0,1.0);  // CMR Green
}
"""

CUBE_PROJECTION_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_Vertex;
uniform mat4 u_mvpMatrix;   // Model-View-Projection matrix
void main() {
    gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);
}
"""


class FullScreenQuad:
    """A simple full-screen quad for rendering textures."""

    def __init__(self, resolution, camera_fov):
        self.is_init = False
        self.resolution = resolution
        self.camera_fov = camera_fov
        self.drawing_type = GL_TRIANGLES
        self.quad = np.array([
            -1, -1,   0, 0,
            1, -1,   1, 0,
            1,  1,   1, 1,
            -1, -1,   0, 0,
            1,  1,   1, 1,
            -1,  1,   0, 1
        ], dtype=np.float32)

    def init(self):
        self.shader = Shader(SCREEN_VERT, SCREEN_FRAG)

        self.quadVAO = glGenVertexArrays(1)
        self.quadVBO = glGenBuffers(1)
        glBindVertexArray(self.quadVAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.quadVBO)
        glBufferData(GL_ARRAY_BUFFER, self.quad.nbytes,
                     self.quad, GL_STATIC_DRAW)

        # positions
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                              4*4, ctypes.c_void_p(0))
        # uvs
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                              4*4, ctypes.c_void_p(8))
        glBindVertexArray(0)

        self.tex_loc = glGetUniformLocation(
            self.shader.get_program_id(), "u_tex")
        self.depth_tex_loc = glGetUniformLocation(
            self.shader.get_program_id(), "u_depthTex")
        self.u_near_loc = glGetUniformLocation(
            self.shader.get_program_id(), "u_near")
        self.u_far_loc = glGetUniformLocation(
            self.shader.get_program_id(), "u_far")
        self.u_fovY_loc = glGetUniformLocation(
            self.shader.get_program_id(), "u_fovY")
        self.u_aspect_loc = glGetUniformLocation(
            self.shader.get_program_id(), "u_aspect")
        glUseProgram(self.shader.get_program_id())
        glUniform1f(self.u_near_loc, 0.2)
        glUniform1f(self.u_far_loc, 10.0)
        glUniform1f(self.u_fovY_loc, self.camera_fov *
                    math.pi / 180.0)  # convert to radians
        glUniform1f(self.u_aspect_loc, 16/9)
        glUseProgram(0)
        #  --------- allocate empty texture once ----------
        self.rgb_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.rgb_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                     self.resolution.width, self.resolution.height,
                     0, GL_BGRA, GL_UNSIGNED_BYTE, None)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        self.depth_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depth_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F,
                     self.resolution.width, self.resolution.height,
                     0, GL_RED, GL_FLOAT, None)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glBindTexture(GL_TEXTURE_2D, 0)

    def draw(self):
        """Draws the full-screen quad with the bound texture."""
        # glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glUseProgram(self.shader.get_program_id())
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.rgb_tex)
        glUniform1i(self.tex_loc, 0)          # texture unit 0
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.depth_tex)
        glUniform1i(self.depth_tex_loc, 1)          # texture unit 1

        glBindVertexArray(self.quadVAO)
        glDrawArrays(self.drawing_type, 0, 6)
        glBindVertexArray(0)

        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
        glDepthMask(GL_TRUE)                  # restore depth mask

    def update(self, image: sl.Mat, depth_map: sl.Mat):
        """Updates the texture to be displayed on the quad."""
        glBindTexture(GL_TEXTURE_2D, self.rgb_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0,
                        0, 0, image.get_width(), image.get_height(),
                        GL_BGRA, GL_UNSIGNED_BYTE,
                        ctypes.c_void_p(image.get_pointer()))   # zero-copy upload
        glBindTexture(GL_TEXTURE_2D, self.depth_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0,
                        0, 0, depth_map.get_width(), depth_map.get_height(),
                        GL_RED, GL_FLOAT,
                        ctypes.c_void_p(depth_map.get_pointer()))   # zero-copy upload
        glBindTexture(GL_TEXTURE_2D, 0)


class Triangle:
    def __init__(self, position=[0, 0, 0], static=True):
        self.vertices = np.array([
            -1.0, -1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, -1.0,  0.0,
        ], dtype=np.float32)

        self.indices = np.array([0, 1, 2], dtype=np.uint32)
        self.static = static
        self.mvp = np.eye(4, dtype=np.float32)
        self.mvp[:3, 3] = position

    def init(self):
        self.shader = Shader(TRIANGLE_VERTEX_SHADER, RED_FRAGMENT_SHADER)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        usage = GL_STATIC_DRAW if self.static else GL_DYNAMIC_DRAW

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes,
                     self.vertices, usage)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes,
                     self.indices, usage)

        glBindVertexArray(0)

        self.index_count = len(self.indices)

    def draw(self, *_args, **_kwargs):
        glUseProgram(self.shader.get_program_id())
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glUseProgram(0)


class Cube:
    def __init__(self, static=True, scale=1.0, v_fov=70.0, position2D=None):
        self.vertices = np.array([
            -1.0, -1.0, -1.0,
            1.0, -1.0, -1.0,
            1.0,  1.0, -1.0,
            -1.0,  1.0, -1.0,
            -1.0, -1.0,  1.0,
            1.0, -1.0,  1.0,
            1.0,  1.0,  1.0,
            -1.0,  1.0,  1.0
        ], dtype=np.float32)

        self.indices = np.array([
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            0, 4, 5, 5, 1, 0,
            2, 6, 7, 7, 3, 2,
            3, 7, 4, 4, 0, 3,
            5, 6, 2, 2, 1, 5
        ], dtype=np.uint32)

        self.static = static
        self.model = np.eye(4, dtype=np.float32)        # identity
        # optional uniform scale
        self.model[0:3, 0:3] *= scale
        if position2D:
            self.model[0:3:2, 3] = position2D
        else:
            self.model[0:3:2, 3] = np.random.random(2)
        self.model[1, 3] = scale/2 + np.random.random()
        self.viewMatrix = np.eye(4, dtype=np.float32)  # identity
        self.proj = self._perspective(v_fov, 16/9, 0.2, 10.0)

    def init(self):
        self.shader = Shader(CUBE_PROJECTION_SHADER, RED_FRAGMENT_SHADER)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        usage = GL_STATIC_DRAW if self.static else GL_DYNAMIC_DRAW

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes,
                     self.vertices, usage)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes,
                     self.indices, usage)

        glBindVertexArray(0)
        self.index_count = len(self.indices)

        self.mvp_loc = glGetUniformLocation(
            self.shader.get_program_id(), "u_mvpMatrix")

    def draw(self):
        glUseProgram(self.shader.get_program_id())
        glUniformMatrix4fv(self.mvp_loc, 1, GL_TRUE,
                           self.proj @ self.viewMatrix @ self.model)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)

        glBindVertexArray(0)
        glUseProgram(0)

    @staticmethod
    def _perspective(fov_y, aspect, z_near, z_far):
        """Return a 4x4 perspective projection matrix as np.float32."""
        f = 1.0 / math.tan(math.radians(fov_y) * 0.5)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (z_far + z_near) / (z_near - z_far)
        proj[2, 3] = (2 * z_far * z_near) / (z_near - z_far)
        proj[3, 2] = -1.0
        return proj


class Simple3DObject:
    def __init__(self, _is_static, drawing_type=GL_TRIANGLES, pts_size=3, clr_size=3):
        self.is_init = False
        self.drawing_type = drawing_type
        self.is_static = _is_static
        self.clear()
        self.pt_type = pts_size
        self.clr_type = clr_size
        self.data = sl.Mat()

    def clear(self):
        """Initialise internal arrays for vertices, colors, and indices."""
        self.vertices = array.array('f')
        self.colors = array.array('f')
        self.indices = array.array('I')
        self.elementbufferSize = 0

    def add_pt(self, _pts):  # _pts [x,y,z]
        for pt in _pts:
            self.vertices.append(pt)

    def add_clr(self, _clrs):    # _clr [r,g,b]
        for clr in _clrs:
            self.colors.append(clr)

    def add_point_clr(self, _pt, _clr):
        self.add_pt(_pt)
        self.add_clr(_clr)
        self.indices.append(len(self.indices))

    def add_line(self, _p1, _p2, _clr):
        self.add_point_clr(_p1, _clr)
        self.add_point_clr(_p2, _clr)

    def addFace(self, p1, p2, p3, clr):
        self.add_point_clr(p1, clr)
        self.add_point_clr(p2, clr)
        self.add_point_clr(p3, clr)

    def push_to_GPU(self):
        if (self.is_init == False):
            self.vboID = glGenBuffers(3)
            self.is_init = True

        if (self.is_static):
            type_draw = GL_STATIC_DRAW
        else:
            type_draw = GL_DYNAMIC_DRAW

        if len(self.vertices):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glBufferData(GL_ARRAY_BUFFER, len(self.vertices) * self.vertices.itemsize,
                         (GLfloat * len(self.vertices))(*self.vertices), type_draw)

        if len(self.colors):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ARRAY_BUFFER, len(self.colors) * self.colors.itemsize,
                         (GLfloat * len(self.colors))(*self.colors), type_draw)

        if len(self.indices):
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(
                self.indices) * self.indices.itemsize, (GLuint * len(self.indices))(*self.indices), type_draw)

        self.elementbufferSize = len(self.indices)

    def init(self, res):
        if (self.is_init == False):
            self.vboID = glGenBuffers(3)
            self.is_init = True

        if (self.is_static):
            type_draw = GL_STATIC_DRAW
        else:
            type_draw = GL_DYNAMIC_DRAW

        self.elementbufferSize = res.width * res.height

        glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
        glBufferData(GL_ARRAY_BUFFER, self.elementbufferSize *
                     self.pt_type * self.vertices.itemsize, None, type_draw)

        if (self.clr_type):
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ARRAY_BUFFER, self.elementbufferSize *
                         self.clr_type * self.colors.itemsize, None, type_draw)

        for i in range(0, self.elementbufferSize):
            self.indices.append(i+1)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.indices) * self.indices.itemsize,
                     (GLuint * len(self.indices))(*self.indices), type_draw)

    def setPoints(self, depth_map):
        glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.elementbufferSize * self.pt_type *
                        self.vertices.itemsize, ctypes.c_void_p(depth_map.get_pointer()))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self):
        if (self.elementbufferSize):
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glVertexAttribPointer(0, self.pt_type, GL_FLOAT, GL_FALSE, 0, None)

            if (self.clr_type):
                glEnableVertexAttribArray(1)
                glBindBuffer(GL_ARRAY_BUFFER, self.vboID[1])
                glVertexAttribPointer(
                    1, self.clr_type, GL_FLOAT, GL_FALSE, 0, None)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            glDrawElements(self.drawing_type,
                           self.elementbufferSize, GL_UNSIGNED_INT, None)

            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)
