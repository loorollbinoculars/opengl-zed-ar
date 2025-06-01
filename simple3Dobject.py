from OpenGL.GL import *
import array
import pyzed.sl as sl


class Simple3DObject:
    def __init__(self, _is_static, pts_size=3, clr_size=3):
        self.is_init = False
        self.drawing_type = GL_TRIANGLES
        self.is_static = _is_static
        self.clear()
        self.pt_type = pts_size
        self.clr_type = clr_size
        self.data = sl.Mat()

    def clear(self):
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

    def set_drawing_type(self, _type):
        self.drawing_type = _type

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
