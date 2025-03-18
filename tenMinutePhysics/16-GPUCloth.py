# The MIT License (MIT)
# Copyright (c) 2022 NVIDIA
# www.youtube.com/c/TenMinutePhysics
# www.matthiasMueller.info/tenMinutePhysics

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# control:

# 'p': toggle paused
# 'h': toggle hidden
# 'c': solve type coloring hybrid
# 'j': solve type Jacobi
# 'r': reset state
# Alt + Left mouse pan
# Ctrl + Middel mouse orbit
# left mouse interact
# scrool mouse zoom
# 'w' 's' 'a' 'd' 'e' 'q' : camera control // may need changes

from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import numpy as np
import warp as wp
import math
import time

wp.init()

# ---------------------------------------------

targetFps = 60
numSubsteps = 30
timeStep = 1.0 / 60.0
gravity = wp.vec3(0.0, -10.0, 0.0)
paused = False
hidden = False
frameNr = 0

# 0 Coloring
# 1 Jacobi
solveType = 0
jacobiScale = 0.2

clothNumX = 500
clothNumY = 500
clothY = 2.2
clothSpacing = 0.01
sphereCenter = wp.vec3(0.0, 1.5, 0.0)
sphereRadius = 0.5

# ---------------------------------------------

class Cloth:

    @wp.kernel
    def computeRestLengths(
            pos: wp.array(dtype = wp.vec3),
            constIds: wp.array(dtype = wp.int32),
            restLengths: wp.array(dtype = float)):
        cNr = wp.tid()
        p0 = pos[constIds[2 * cNr]]
        p1 = pos[constIds[2 * cNr + 1]]
        restLengths[cNr] = wp.length(p1 - p0)

    # -----------------------------------------------------
    def __init__(self, yOffset, numX, numY, spacing, sphereCenter, sphereRadius):
        device = "cpu"

        self.dragParticleNr = -1
        self.dragDepth = 0.0
        self.dragInvMass = 0.0
        self.renderParticles = []
        
        self.sphereCenter = sphereCenter
        self.sphereRadius = sphereRadius

        if numX % 2 == 1:
            numX = numX + 1
        if numY % 2 == 1:
            numY = numY + 1

        self.spacing = spacing
        self.numParticles = (numX + 1) * (numY + 1)
        pos = np.zeros(3 * self.numParticles)
        normals = np.zeros(3 * self.numParticles)
        invMass = np.zeros(self.numParticles)

        for xi in range(numX + 1):
            for yi in range(numY + 1):
                id = xi * (numY + 1) + yi
                pos[3 * id] = (-numX * 0.5 + xi) * spacing
                pos[3 * id + 1] = yOffset
                pos[3 * id + 2] = (-numY * 0.5 + yi) * spacing
                invMass[id] = 1.0
                # if yi == numY and (xi == 0 or xi == numX):
                #     invMass[id] = 0.0
                #     self.renderParticles.append(id)

        self.pos = wp.array(pos, dtype = wp.vec3, device = "cuda")
        self.prevPos = wp.array(pos, dtype = wp.vec3, device = "cuda")
        self.restPos = wp.array(pos, dtype = wp.vec3, device = "cuda")
        self.invMass = wp.array(invMass, dtype = float, device = "cuda")
        self.corr = wp.array(np.zeros(3 * self.numParticles), dtype = wp.vec3, device = "cuda")
        self.vel = wp.array(np.zeros(3 * self.numParticles), dtype = wp.vec3, device = "cuda")
        self.normals = wp.array(normals, dtype = wp.vec3, device = "cuda")

        self.hostInvMass = wp.array(invMass, dtype = float, device = "cpu")
        self.hostPos = wp.array(pos, dtype = wp.vec3, device = "cpu")
        self.hostNormals = wp.array(normals, dtype = wp.vec3, device = "cpu")

        # constraints

        self.passSizes = [
            (numX + 1) * math.floor(numY / 2),
            (numX + 1) * math.floor(numY / 2),
            math.floor(numX / 2) * (numY + 1),
            math.floor(numX / 2) * (numY + 1),
            2 * numX * numY + (numX + 1) * (numY - 1) + (numY + 1) * (numX - 1)
            ]
        self.passIndependent = [
            True, True, True, True, False
        ]

        self.numDistConstraints = 0
        for passSize in self.passSizes:            
            self.numDistConstraints = self.numDistConstraints + passSize

        distConstIds = np.zeros(2 * self.numDistConstraints, dtype = wp.int32)

        # stretch constraints

        i = 0
        for passNr in range(2):
            for xi in range(numX + 1):
                for yi in range(math.floor(numY / 2)):
                    distConstIds[2 * i] = xi * (numY + 1) + 2 * yi + passNr
                    distConstIds[2 * i + 1] = xi * (numY + 1) + 2 * yi + passNr + 1
                    i = i + 1

        for passNr in range(2):
            for xi in range(math.floor(numX / 2)):
                for yi in range(numY + 1):
                    distConstIds[2 * i] = (2 * xi + passNr) * (numY + 1) + yi
                    distConstIds[2 * i + 1] = (2 * xi + passNr + 1) * (numY + 1) + yi
                    i = i + 1

        # shear constraints

        for xi in range(numX):
            for yi in range(numY):
                distConstIds[2 * i] = xi * (numY + 1) + yi
                distConstIds[2 * i + 1] = (xi + 1) * (numY + 1) + yi + 1
                i = i + 1
                distConstIds[2 * i] = (xi + 1) * (numY + 1) + yi
                distConstIds[2 * i + 1] = xi * (numY + 1) + yi + 1
                i = i + 1

        # bending constraints

        for xi in range(numX + 1):
            for yi in range(numY - 1):
                distConstIds[2 * i] = xi * (numY + 1) + yi
                distConstIds[2 * i + 1] = xi * (numY + 1) + yi + 2
                i = i + 1

        for xi in range(numX - 1):
            for yi in range(numY + 1):
                distConstIds[2 * i] = xi * (numY + 1) + yi
                distConstIds[2 * i + 1] = (xi + 2) * (numY + 1) + yi                
                i = i + 1

        self.distConstIds = wp.array(distConstIds, dtype = wp.int32, device = "cuda")
        self.constRestLengths = wp.zeros(self.numDistConstraints, dtype = float, device = "cuda")

        wp.launch(kernel = self.computeRestLengths,
                inputs = [self.pos, self.distConstIds, self.constRestLengths], 
                dim = self.numDistConstraints,  device = "cuda")

        # tri ids

        self.numTris = 2 * numX * numY
        self.hostTriIds = np.zeros(3 * self.numTris, dtype = np.int32)

        i = 0
        for xi in range(numX):
            for yi in range(numY):
                id0 = xi * (numY + 1) + yi
                id1 = (xi + 1) * (numY + 1) + yi
                id2 = (xi + 1) * (numY + 1) + yi + 1
                id3 = xi * (numY + 1) + yi + 1

                self.hostTriIds[i] = id0
                self.hostTriIds[i + 1] = id1
                self.hostTriIds[i + 2] = id2

                self.hostTriIds[i + 3] = id0
                self.hostTriIds[i + 4] = id2
                self.hostTriIds[i + 5] = id3

                i = i + 6

        self.triIds = wp.array(self.hostTriIds, dtype = wp.int32, device = "cuda")

        self.triDist = wp.zeros(self.numTris, dtype = float, device = "cuda")
        self.hostTriDist = wp.zeros(self.numTris, dtype = float, device = "cpu")

        print(str(self.numTris) + " triangles created")
        print(str(self.numDistConstraints) + " distance constraints created")
        print(str(self.numParticles) + " particles created")


    # ----------------------------------
    @wp.kernel
    def addNormals(
            pos: wp.array(dtype = wp.vec3),
            triIds: wp.array(dtype = wp.int32),
            normals: wp.array(dtype = wp.vec3)):
        triNr = wp.tid()

        id0 = triIds[3 * triNr]
        id1 = triIds[3 * triNr + 1]
        id2 = triIds[3 * triNr + 2]
        normal = wp.cross(pos[id1] - pos[id0], pos[id2] - pos[id0])
        wp.atomic_add(normals, id0, normal)
        wp.atomic_add(normals, id1, normal)
        wp.atomic_add(normals, id2, normal)

    @wp.kernel
    def normalizeNormals(
            normals: wp.array(dtype = wp.vec3)):

        pNr = wp.tid()
        normals[pNr] = wp.normalize(normals[pNr])

    def updateMesh(self):
        self.normals.zero_()
        wp.launch(kernel = self.addNormals, inputs = [self.pos, self.triIds, self.normals], dim = self.numTris, device = "cuda")
        wp.launch(kernel = self.normalizeNormals, inputs = [self.normals], dim = self.numParticles, device = "cuda")
        wp.copy(self.hostNormals, self.normals)

    # ----------------------------------

    @wp.kernel
    def integrate(
            dt: float,
            gravity: wp.vec3,
            invMass: wp.array(dtype = float),
            prevPos: wp.array(dtype = wp.vec3),
            pos: wp.array(dtype = wp.vec3),
            vel: wp.array(dtype = wp.vec3),
            sphereCenter: wp.vec3,
            sphereRadius: float):

        pNr = wp.tid()

        prevPos[pNr] = pos[pNr]
        if invMass[pNr] == 0.0:
            return
        vel[pNr] = vel[pNr] + gravity * dt
        pos[pNr] = pos[pNr] + vel[pNr] * dt
        
        # collisions
        
        thickness = 0.001
        friction = 0.01

        d = wp.length(pos[pNr] - sphereCenter)
        if d < (sphereRadius + thickness):
            p = pos[pNr] * (1.0 - friction) + prevPos[pNr] * friction
            r = p - sphereCenter
            d = wp.length(r)            
            pos[pNr] = sphereCenter + r * ((sphereRadius + thickness) / d)
            
        p = pos[pNr]
        if p[1] < thickness:
            p = pos[pNr] * (1.0 - friction) + prevPos[pNr] * friction
            pos[pNr] = wp.vec3(p[0], thickness, p[2])

    # ----------------------------------
    @wp.kernel
    def solveDistanceConstraints(
            solveType: wp.int32,
            firstConstraint: wp.int32,
            invMass: wp.array(dtype = float),
            pos: wp.array(dtype = wp.vec3),
            corr: wp.array(dtype = wp.vec3),
            constIds: wp.array(dtype = wp.int32),
            restLengths: wp.array(dtype = float)):

        cNr = firstConstraint + wp.tid()
        id0 = constIds[2 * cNr]
        id1 = constIds[2 * cNr + 1]
        w0 = invMass[id0]
        w1 = invMass[id1]
        w = w0 + w1
        if w == 0.0:
            return
        p0 = pos[id0]
        p1 = pos[id1]
        d = p1 - p0
        n = wp.normalize(d)
        l = wp.length(d)
        l0 = restLengths[cNr]
        dP = n * (l - l0) / w
        if solveType == 1:
            wp.atomic_add(corr, id0, w0 * dP)
            wp.atomic_sub(corr, id1, w1 * dP)
        else:
            wp.atomic_add(pos, id0, w0 * dP)
            wp.atomic_sub(pos, id1, w1 * dP)

    # ----------------------------------
    @wp.kernel
    def addCorrections(
            pos: wp.array(dtype = wp.vec3),
            corr: wp.array(dtype = wp.vec3),
            scale: float):
        pNr = wp.tid()
        pos[pNr] = pos[pNr] + corr[pNr] * scale

    # ----------------------------------
    @wp.kernel
    def updateVel(
            dt: float,
            prevPos: wp.array(dtype = wp.vec3),
            pos: wp.array(dtype = wp.vec3),
            vel: wp.array(dtype = wp.vec3)):
        pNr = wp.tid()
        vel[pNr] = (pos[pNr] - prevPos[pNr]) / dt

    # ----------------------------------
    def simulate(self):

        # ----------------------------------
        dt = timeStep / numSubsteps
        numPasses = len(self.passSizes)

        for step in range(numSubsteps):  
            wp.launch(kernel = self.integrate, 
                inputs = [dt, gravity, self.invMass, self.prevPos, self.pos, self.vel, self.sphereCenter, self.sphereRadius], 
                dim = self.numParticles, device = "cuda")

            if solveType == 0:
                firstConstraint = 0
                for passNr in range(numPasses):
                    numConstraints = self.passSizes[passNr]

                    if self.passIndependent[passNr]:
                        wp.launch(kernel = self.solveDistanceConstraints,
                            inputs = [0, firstConstraint, self.invMass, self.pos, self.corr, self.distConstIds, self.constRestLengths], 
                            dim = numConstraints,  device = "cuda")
                    else:
                        self.corr.zero_()
                        wp.launch(kernel = self.solveDistanceConstraints,
                            inputs = [1, firstConstraint, self.invMass, self.pos, self.corr, self.distConstIds, self.constRestLengths], 
                            dim = numConstraints,  device = "cuda")
                        wp.launch(kernel = self.addCorrections,
                            inputs = [self.pos, self.corr, jacobiScale], 
                            dim = self.numParticles,  device = "cuda")
                    
                    firstConstraint = firstConstraint + numConstraints

            elif solveType == 1:
                self.corr.zero_()
                wp.launch(kernel = self.solveDistanceConstraints, 
                    inputs = [1, 0, self.invMass, self.pos, self.corr, self.distConstIds, self.constRestLengths], 
                    dim = self.numDistConstraints,  device = "cuda")
                wp.launch(kernel = self.addCorrections,
                    inputs = [self.pos, self.corr, jacobiScale], 
                    dim = self.numParticles,  device = "cuda")

            wp.launch(kernel = self.updateVel, 
                inputs = [dt, self.prevPos, self.pos, self.vel], dim = self.numParticles, device = "cuda")
            
        wp.copy(self.hostPos, self.pos)

    # -------------------------------------------------
    def reset(self):
        self.vel.zero_()
        wp.copy(self.pos, self.restPos)

    # -------------------------------------------------
    @wp.kernel
    def raycastTriangle(
            orig: wp.vec3,
            dir: wp.vec3,
            pos: wp.array(dtype = wp.vec3),
            triIds: wp.array(dtype = wp.int32),
            dist: wp.array(dtype = float)):
        triNr = wp.tid()
        noHit = 1.0e6

        id0 = triIds[3 * triNr]
        id1 = triIds[3 * triNr + 1]
        id2 = triIds[3 * triNr + 2]              
        pNr = wp.tid()

        edge1 = pos[id1] - pos[id0]
        edge2 = pos[id2] - pos[id0]
        pvec = wp.cross(dir, edge2)
        det = wp.dot(edge1, pvec)

        if (det == 0.0):
            dist[triNr] = noHit
            return

        inv_det = 1.0 / det
        tvec = orig - pos[id0]
        u = wp.dot(tvec, pvec) * inv_det
        if u < 0.0 or u > 1.0:
            dist[triNr] = noHit
            return 

        qvec = wp.cross(tvec, edge1)
        v = wp.dot(dir, qvec) * inv_det
        if v < 0.0 or u + v > 1.0:
            dist[triNr] = noHit
            return

        dist[triNr] = wp.dot(edge2, qvec) * inv_det

    # ------------------------------------------------
    def startDrag(self, orig, dir):
        
        wp.launch(kernel = self.raycastTriangle, inputs = [
            wp.vec3(orig[0], orig[1], orig[2]), wp.vec3(dir[0], dir[1], dir[2]), 
            self.pos, self.triIds, self.triDist], dim = self.numTris, device = "cuda")
        wp.copy(self.hostTriDist, self.triDist)

        pos = self.hostPos.numpy()
        self.dragDepth = 0.0

        dists = self.hostTriDist.numpy()
        minTriNr = np.argmin(dists)
        if dists[minTriNr] < 1.0e6:
            self.dragParticleNr = self.hostTriIds[3 * minTriNr]
            self.dragDepth = dists[minTriNr]
            invMass = self.hostInvMass.numpy()
            self.dragInvMass = invMass[self.dragParticleNr]
            invMass[self.dragParticleNr] = 0.0
            wp.copy(self.invMass, self.hostInvMass)

            pos = self.hostPos.numpy()
            dragPos = wp.vec3(
                orig[0] + self.dragDepth * dir[0], 
                orig[1] + self.dragDepth * dir[1], 
                orig[2] + self.dragDepth * dir[2])
            pos[self.dragParticleNr] = dragPos

            wp.copy(self.pos, self.hostPos)
        
    def drag(self, orig, dir):
        if self.dragParticleNr >= 0:
            pos = self.hostPos.numpy()
            dragPos = wp.vec3(
                orig[0] + self.dragDepth * dir[0], 
                orig[1] + self.dragDepth * dir[1], 
                orig[2] + self.dragDepth * dir[2])
            pos[self.dragParticleNr] = dragPos
            wp.copy(self.pos, self.hostPos)

    def endDrag(self):
        if self.dragParticleNr >= 0:
            invMass = self.hostInvMass.numpy()
            invMass[self.dragParticleNr] = self.dragInvMass
            wp.copy(self.invMass, self.hostInvMass)
            self.dragParticleNr = -1

    def render(self):

        # cloth

        twoColors = False

        glColor3f(1.0, 0.0, 0.0)
        glNormal3f(0.0, 0.0, -1.0)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, self.hostPos.numpy())
        glNormalPointer(GL_FLOAT, 0, self.hostNormals.numpy())

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
  
        if twoColors:
            glCullFace(GL_FRONT)
            glColor3f(1.0, 1.0, 0.0)
            glDrawElementsui(GL_TRIANGLES, self.hostTriIds)
            glCullFace(GL_BACK)
            glColor3f(1.0, 0.0, 0.0)
            glDrawElementsui(GL_TRIANGLES, self.hostTriIds)
        else:
            glDisable(GL_CULL_FACE)
            glColor3f(1.0, 0.0, 0.0)
            glDrawElementsui(GL_TRIANGLES, self.hostTriIds)
            glEnable(GL_CULL_FACE)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

        # kinematic particles

        glColor3f(1.0, 1.0, 1.0)
        pos = self.hostPos.numpy()

        q = gluNewQuadric()

        if self.dragParticleNr >= 0:
            if self.dragParticleNr not in self.renderParticles:
                self.renderParticles.append(self.dragParticleNr)

        for id in self.renderParticles:
            glPushMatrix()
            p = pos[id]
            glTranslatef(p[0], p[1], p[2])
            gluSphere(q, 0.02, 40, 40)
            glPopMatrix()

        if self.dragParticleNr >= 0:
            if self.dragParticleNr in self.renderParticles:
                self.renderParticles.remove(self.dragParticleNr)
            
        # sphere
        glColor3f(0.8, 0.8, 0.8)

        glPushMatrix()
        glTranslatef(self.sphereCenter[0], self.sphereCenter[1], self.sphereCenter[2])
        gluSphere(q, self.sphereRadius, 40, 40)
        glPopMatrix()

        gluDeleteQuadric(q)

# --------------------------------------------------------------------
# Demo Viewer using GLFW
# --------------------------------------------------------------------

groundVerts = []
groundIds = []
groundColors = []
cloth = None

# -------------------------------------------------------
def initScene():
    global cloth
  
    cloth = Cloth(clothY, clothNumX, clothNumY, clothSpacing, sphereCenter, sphereRadius)

# --------------------------------
def showScreen():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # ground plane

    glColor3f(1.0, 1.0, 1.0)
    glNormal3f(0.0, 1.0, 0.0)

    numVerts = math.floor(len(groundVerts) / 3)

    glVertexPointer(3, GL_FLOAT, 0, groundVerts)
    glColorPointer(3, GL_FLOAT, 0, groundColors)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glDrawArrays(GL_QUADS, 0, numVerts)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)

    # objects

    if not hidden:
        cloth.render()

# -----------------------------------
class Camera:
    def __init__(self):
        self.pos = wp.vec3(0.0, 1.0, 5.0)
        self.forward = wp.vec3(0.0, 0.0, -1.0)
        self.up = wp.vec3(0.0, 1.0, 0.0)
        self.right = wp.cross(self.forward, self.up)
        self.speed = 0.1
        self.keyDown = [False] * 256

    def rot(self, unitAxis, angle, v):
       q = wp.quat_from_axis_angle(unitAxis, angle)
       return wp.quat_rotate(q, v)

    def setView(self):
        width, height = glfw.get_framebuffer_size(window)
        aspect_ratio = width / height if height > 0 else 1.0

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40.0, aspect_ratio, 0.01, 1000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        gluLookAt( 
            self.pos[0], self.pos[1], self.pos[2], 
            self.pos[0] + self.forward[0], self.pos[1] + self.forward[1], self.pos[2] + self.forward[2], 
            self.up[0], self.up[1], self.up[2])

    def lookAt(self, pos, at):
        self.pos = pos
        self.forward = wp.sub(at, pos)
        self.forward = wp.normalize(self.forward)
        self.up = wp.vec3(0.0, 1.0, 0.0)
        self.right = wp.cross(self.forward, self.up)
        self.right = wp.normalize(self.right)
        self.up = wp.cross(self.right, self.forward)

    def handleMouseTranslate(self, dx, dy):
        
        scale = wp.length(self.pos) * 0.001
        self.pos = wp.sub(self.pos, wp.mul(self.right, scale * float(dx)))
        self.pos = wp.add(self.pos, wp.mul(self.up, scale * float(dy)))

    def handleWheel(self, direction):
        self.pos = wp.add(self.pos, wp.mul(self.forward, direction * self.speed))

    def handleMouseView(self, dx, dy):
        scale = 0.005
        self.forward = self.rot(self.up, -dx * scale, self.forward)
        self.forward = self.rot(self.right, -dy * scale, self.forward)
        self.forward = wp.normalize(self.forward)
        self.right = wp.cross(self.forward, self.up)
        self.right = wp.vec3(self.right[0], 0.0, self.right[2])
        self.right = wp.normalize(self.right)
        self.up = wp.cross(self.right, self.forward)
        self.up = wp.normalize(self.up)
        self.forward = wp.cross(self.up, self.right)
    
    def handleKeyDown(self, key):
        if 0 <= key < 256:
            self.keyDown[key] = True

    def handleKeyUp(self, key):
        if 0 <= key < 256:
            self.keyDown[key] = False

    def handleKeys(self):
        if self.keyDown[ord('+')]:
            self.speed = self.speed * 1.2
        if self.keyDown[ord('-')]:
            self.speed = self.speed * 0.8
        if self.keyDown[ord('w')]:
            self.pos = wp.add(self.pos, wp.mul(self.forward, self.speed))
        if self.keyDown[ord('s')]:
            self.pos = wp.sub(self.pos, wp.mul(self.forward, self.speed))
        if self.keyDown[ord('a')]:
            self.pos = wp.sub(self.pos, wp.mul(self.right, self.speed))
        if self.keyDown[ord('d')]:
            self.pos = wp.add(self.pos, wp.mul(self.right, self.speed))
        if self.keyDown[ord('e')]:
            self.pos = wp.sub(self.pos, wp.mul(self.up, self.speed))
        if self.keyDown[ord('q')]:
            self.pos = wp.add(self.pos, wp.mul(self.up, self.speed))

    def handleMouseOrbit(self, dx, dy, center):

        offset = wp.sub(self.pos, center)
        offset = [
            wp.dot(self.right, offset),
            wp.dot(self.forward, offset),
            wp.dot(self.up, offset)]
    
        scale = 0.01
        self.forward = self.rot(self.up, -dx * scale, self.forward)
        self.forward = self.rot(self.right, -dy * scale, self.forward)
        self.up = self.rot(self.up, -dx * scale, self.up)
        self.up = self.rot(self.right, -dy * scale, self.up)

        self.right = wp.cross(self.forward, self.up)
        self.right = wp.vec3(self.right[0], 0.0, self.right[2])
        self.right = wp.normalize(self.right)
        self.up = wp.cross(self.right, self.forward)
        self.up = wp.normalize(self.up)
        self.forward = wp.cross(self.up, self.right)
        self.pos = wp.add(center, wp.mul(self.right, offset[0]))
        self.pos = wp.add(self.pos, wp.mul(self.forward, offset[1]))
        self.pos = wp.add(self.pos, wp.mul(self.up, offset[2]))

camera = Camera()

# ---- Callbacks and Input Handling ----------------------------------------------------

interactionMode = "camera"  # Possible values: "camera", "cloth"
mouseButton = None
mouseX = 0
mouseY = 0
modifier = None  # Possible values: "ctrl", "alt", None

def getMouseRay(x, y):
    width, height = glfw.get_framebuffer_size(window)
    viewport = glGetIntegerv(GL_VIEWPORT)
    modelMatrix = glGetDoublev(GL_MODELVIEW_MATRIX)
    projMatrix = glGetDoublev(GL_PROJECTION_MATRIX)

    y_gl = viewport[3] - y - 1
    p0 = gluUnProject(x, y_gl, 0.0, modelMatrix, projMatrix, viewport)
    p1 = gluUnProject(x, y_gl, 1.0, modelMatrix, projMatrix, viewport)
    orig = wp.vec3(p0[0], p0[1], p0[2])
    dir = wp.sub(wp.vec3(p1[0], p1[1], p1[2]), orig)
    dir = wp.normalize(dir)
    return [orig, dir]

def mouse_button_callback(window, button, action, mods):
    global mouseX, mouseY, mouseButton, interactionMode, modifier, paused
    if action == glfw.PRESS:
        mouseButton = button
        # Determine interaction mode based on modifier keys
        if mods & glfw.MOD_CONTROL:
            interactionMode = "camera_orbit"
            modifier = "ctrl"
        elif mods & glfw.MOD_ALT:
            interactionMode = "camera_translate"
            modifier = "alt"
        else:
            interactionMode = "cloth"
            modifier = None

        xpos, ypos = glfw.get_cursor_pos(window)
        mouseX = xpos
        mouseY = ypos

        if interactionMode == "cloth":
            ray = getMouseRay(xpos, ypos)
            cloth.startDrag(ray[0], ray[1])
            paused = False
    elif action == glfw.RELEASE:
        if interactionMode == "cloth":
            cloth.endDrag()
        mouseButton = None
        interactionMode = "camera"
        modifier = None

def cursor_position_callback(window, xpos, ypos):
    global mouseX, mouseY, mouseButton, interactionMode, modifier

    dx = xpos - mouseX
    dy = ypos - mouseY

    if interactionMode == "cloth":
        ray = getMouseRay(xpos, ypos)
        cloth.drag(ray[0], ray[1])
    elif interactionMode == "camera_orbit":
        camera.handleMouseView(dx, dy)
    elif interactionMode == "camera_translate":
        camera.handleMouseTranslate(dx, dy)

    mouseX = xpos
    mouseY = ypos        

def scroll_callback(window, xoffset, yoffset):
    camera.handleWheel(yoffset)

def key_callback(window, key, scancode, action, mods):
    global paused, solveType, hidden
    if action == glfw.PRESS:
        camera.handleKeyDown(key)
        if key == glfw.KEY_P:
            paused = not paused
        elif key == glfw.KEY_H:
            hidden = not hidden
        elif key == glfw.KEY_C:
            solveType = 0
        elif key == glfw.KEY_J:
            solveType = 1
        elif key == glfw.KEY_R:
            cloth.reset()
    elif action == glfw.RELEASE:
        camera.handleKeyUp(key)

# -----------------------------------------------------------
def setupOpenGL():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_CULL_FACE)
    glShadeModel(GL_SMOOTH)
    glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
    glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    ambientColor = [0.2, 0.2, 0.2, 1.0]
    diffuseColor = [0.8, 0.8 ,0.8, 1.0]
    specularColor = [1.0, 1.0, 1.0, 1.0]

    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientColor)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseColor)
    glLightfv(GL_LIGHT0, GL_SPECULAR, specularColor)

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specularColor)
    glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 50.0)

    lightPosition = [10.0, 10.0 , 10.0, 0.0]
    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition)

    glEnable(GL_NORMALIZE)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)

    groundNumTiles = 30
    groundTileSize = 0.5

    global groundVerts
    global groundIds
    global groundColors

    groundVerts = np.zeros(3 * 4 * groundNumTiles * groundNumTiles, dtype = float)
    groundColors = np.zeros(3 * 4 * groundNumTiles * groundNumTiles, dtype = float)

    squareVerts = [[0,0], [0,1], [1,1], [1,0]]
    r = groundNumTiles / 2.0 * groundTileSize

    for xi in range(groundNumTiles):
        for zi in range(groundNumTiles):
            x = (-groundNumTiles / 2.0 + xi) * groundTileSize
            z = (-groundNumTiles / 2.0 + zi) * groundTileSize
            p = xi * groundNumTiles + zi
            for i in range(4):
                q = 4 * p + i
                px = x + squareVerts[i][0] * groundTileSize
                pz = z + squareVerts[i][1] * groundTileSize
                groundVerts[3 * q] = px
                groundVerts[3 * q + 1] = 0.0  # Assuming ground plane is at y=0
                groundVerts[3 * q + 2] = pz
                col = 0.4
                if (xi + zi) % 2 == 1:
                    col = 0.8
                pr = math.sqrt(px * px + pz * pz)
                d = max(0.0, 1.0 - pr / r)
                col = col * d
                for j in range(3):
                    groundColors[3 * q + j] = col

# ------------------------------

def main():
    global window

    if not glfw.init():
        print("Failed to initialize GLFW")
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    window = glfw.create_window(800, 500, "Parallel cloth simulation", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window")
        return

    glfw.make_context_current(window)

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    initScene()

    setupOpenGL()

    prevTime = time.perf_counter()

    while not glfw.window_should_close(window):
        currentTime = time.perf_counter()
        elapsed = currentTime - prevTime
        if elapsed >= 1.0 / targetFps:
            prevTime = currentTime

            camera.handleKeys()
            
            if not paused:
                cloth.simulate()

            cloth.updateMesh()

            camera.setView()

            showScreen()

            glfw.swap_buffers(window)

            glfw.poll_events()

            global frameNr
            frameNr += 1
            numFpsFrames = 30
            if frameNr % numFpsFrames == 0:
                fps = targetFps
                glfw.set_window_title(window, f"Parallel cloth simulation {fps} fps")

    glfw.terminate()

# Run the application
if __name__ == "__main__":
    main()