from threading import Thread
from multiprocessing import Process
import numpy as np
import pybullet as p
import time
import pybullet_data
import math
import matplotlib.pyplot as plt

point_x = []
point_y = []
numRays = 8
rayLen = 10
rayColor = [0, 0, 1]


def mapping(x, y):
    plt.scatter(x, y, color='blue', marker='o')
    plt.ion()
    plt.show()
    plt.pause(0.5)
    plt.clf()


class mir_with_ur:
    def __init__(self):
        physicsClient = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.obstacleSratPos_1 = [3.5, 2.5, 1]
        self.roomStartPos = [0, 0, 1]
        self.planeId = p.loadURDF("plane.urdf")
        self.obstacleSratPos_2 = [-3, 1, 1]
        self.obstacleSratPos_3 = [3, -4, 1]
        self.obstacleSratPos_4 = [6, 0, 1]
        self.obstacleSratPos_5 = [-3, -4, 1]
        self.obstacleSratPos_6 = [-5, -2, 1]
        self.room = p.loadURDF("ur_mir_data/room.urdf", globalScaling=5, basePosition=self.roomStartPos)
        self.obstacle_1 = p.loadURDF("ur_mir_data/obstacle.urdf", basePosition=self.obstacleSratPos_1)
        self.obstacle_2 = p.loadURDF("ur_mir_data/obstacle.urdf", basePosition=self.obstacleSratPos_2)
        self.obstacle_3 = p.loadURDF("ur_mir_data/obstacle.urdf", basePosition=self.obstacleSratPos_3)
        self.obstacle_4 = p.loadURDF("ur_mir_data/obstacle.urdf", basePosition=self.obstacleSratPos_4)
        self.obstacle_5 = p.loadURDF("ur_mir_data/obstacle.urdf", basePosition=self.obstacleSratPos_5)
        self.obstacle_6 = p.loadURDF("ur_mir_data/obstacle.urdf", basePosition=self.obstacleSratPos_6)
        self.robot = p.loadURDF("ur_mir_data/mir_ur.urdf")
        p.setRealTimeSimulation(1)
        numJoints = p.getNumJoints(self.robot)
        self.baseYaw = p.getEulerFromQuaternion(p.getLinkState(self.robot, 3)[1])[2]
        for joint in range(numJoints):
            print(p.getJointInfo(self.robot, joint))
            p.setJointMotorControl(self.robot, joint, p.POSITION_CONTROL, 0, 100)
        # p.setJointMotorControl2(self.robot, 8, p.VELOCITY_CONTROL, targetVelocity=-15)
        # p.setJointMotorControl2(self.robot, 7, p.VELOCITY_CONTROL, targetVelocity=-15)

    def Camera(self):
        basePos = p.getLinkState(self, 28)[0]
        baseOrientation = p.getLinkState(self, 28)[1]
        matrix = p.getMatrixFromQuaternion(baseOrientation)
        tx_vec = np.array([matrix[0], matrix[3], matrix[6]])
        tz_vec = np.array([matrix[2], matrix[5], matrix[8]])

        basePos = np.array(basePos)
        targetPos = basePos + 1 * tx_vec

        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=basePos,
            cameraTargetPosition=targetPos,
            cameraUpVector=tz_vec,
        )
        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=50.0,
            aspect=1.0,
            nearVal=0.01,
            farVal=20,
        )

        p.getCameraImage(
            width=214, height=214,
            viewMatrix=viewMatrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            projectionMatrix=projectionMatrix,
        )

    def Laser(self, Yaw1):
        rayFrom1 = []
        rayTo1 = []
        rayFrom2 = []
        rayTo2 = []
        rayFrom3 = []
        rayTo3 = []
        for i in range(numRays):
            results1 = []
            results2 = []
            results3 = []

            basePos = p.getLinkState(self, 3)[0]
            Yaw = p.getEulerFromQuaternion(p.getLinkState(self, 3)[1])[2] - Yaw1

            ray_x1 = basePos[0] + rayLen * math.sin((0.5 * math.pi * float(i) / numRays) - Yaw)
            ray_y1 = basePos[1] + rayLen * math.cos((0.5 * math.pi * float(i) / numRays) - Yaw)
            ray_z1 = basePos[2]
            rayFrom1.append(basePos)
            rayTo1.append([ray_x1, ray_y1, ray_z1])
            p.addUserDebugLine(rayFrom1[i], rayTo1[i], rayColor)
            results1.append(p.rayTest(rayFrom1[i], rayTo1[i]))

            basePos = p.getLinkState(self, 3)[0]
            ray_x2 = basePos[0] + rayLen * math.sin((0.5 * math.pi * float(i) / numRays) + (0.5 * math.pi) - Yaw)
            ray_y2 = basePos[1] + rayLen * math.cos((0.5 * math.pi * float(i) / numRays) + (0.5 * math.pi) - Yaw)
            ray_z2 = basePos[2]
            rayFrom2.append(basePos)
            rayTo2.append([ray_x2, ray_y2, ray_z2])
            p.addUserDebugLine(rayFrom2[i], rayTo2[i], rayColor)
            results2.append(p.rayTest(rayFrom2[i], rayTo2[i]))

            basePos = p.getLinkState(self, 3)[0]
            ray_x3 = basePos[0] + rayLen * math.sin((0.5 * math.pi * float(i) / numRays) + (1.5 * math.pi) - Yaw)
            ray_y3 = basePos[1] + rayLen * math.cos((0.5 * math.pi * float(i) / numRays) + (1.5 * math.pi) - Yaw)
            ray_z3 = basePos[2]
            rayFrom3.append(basePos)
            rayTo3.append([ray_x3, ray_y3, ray_z3])
            p.addUserDebugLine(rayFrom3[i], rayTo3[i], rayColor)
            results3.append(p.rayTest(rayFrom3[i], rayTo3[i]))

            for j in range(len(results1)):
                point_x.append(results1[j][0][3][0] - rayFrom1[j][0])
                point_y.append(results1[j][0][3][1] - rayFrom1[j][1])

            for j in range(len(results2)):
                point_x.append(results2[j][0][3][0] - rayFrom2[j][0])
                point_y.append(results2[j][0][3][1] - rayFrom2[j][0])

            for j in range(len(results3)):
                point_x.append(results3[j][0][3][0] - rayFrom3[j][0])
                point_y.append(results3[j][0][3][1] - rayFrom3[j][0])
        p.removeAllUserDebugItems()

    def run(self):
        try:
            while True:
                do_laser = Thread(target=mir_with_ur.Laser, args=(self.robot, self.baseYaw))
                do_camera = Thread(target=mir_with_ur.Camera, args=(self.robot,))
                do_laser.start()
                do_camera.start()
                # mapping(point_x, point_y)
                do_laser.join()
                do_camera.join()
        finally:
            p.disconnect()


if __name__ == '__main__':
    agv = mir_with_ur()
    agv.run()
