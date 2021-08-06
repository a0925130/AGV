from threading import Thread
import numpy as np
import pybullet as p
import time
import pybullet_data
import math
import matplotlib.pyplot as plt
point_x = []
point_y = []
numRays = 16
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

    def Laser(self):
        rayFrom_front = []
        rayFrom_back = []
        rayTo_front = []
        rayTo_back = []
        for i in range(numRays):
            results = []
            basePos_front = p.getLinkState(self, 3)[0]
            ray_front_x = basePos_front[0] + rayLen * math.sin(2. * math.pi * float(i) / numRays)
            ray_front_y = basePos_front[1] + rayLen * math.cos(2. * math.pi * float(i) / numRays)
            ray_front_z = basePos_front[2]
            rayFrom_front.append(basePos_front)
            rayTo_front.append([ray_front_x, ray_front_y, ray_front_z])
            p.addUserDebugLine(rayFrom_front[i], rayTo_front[i], rayColor)
            results.append(p.rayTest(rayFrom_front[i], rayTo_front[i]))

            basePos_back = p.getLinkState(self, 4)[0]
            ray_back_x = basePos_back[0] + rayLen * math.sin(2. * math.pi * float(i) / numRays)
            ray_back_y = basePos_back[1] + rayLen * math.cos(2. * math.pi * float(i) / numRays)
            ray_back_z = basePos_back[2]
            rayFrom_back.append(basePos_back)
            rayTo_back.append([ray_back_x, ray_back_y, ray_back_z])
            p.addUserDebugLine(rayFrom_back[i], rayTo_back[i], rayColor)
            results.append(p.rayTest(rayFrom_back[i], rayTo_back[i]))

            for j in range(len(results)):
                if basePos_front[0] > basePos_back[0]:
                    if basePos_front[1] > basePos_back[1]:
                        if not ((basePos_front[0] > results[j][0][3][0]) & (basePos_back[0] < results[j][0][3][0])) & (
                                (basePos_front[1] > results[j][0][3][1]) & (basePos_back[1] < results[j][0][3][1])):
                            point_x.append(results[j][0][3][0])
                            point_y.append(results[j][0][3][1])
                    else:
                        if not ((basePos_front[0] > results[j][0][3][0]) & (basePos_back[0] < results[j][0][3][0])) & (
                                (basePos_front[1] < results[j][0][3][1]) & (basePos_back[1] > results[j][0][3][1])):
                            point_x.append(results[j][0][3][0])
                            point_y.append(results[j][0][3][1])
                else:
                    if basePos_front[1] > basePos_back[1]:
                        if not ((basePos_front[0] < results[j][0][3][0]) & (basePos_back[0] > results[j][0][3][0])) & (
                                (basePos_front[1] > results[j][0][3][1]) & (basePos_back[1] < results[j][0][3][1])):
                            point_x.append(results[j][0][3][0])
                            point_y.append(results[j][0][3][1])
                    else:
                        if not ((basePos_front[0] < results[j][0][3][0]) & (basePos_back[0] > results[j][0][3][0])) & (
                                (basePos_front[1] < results[j][0][3][1]) & (basePos_back[1] > results[j][0][3][1])):
                            point_x.append(results[j][0][3][0])
                            point_y.append(results[j][0][3][1])
        p.removeAllUserDebugItems()

    def run(self):

        try:
            while True:
                do_laser = Thread(target=mir_with_ur.Laser, args=(self.robot,))
                do_camera = Thread(target=mir_with_ur.Camera, args=(self.robot,))
                do_laser.start()
                do_camera.start()
                mapping(point_x, point_y)
                do_laser.join()
                do_camera.join()
        finally:
            p.disconnect()


if __name__ == '__main__':
    agv = mir_with_ur()
    agv.run()
