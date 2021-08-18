from threading import Thread
from multiprocessing import Process
import numpy as np
import pybullet as p
import time
import pybullet_data
import math
import matplotlib.pyplot as plt
import Slam_fuction

point_x = []
point_y = []
point_z = []
numRays = 8
rayLen = 10
rayColor = [0, 0, 1]
t1 = 0
t2 = 0
t3 = 0
t1_pre = np.zeros(numRays)
t2_pre = np.zeros(numRays)
t3_pre = np.zeros(numRays)
t1_after = np.zeros(numRays)
t2_after = np.zeros(numRays)
t3_after = np.zeros(numRays)
theta1 = 0
theta2 = 0
theta3 = 0
theta1_pre = np.zeros(numRays)
theta2_pre = np.zeros(numRays)
theta3_pre = np.zeros(numRays)
theta1_after = np.zeros(numRays)
theta2_after = np.zeros(numRays)
theta3_after = np.zeros(numRays)
yaw1_rate = 0
yaw2_rate = 0
yaw3_rate = 0
v1 = 0
v2 = 0
v3 = 0
s1 = 0
s2 = 0
s3 = 0
pos1 = 0
pos2 = 0
pos3 = 0
s1_pre = np.zeros((numRays, 3))
s2_pre = np.zeros((numRays, 3))
s3_pre = np.zeros((numRays, 3))
s1_after = np.zeros((numRays, 3))
s2_after = np.zeros((numRays, 3))
s3_after = np.zeros((numRays, 3))
rfid = []
u = []
n_landmark = 0

def mapping(RFID, particles, hxTrue, hxDR, hxEst, xEst):
    plt.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event', lambda event:
        [exit(0) if event.key == 'escape' else None])
    plt.plot(RFID[:, 0], RFID[:, 1], "*k")

    for i in range(Slam_fuction.N_PARTICLE):
        plt.plot(particles[i].x, particles[i].y, ".r")
        plt.plot(particles[i].lm[:, 0], particles[i].lm[:, 1], "xb")

    plt.plot(hxTrue[0, :], hxTrue[1, :], "-b")
    plt.plot(hxDR[0, :], hxDR[1, :], "-k")
    plt.plot(hxEst[0, :], hxEst[1, :], "-r")
    plt.plot(xEst[0], xEst[1], "xk")
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.001)


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
        self.z = 0
        p.setRealTimeSimulation(1)
        numJoints = p.getNumJoints(self.robot)
        self.baseYaw = p.getEulerFromQuaternion(p.getLinkState(self.robot, 3)[1])[2]
        for joint in range(numJoints):
            print(p.getJointInfo(self.robot, joint))
            p.setJointMotorControl(self.robot, joint, p.POSITION_CONTROL, 0, 100)
        # p.setJointMotorControl2(self.robot, 8, p.VELOCITY_CONTROL, targetVelocity=-15)
        # p.setJointMotorControl2(self.robot, 7, p.VELOCITY_CONTROL, targetVelocity=-15)
        # p.setJointMotorControl2(self.robot, 19, p.POSITION_CONTROL, targetPosition=1)

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
        yaw_1 = np.zeros(numRays)
        yaw_2 = np.zeros(numRays)
        yaw_3 = np.zeros(numRays)
        for i in range(numRays):
            results1 = []
            results2 = []
            results3 = []
            basePos = p.getLinkState(self, 3)[0]
            Yaw = p.getEulerFromQuaternion(p.getLinkState(self, 3)[1])[2] - Yaw1
            yaw_1[i] = Yaw
            ray_x1 = basePos[0] + rayLen * math.sin((0.5 * math.pi * float(i) / numRays) - Yaw)
            ray_y1 = basePos[1] + rayLen * math.cos((0.5 * math.pi * float(i) / numRays) - Yaw)
            ray_z1 = basePos[2]
            rayFrom1.append(basePos)
            rayTo1.append([ray_x1, ray_y1, ray_z1])
            theta1_pre[i] = theta1_after[i]
            theta1_after[i] = yaw_1[i]
            theta1 = theta1_after[i] - theta1_pre[i]
            for j in range(3):
                s1_pre[i][j] = s1_after[i][j]
                s1_after[i][j] = rayFrom1[i][j]
            pos1 = s1_after[i] - s1_pre[i]
            s1 = math.sqrt((pos1[0] ** 2) + (pos1[1] ** 2))
            t1_pre[i] = t1_after[i]
            t1_after[i] = time.time()
            t1 = t1_after[i] - t1_pre[i]
            v1 = s1 / t1
            yaw1_rate = theta1 / t1
            p.addUserDebugLine(rayFrom1[i], rayTo1[i], rayColor)
            results1 = p.rayTest(rayFrom1[i], rayTo1[i])
            point_x.append(float(results1[0][3][0]) - rayFrom1[i][0])
            point_y.append(float(results1[0][3][1]) - rayFrom1[i][1])
            u.append(np.array([v1, yaw1_rate]).reshape(2, 1))

            basePos = p.getLinkState(self, 3)[0]
            Yaw = p.getEulerFromQuaternion(p.getLinkState(self, 3)[1])[2] - Yaw1
            yaw_2[i] = Yaw
            ray_x2 = basePos[0] + rayLen * math.sin((0.5 * math.pi * float(i) / numRays) + (0.5 * math.pi) - Yaw)
            ray_y2 = basePos[1] + rayLen * math.cos((0.5 * math.pi * float(i) / numRays) + (0.5 * math.pi) - Yaw)
            ray_z2 = basePos[2]
            rayFrom2.append(basePos)
            rayTo2.append([ray_x2, ray_y2, ray_z2])
            theta2_pre[i] = theta2_after[i]
            theta2_after[i] = yaw_2[i]
            theta2 = theta2_after[i] - theta2_pre[i]
            for j in range(3):
                s2_pre[i][j] = s2_after[i][j]
                s2_after[i][j] = rayFrom2[i][j]
            pos2 = s2_after[i] - s2_pre[i]
            s2 = math.sqrt((pos2[0] ** 2) + (pos2[1] ** 2))
            t2_pre[i] = t2_after[i]
            t2_after[i] = time.time()
            t2 = t2_after[i] - t2_pre[i]
            v2 = s2 / t2
            yaw2_rate = theta2 / t2
            p.addUserDebugLine(rayFrom2[i], rayTo2[i], rayColor)
            results2 = p.rayTest(rayFrom2[i], rayTo2[i])
            point_x.append(float(results2[0][3][0]) - rayFrom2[i][0])
            point_y.append(float(results2[0][3][1]) - rayFrom2[i][1])
            u.append(np.array([v2, yaw2_rate]).reshape(2, 1))

            basePos = p.getLinkState(self, 3)[0]
            Yaw = p.getEulerFromQuaternion(p.getLinkState(self, 3)[1])[2] - Yaw1
            yaw_3[i] = Yaw
            ray_x3 = basePos[0] + rayLen * math.sin((0.5 * math.pi * float(i) / numRays) + (1.5 * math.pi) - Yaw)
            ray_y3 = basePos[1] + rayLen * math.cos((0.5 * math.pi * float(i) / numRays) + (1.5 * math.pi) - Yaw)
            ray_z3 = basePos[2]
            rayFrom3.append(basePos)
            rayTo3.append([ray_x3, ray_y3, ray_z3])
            theta3_pre[i] = theta3_after[i]
            theta3_after[i] = yaw_3[i]
            theta3 = theta3_after[i] - theta3_pre[i]
            for j in range(3):
                s3_pre[i][j] = s3_after[i][j]
                s3_after[i][j] = rayFrom3[i][j]
            pos3 = s3_after[i] - s3_pre[i]
            s3 = math.sqrt((pos3[0] ** 2) + (pos3[1] ** 2))
            t3_pre[i] = t3_after[i]
            t3_after[i] = time.time()
            t3 = t3_after[i] - t3_pre[i]
            v3 = s3 / t3
            yaw3_rate = theta3 / t3
            p.addUserDebugLine(rayFrom3[i], rayTo3[i], rayColor)
            results3 = p.rayTest(rayFrom3[i], rayTo3[i])
            point_x.append(float(results3[0][3][0]) - rayFrom3[i][0])
            point_y.append(float(results3[0][3][1]) - rayFrom3[i][1])

            rfid.append(np.array([point_x, point_y]))
            u.append(np.array([v3, yaw3_rate]).reshape(2, 1))

            print("u : ", u)
        p.removeAllUserDebugItems()

    def run(self):
        xEst = np.zeros((Slam_fuction.STATE_SIZE, 1))
        xTrue = np.zeros((Slam_fuction.STATE_SIZE, 1))
        xDR = np.zeros((Slam_fuction.STATE_SIZE, 1))
        hxEst = xEst
        hxTrue = xTrue
        hxDR = xTrue
        n_landmark = np.array(rfid).shape[0]
        particles = [Slam_fuction.Particle(n_landmark) for _ in range(Slam_fuction.N_PARTICLE)]
        try:
            while True:
                do_laser = Thread(target=mir_with_ur.Laser, args=(self.robot, self.baseYaw))
                do_camera = Thread(target=mir_with_ur.Camera, args=(self.robot,))
                do_laser.start()
                do_camera.start()

                # xTrue, z, xDR, ud = Slam_fuction.observation(xTrue, xDR, u, rfid)
                # particles = Slam_fuction.fast_slam1(particles, ud, z)
                # xEst = Slam_fuction.calc_final_state(particles)
                # x_state = xEst[0: Slam_fuction.STATE_SIZE]
                # hxEst = np.hstack((hxEst, x_state))
                # hxDR = np.hstack((hxDR, xDR))
                # hxTrue = np.hstack((hxTrue, xTrue))

                do_laser.join()
                do_camera.join()
        finally:
            p.disconnect()


if __name__ == '__main__':
    agv = mir_with_ur()
    agv.run()
