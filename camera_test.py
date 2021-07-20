import time

import pybullet as pb
import pybullet_data
import os, glob, random
import numpy as np

physicsClient = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = pb.loadURDF('plane.urdf')
pb.setGravity(0, 0, -9.8)
camera_start = [0, 0, 1]
ball_start = [1, 0, 0.04]
multiBodyId = pb.loadURDF("C:/Users/Cat/Desktop/camera.urdf", camera_start)
ball = pb.loadURDF("ball.urdf", ball_start)

numJoints = pb.getNumJoints(multiBodyId)
for joint in range(numJoints):
    print(pb.getJointInfo(multiBodyId, joint))


def setCameraPicAndGetPic(robot_id : int, width : int = 224, height : int = 224, physicsClientId : int = 0):
    """
    给合成摄像头设置图像并返回robot_id对应的图像
    摄像头的位置为miniBox前头的位置
    """
    basePos, baseOrientation = pb.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)
    # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
    matrix = pb.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)
    tx_vec = np.array([matrix[0], matrix[3], matrix[6]])              # 变换后的x轴
    tz_vec = np.array([matrix[2], matrix[5], matrix[8]])              # 变换后的z轴

    basePos = np.array(basePos)
    # 摄像头的位置
    cameraPos = basePos + BASE_RADIUS * tx_vec + 0.5 * BASE_THICKNESS * tz_vec
    targetPos = cameraPos + 1 * tx_vec

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=tz_vec,
        physicsClientId=physicsClientId
    )
    projectionMatrix = pb.computeProjectionMatrixFOV(
        fov=50.0,               # 摄像头的视线夹角
        aspect=1.0,
        nearVal=0.01,            # 摄像头焦距下限
        farVal=20,               # 摄像头能看上限
        physicsClientId=physicsClientId
    )


pb.setRealTimeSimulation(1)
time.sleep(5.)



while True:
    pos = np.array(pb.getLinkState(multiBodyId, 0))[0]
    pos1 = np.array(pb.getLinkState(multiBodyId, 1))[0]
    view = np.array(pos, dtype=float) - np.array(pos1, dtype=float)
    print(np.array(pb.getLinkState(multiBodyId, 1))[0])
    print(np.array(pb.getLinkState(multiBodyId, 0))[0])
    print(view * 100)
    camera(pos, view*100)
    time.sleep(1. / 240.)