import numpy as np
import pybullet as p
import time
import pybullet_data
from opt_einsum.backends import torch

physicsClient = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("ur_mir_data/plane.urdf")

roomStartPos = [0, 0, 1]
obstacleSratPos_1 = [3.5, 2.5, 1]
obstacleSratPos_2 = [-3, 1, 1]
obstacleSratPos_3 = [3, -4, 1]
obstacleSratPos_4 = [6, 0, 1]
obstacleSratPos_5 = [-3, -4, 1]
obstacleSratPos_6 = [-5, -2, 1]

room = p.loadURDF("ur_mir_data/room.urdf", globalScaling=5, basePosition=roomStartPos)
obstacle_1 = p.loadURDF("ur_mir_data/obstacle.urdf", basePosition=obstacleSratPos_1)
obstacle_2 = p.loadURDF("ur_mir_data/obstacle.urdf", basePosition=obstacleSratPos_2)
obstacle_3 = p.loadURDF("ur_mir_data/obstacle.urdf", basePosition=obstacleSratPos_3)
obstacle_4 = p.loadURDF("ur_mir_data/obstacle.urdf", basePosition=obstacleSratPos_4)
obstacle_5 = p.loadURDF("ur_mir_data/obstacle.urdf", basePosition=obstacleSratPos_5)
obstacle_6 = p.loadURDF("ur_mir_data/obstacle.urdf", basePosition=obstacleSratPos_6)
boxId = p.loadURDF("ur_mir_data/mir_ur.urdf")

numJoints = p.getNumJoints(boxId)
for joint in range(numJoints):
    print(p.getJointInfo(boxId, joint))
    p.setJointMotorControl(boxId, joint, p.POSITION_CONTROL, 0, 100)

# p.setJointMotorControl2(boxId, 21, p.POSITION_CONTROL, targetVelocity=-30)
# p.setJointMotorControl2(boxId, 22, p.POSITION_CONTROL, targetVelocity=10)
# p.setJointMotorControl2(boxId, 23, p.POSITION_CONTROL, targetVelocity=5)
# p.setJointMotorControl2(boxId, 24, p.POSITION_CONTROL, targetVelocity=5)
# p.setJointMotorControl2(boxId, 20, p.VELOCITY_CONTROL, targetVelocity=5)

# p.setJointMotorControl2(boxId, 8, p.VELOCITY_CONTROL, targetVelocity=15)
# p.setJointMotorControl2(boxId, 9, p.VELOCITY_CONTROL, targetVelocity=15)
p.setRealTimeSimulation(1)
while 1:
    time.sleep(1. / 240.)
