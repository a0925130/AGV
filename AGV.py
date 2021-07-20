import numpy as np
import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
# planeId = p.loadURDF("ur_mir_data/plane.urdf")
planeId = p.loadURDF("plane.urdf")
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
robot = p.loadURDF("ur_mir_data/mir_ur.urdf")

numJoints = p.getNumJoints(robot)
for joint in range(numJoints):
    print(p.getJointInfo(robot, joint))
    p.setJointMotorControl(robot, joint, p.POSITION_CONTROL, 0, 100)


def setCameraPicAndGetPic(robot_id: int, robot_link: int, width: int = 224, height: int = 224):
    # basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id)
    basePos = p.getLinkState(robot_id, robot_link)[0]
    baseOrientation = p.getLinkState(robot_id, robot_link)[1]
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

    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=width, height=height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
    )
    return width, height, rgbImg, depthImg, segImg


# p.setJointMotorControl2(boxId, 21, p.POSITION_CONTROL, targetVelocity=-30)
# p.setJointMotorControl2(boxId, 22, p.POSITION_CONTROL, targetVelocity=10)
# p.setJointMotorControl2(boxId, 23, p.POSITION_CONTROL, targetVelocity=5)
# p.setJointMotorControl2(boxId, 24, p.POSITION_CONTROL, targetVelocity=5)
# p.setJointMotorControl2(boxId, 20, p.VELOCITY_CONTROL, targetVelocity=5)
# p.setJointMotorControl2(robot, 23, p.POSITION_CONTROL, targetVelocity=5)
p.setJointMotorControl2(robot, 8, p.VELOCITY_CONTROL, targetVelocity=-15)
p.setJointMotorControl2(robot, 7, p.VELOCITY_CONTROL, targetVelocity=-15)
while 1:
    setCameraPicAndGetPic(robot, 28)
    time.sleep(1. / 240.)
