"""

FastSLAM 2.0 example

author: Atsushi Sakai (@Atsushi_twi)

"""

from threading import Thread
import pybullet as p
import time
import pybullet_data
import math
import matplotlib.pyplot as plt
import numpy as np

# Fast SLAM covariance
Q = np.diag([3.0, np.deg2rad(10.0)]) ** 2
R = np.diag([1.0, np.deg2rad(20.0)]) ** 2

#  Simulation parameter
Q_sim = np.diag([0.3, np.deg2rad(2.0)]) ** 2
R_sim = np.diag([0.5, np.deg2rad(10.0)]) ** 2
OFFSET_YAW_RATE_NOISE = 0.01

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]
N_PARTICLE = 100  # number of particle
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling

show_animation = True
numRays = 8
rayLen = 10
rayColor = [0, 0, 1]
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
s1_pre = np.zeros((numRays, 3))
s2_pre = np.zeros((numRays, 3))
s3_pre = np.zeros((numRays, 3))
s1_after = np.zeros((numRays, 3))
s2_after = np.zeros((numRays, 3))
s3_after = np.zeros((numRays, 3))
physicsClient = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
obstacleSratPos_1 = [3.5, 2.5, 1]
roomStartPos = [0, 0, 1]
planeId = p.loadURDF("plane.urdf")
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
p.setRealTimeSimulation(1)
numJoints = p.getNumJoints(robot)
baseYaw = p.getEulerFromQuaternion(p.getLinkState(robot, 3)[1])[2]
for joint in range(numJoints):
    print(p.getJointInfo(robot, joint))
    p.setJointMotorControl(robot, joint, p.POSITION_CONTROL, 0, 100)


class Particle:

    def __init__(self, N_LM):
        self.w = 1.0 / N_PARTICLE
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.P = np.eye(3)
        # landmark x-y positions
        self.lm = np.zeros((N_LM, LM_SIZE))
        # landmark position covariance
        self.lmP = np.zeros((N_LM * LM_SIZE, LM_SIZE))


def fast_slam2(particles, u, z):
    particles = predict_particles(particles, u)

    particles = update_with_observation(particles, z)

    particles = resampling(particles)

    return particles


def normalize_weight(particles):
    sum_w = sum([p.w for p in particles])

    try:
        for i in range(N_PARTICLE):
            particles[i].w /= sum_w
    except ZeroDivisionError:
        for i in range(N_PARTICLE):
            particles[i].w = 1.0 / N_PARTICLE

        return particles

    return particles


def calc_final_state(particles):
    xEst = np.zeros((STATE_SIZE, 1))

    particles = normalize_weight(particles)

    for i in range(N_PARTICLE):
        xEst[0, 0] += particles[i].w * particles[i].x
        xEst[1, 0] += particles[i].w * particles[i].y
        xEst[2, 0] += particles[i].w * particles[i].yaw

    xEst[2, 0] = pi_2_pi(xEst[2, 0])

    return xEst


def predict_particles(particles, u):
    for i in range(N_PARTICLE):
        px = np.zeros((STATE_SIZE, 1))
        px[0, 0] = particles[i].x
        px[1, 0] = particles[i].y
        px[2, 0] = particles[i].yaw
        ud = u + (np.random.randn(1, 2) @ R ** 0.5).T  # add noise
        px = motion_model(px, ud)
        particles[i].x = px[0, 0]
        particles[i].y = px[1, 0]
        particles[i].yaw = px[2, 0]

    return particles


def add_new_lm(particle, z, Q_cov):
    r = z[0]
    b = z[1]
    lm_id = int(z[2])

    s = math.sin(pi_2_pi(particle.yaw + b))
    c = math.cos(pi_2_pi(particle.yaw + b))

    particle.lm[lm_id, 0] = particle.x + r * c
    particle.lm[lm_id, 1] = particle.y + r * s

    # covariance
    dx = r * c
    dy = r * s
    d2 = dx ** 2 + dy ** 2
    d = math.sqrt(d2)
    Gz = np.array([[dx / d, dy / d],
                   [-dy / d2, dx / d2]])
    particle.lmP[2 * lm_id:2 * lm_id + 2] = np.linalg.inv(
        Gz) @ Q_cov @ np.linalg.inv(Gz.T)

    return particle


def compute_jacobians(particle, xf, Pf, Q_cov):
    dx = xf[0, 0] - particle.x
    dy = xf[1, 0] - particle.y
    d2 = dx ** 2 + dy ** 2
    d = math.sqrt(d2)

    zp = np.array(
        [d, pi_2_pi(math.atan2(dy, dx) - particle.yaw)]).reshape(2, 1)

    Hv = np.array([[-dx / d, -dy / d, 0.0],
                   [dy / d2, -dx / d2, -1.0]])

    Hf = np.array([[dx / d, dy / d],
                   [-dy / d2, dx / d2]])

    Sf = Hf @ Pf @ Hf.T + Q_cov

    return zp, Hv, Hf, Sf


def update_kf_with_cholesky(xf, Pf, v, Q_cov, Hf):
    PHt = Pf @ Hf.T
    S = Hf @ PHt + Q_cov

    S = (S + S.T) * 0.5
    SChol = np.linalg.cholesky(S).T
    SCholInv = np.linalg.inv(SChol)
    W1 = PHt @ SCholInv
    W = W1 @ SCholInv.T

    x = xf + W @ v
    P = Pf - W1 @ W1.T

    return x, P


def update_landmark(particle, z, Q_cov):
    lm_id = int(z[2])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])

    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q_cov)

    dz = z[0:2].reshape(2, 1) - zp
    dz[1, 0] = pi_2_pi(dz[1, 0])

    xf, Pf = update_kf_with_cholesky(xf, Pf, dz, Q, Hf)

    particle.lm[lm_id, :] = xf.T
    particle.lmP[2 * lm_id:2 * lm_id + 2, :] = Pf

    return particle


def compute_weight(particle, z, Q_cov):
    lm_id = int(z[2])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])
    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q_cov)

    dz = z[0:2].reshape(2, 1) - zp
    dz[1, 0] = pi_2_pi(dz[1, 0])

    try:
        invS = np.linalg.inv(Sf)
    except np.linalg.linalg.LinAlgError:
        return 1.0

    num = math.exp(-0.5 * dz.T @ invS @ dz)
    den = 2.0 * math.pi * math.sqrt(np.linalg.det(Sf))

    w = num / den

    return w


def proposal_sampling(particle, z, Q_cov):
    lm_id = int(z[2])
    xf = particle.lm[lm_id, :].reshape(2, 1)
    Pf = particle.lmP[2 * lm_id:2 * lm_id + 2]
    # State
    x = np.array([particle.x, particle.y, particle.yaw]).reshape(3, 1)
    P = particle.P
    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q_cov)

    Sfi = np.linalg.inv(Sf)
    dz = z[0:2].reshape(2, 1) - zp
    dz[1] = pi_2_pi(dz[1])

    Pi = np.linalg.inv(P)

    particle.P = np.linalg.inv(Hv.T @ Sfi @ Hv + Pi)  # proposal covariance
    x += particle.P @ Hv.T @ Sfi @ dz  # proposal mean

    particle.x = x[0, 0]
    particle.y = x[1, 0]
    particle.yaw = x[2, 0]

    return particle


def update_with_observation(particles, z):
    for iz in range(len(z[0, :])):
        landmark_id = int(z[2, iz])

        for ip in range(N_PARTICLE):
            # new landmark
            if abs(particles[ip].lm[landmark_id, 0]) <= 0.01:
                particles[ip] = add_new_lm(particles[ip], z[:, iz], Q)
            # known landmark
            else:
                w = compute_weight(particles[ip], z[:, iz], Q)
                particles[ip].w *= w

                particles[ip] = update_landmark(particles[ip], z[:, iz], Q)
                particles[ip] = proposal_sampling(particles[ip], z[:, iz], Q)

    return particles


def resampling(particles):
    """
    low variance re-sampling
    """

    particles = normalize_weight(particles)

    pw = []
    for i in range(N_PARTICLE):
        pw.append(particles[i].w)

    pw = np.array(pw)

    n_eff = 1.0 / (pw @ pw.T)  # Effective particle number

    if n_eff < NTH:  # resampling
        w_cum = np.cumsum(pw)
        base = np.cumsum(pw * 0.0 + 1 / N_PARTICLE) - 1 / N_PARTICLE
        resample_id = base + np.random.rand(base.shape[0]) / N_PARTICLE

        inds = []
        ind = 0
        for ip in range(N_PARTICLE):
            while (ind < w_cum.shape[0] - 1) \
                    and (resample_id[ip] > w_cum[ind]):
                ind += 1
            inds.append(ind)

        tmp_particles = particles[:]
        for i in range(len(inds)):
            particles[i].x = tmp_particles[inds[i]].x
            particles[i].y = tmp_particles[inds[i]].y
            particles[i].yaw = tmp_particles[inds[i]].yaw
            particles[i].lm = tmp_particles[inds[i]].lm[:, :]
            particles[i].lmP = tmp_particles[inds[i]].lmP[:, :]
            particles[i].w = 1.0 / N_PARTICLE

    return particles


def calc_input1(s1, s2, yaw1, yaw2, t1, t2):
    v = (math.sqrt(((s2[0] - s1[0]) ** 2) + (s2[1] - s1[1]) ** 2)) / (t2 - t1)
    yaw_rate = (yaw2 - yaw1) / (t2 - t1)
    u = np.array([v, yaw_rate]).reshape(2, 1)

    return u


def calc_input(time):
    if time <= 3.0:  # wait at first
        v = 0.0
        yaw_rate = 0.0
    else:
        v = 1.0  # [m/s]
        yaw_rate = 0.1  # [rad/s]

    u = np.array([v, yaw_rate]).reshape(2, 1)

    return u


def observation(xTrue, xd, u, RFID):
    # calc true state
    xTrue = motion_model(xTrue, u)

    # add noise to range observation
    z = np.zeros((3, 0))

    for i in range(len(RFID[:, 0])):

        dx = RFID[i, 0] - xTrue[0, 0]
        dy = RFID[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5  # add noise
            angle_noise = np.random.randn() * Q_sim[1, 1] ** 0.5
            angle_with_noise = angle + angle_noise  # add noise
            zi = np.array([dn, pi_2_pi(angle_with_noise), i]).reshape(3, 1)
            z = np.hstack((z, zi))

    # add noise to input
    ud1 = u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5
    ud2 = u[1, 0] + np.random.randn() * R_sim[
        1, 1] ** 0.5 + OFFSET_YAW_RATE_NOISE
    ud = np.array([ud1, ud2]).reshape(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])

    x = F @ x + B @ u

    x[2, 0] = pi_2_pi(x[2, 0])

    return x


def Laser():
    rayFrom1 = []
    rayTo1 = []
    rayFrom2 = []
    rayTo2 = []
    rayFrom3 = []
    rayTo3 = []
    point_x = []
    point_y = []
    yaw_1 = np.zeros(numRays)
    yaw_2 = np.zeros(numRays)
    yaw_3 = np.zeros(numRays)
    yaw1 = 0.
    yaw2 = 0.
    rfid = []
    sd1 = np.zeros(2)
    sd2 = np.zeros(2)
    td1 = 0.
    td2 = 0.
    v = 0.
    yaw_rate = 0.
    for i in range(numRays):
        td1 = time.time()
        basePos = p.getLinkState(robot, 3)[0]
        Yaw = p.getEulerFromQuaternion(p.getLinkState(robot, 3)[1])[2] - baseYaw
        yaw_1[i] = Yaw
        ray_x1 = basePos[0] + rayLen * math.sin((0.5 * math.pi * float(i) / numRays) - Yaw)
        ray_y1 = basePos[1] + rayLen * math.cos((0.5 * math.pi * float(i) / numRays) - Yaw)
        ray_z1 = basePos[2]
        rayFrom1.append(basePos)
        rayTo1.append([ray_x1, ray_y1, ray_z1])
        yaw1 = Yaw
        sd1 = np.array([ray_x1, ray_y1])
        # theta1_pre[i] = theta1_after[i]
        # theta1_after[i] = yaw_1[i]
        # theta1 = theta1_after[i] - theta1_pre[i]
        # for j in range(3):
        #     s1_pre[i][j] = s1_after[i][j]
        #     s1_after[i][j] = rayFrom1[i][j]
        # pos1 = s1_after[i] - s1_pre[i]
        # s1 = math.sqrt((pos1[0] ** 2) + (pos1[1] ** 2))
        # t1_pre[i] = t1_after[i]
        # t1_after[i] = time.time()
        # t1 = t1_after[i] - t1_pre[i]
        # v1 = s1 / t1
        # yaw1_rate = theta1 / t1
        p.addUserDebugLine(rayFrom1[i], rayTo1[i], rayColor)
        results1 = p.rayTest(rayFrom1[i], rayTo1[i])
        rfid.append([(results1[0][3][0] - rayFrom1[i][0]), (results1[0][3][1] - rayFrom1[i][1])])
        # v.append(v1)
        # yaw_rate.append(yaw1_rate)
        # point_x.append(results1[0][3][0] - rayFrom1[i][0])
        # point_y.append(results1[0][3][1] - rayFrom1[i][1])
        basePos = p.getLinkState(robot, 3)[0]
        Yaw = p.getEulerFromQuaternion(p.getLinkState(robot, 3)[1])[2] - baseYaw
        yaw_2[i] = Yaw
        ray_x2 = basePos[0] + rayLen * math.sin((0.5 * math.pi * float(i) / numRays) + (0.5 * math.pi) - Yaw)
        ray_y2 = basePos[1] + rayLen * math.cos((0.5 * math.pi * float(i) / numRays) + (0.5 * math.pi) - Yaw)
        ray_z2 = basePos[2]
        rayFrom2.append(basePos)
        rayTo2.append([ray_x2, ray_y2, ray_z2])
        # theta2_pre[i] = theta2_after[i]
        # theta2_after[i] = yaw_2[i]
        # theta2 = theta2_after[i] - theta2_pre[i]
        # for j in range(3):
        #     s2_pre[i][j] = s2_after[i][j]
        #     s2_after[i][j] = rayFrom2[i][j]
        # pos2 = s2_after[i] - s2_pre[i]
        # s2 = math.sqrt((pos2[0] ** 2) + (pos2[1] ** 2))
        # t2_pre[i] = t2_after[i]
        # t2_after[i] = time.time()
        # t2 = t2_after[i] - t2_pre[i]
        # v2 = s2 / t2
        # yaw2_rate = theta2 / t2
        p.addUserDebugLine(rayFrom2[i], rayTo2[i], rayColor)
        results2 = p.rayTest(rayFrom2[i], rayTo2[i])
        rfid.append([(results2[0][3][0] - rayFrom2[i][0]), (results2[0][3][1] - rayFrom2[i][1])])
        # v.append(v2)
        # yaw_rate.append(yaw2_rate)
        # point_x.append(results2[0][3][0] - rayFrom2[i][0])
        # point_y.append(results2[0][3][1] - rayFrom2[i][1])

        basePos = p.getLinkState(robot, 3)[0]
        Yaw = p.getEulerFromQuaternion(p.getLinkState(robot, 3)[1])[2] - baseYaw
        yaw_3[i] = Yaw
        ray_x3 = basePos[0] + rayLen * math.sin((0.5 * math.pi * float(i) / numRays) + (1.5 * math.pi) - Yaw)
        ray_y3 = basePos[1] + rayLen * math.cos((0.5 * math.pi * float(i) / numRays) + (1.5 * math.pi) - Yaw)
        ray_z3 = basePos[2]
        rayFrom3.append(basePos)
        rayTo3.append([ray_x3, ray_y3, ray_z3])
        yaw2 = Yaw
        sd1 = np.array([ray_x1, ray_y1])
        # theta3_pre[i] = theta3_after[i]
        # theta3_after[i] = yaw_3[i]
        # theta3 = theta3_after[i] - theta3_pre[i]
        # for j in range(3):
        #     s3_pre[i][j] = s3_after[i][j]
        #     s3_after[i][j] = rayFrom3[i][j]
        # pos3 = s3_after[i] - s3_pre[i]
        # s3 = math.sqrt((pos3[0] ** 2) + (pos3[1] ** 2))
        # t3_pre[i] = t3_after[i]
        # t3_after[i] = time.time()
        # t3 = t3_after[i] - t3_pre[i]
        # v3 = s3 / t3
        # yaw3_rate = theta3 / t3
        p.addUserDebugLine(rayFrom3[i], rayTo3[i], rayColor)
        results3 = p.rayTest(rayFrom3[i], rayTo3[i])
        rfid.append([(results3[0][3][0] - rayFrom3[i][0]), (results3[0][3][1] - rayFrom3[i][1])])
        # v.append(v3)
        # yaw_rate.append(yaw3_rate)
        # point_x.append(results3[0][3][0] - rayFrom3[i][0])
        # point_y.append(results3[0][3][1] - rayFrom3[i][1])
        td2 = time.time()
    # rfid = np.concatenate([point_x, point_y])
    u = calc_input1(sd1, sd2, yaw1, yaw2, 0, 1)

    return np.array(rfid), u


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main():
    print(__file__ + " start!!")

    time = 0.0
    # RFID positions [x, y]
    # RFID = np.array([[10.0, -2.0],
    #                  [15.0, 10.0],
    #                  [15.0, 15.0],
    #                  [10.0, 20.0],
    #                  [3.0, 15.0],
    #                  [-5.0, 20.0],
    #                  [-5.0, 5.0],
    #                  [-10.0, 15.0]
    #                  ])
    RFID, _ = np.array(Laser())
    print("RFID = ", RFID)
    n_landmark = RFID.shape[0]

    # State Vector [x y yaw v]'
    xEst = np.zeros((STATE_SIZE, 1))  # SLAM estimation
    xTrue = np.zeros((STATE_SIZE, 1))  # True state
    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    particles = [Particle(n_landmark) for _ in range(N_PARTICLE)]

    while SIM_TIME >= time:
        # RFID, u = np.array(Laser())
        time += DT
        u = calc_input(time)
        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID)

        particles = fast_slam2(particles, ud, z)

        xEst = calc_final_state(particles)

        x_state = xEst[0: STATE_SIZE]

        # store data history
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(RFID[:, 0], RFID[:, 1], "*k")

            for iz in range(len(z[:, 0])):
                landmark_id = int(z[2, iz])
                plt.plot([xEst[0], RFID[landmark_id, 0]], [
                    xEst[1], RFID[landmark_id, 1]], "-k")

            for i in range(N_PARTICLE):
                plt.plot(particles[i].x, particles[i].y, ".r")
                plt.plot(particles[i].lm[:, 0], particles[i].lm[:, 1], "xb")

            plt.plot(hxTrue[0, :], hxTrue[1, :], "-b")
            plt.plot(hxDR[0, :], hxDR[1, :], "-k")
            plt.plot(hxEst[0, :], hxEst[1, :], "-r")
            plt.plot(xEst[0], xEst[1], "xk")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()
