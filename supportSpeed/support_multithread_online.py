"""
Copy-Right by: PHAN HONG SON
Univ: Sungkyunkwan University
"""


#import nessessary libraries
import numpy as np
import math
fps = 50
def calculateSpeeds(data, track):
    vp1, vp2, vp3, pp, roadPlane, focal = computeCameraCalibration(data["camera_calibration"]["vp1"],
                                                            data["camera_calibration"]["vp2"],
                                                            data["camera_calibration"]["pp"])
    projector = lambda p: getWorldCoordinagesOnRoadPlane(p, focal, roadPlane, pp)
    startFrame = track.frames[0]
    endFrame = track.frames[-1]
    elapsedTime = abs(endFrame - startFrame)/fps
    startPoint = np.array([track.poses[0][0], track.poses[0][1], 1])
    endPoint = np.array([track.poses[-1][0], track.poses[-1][1], 1])
    distance = data["camera_calibration"]["scale"] * np.linalg.norm(projector(startPoint) - projector(endPoint))
    return distance/elapsedTime * 3.6


def computeCameraCalibration(_vp1, _vp2, _pp):
    vp1 = np.concatenate((_vp1, [1]))
    vp2 = np.concatenate((_vp2, [1]))
    pp = np.concatenate((_pp, [1]))
    focal = getFocal(vp1, vp2, pp)
    vp1W = np.concatenate((_vp1, [focal]))
    vp2W = np.concatenate((_vp2, [focal]))
    ppW = np.concatenate((_pp, [0]))
    vp3W = np.cross(vp1W-ppW, vp2W-ppW)
    vp3 = np.concatenate((vp3W[0:2]/vp3W[2]*focal + ppW[0:2], [1]))
    vp3Direction = np.concatenate((vp3[0:2], [focal]))-ppW
    roadPlane = np.concatenate((vp3Direction/np.linalg.norm(vp3Direction), [10]))
    return vp1, vp2, vp3, pp, roadPlane, focal


def getFocal(vp1, vp2, pp):
    return math.sqrt(- np.dot(vp1[0:2]-pp[0:2], vp2[0:2]-pp[0:2]))


def getWorldCoordinagesOnRoadPlane(p, focal, roadPlane, pp):
    p = p/p[2]
    pp = pp/pp[2]
    ppW = np.concatenate((pp[0:2], [0]))
    pW = np.concatenate((p[0:2], [focal]))
    dirVec = pW - ppW
    t = -np.dot(roadPlane, np.concatenate((ppW, [1])))/np.dot(roadPlane[0:3], dirVec)
    return ppW + t*dirVec

# DEBUG code for a new way to calculate speed
def speed_estimation(track):
    startFrame = track.frames[0]
    endFrame = track.frames[-1]
    elapsedTime = abs(endFrame - startFrame) / fps
    startPoint = np.array([track.poses[0][0], track.poses[0][1], 1])
    endPoint = np.array([track.poses[-1][0], track.poses[-1][1], 1])
    distance = math.sqrt(np.sum(np.square(getWorldPoint(endPoint)[0:2]/getWorldPoint(endPoint)[2]- getWorldPoint(startPoint)[0:2]/getWorldPoint(startPoint)[2])))
    print('This is world point {} and {} '.format(getWorldPoint(endPoint)[0:2], getWorldPoint(startPoint)[0:2]))
    print('This is distance: ', distance)
    return distance*0.9877456 / elapsedTime * 3.6
def getWorldPoint(point):
    H = np.array([[2.23248225e-02, -2.08625814e-02, -2.57720710e+00],
                  [1.37239783e-02, 1.44999149e-01, -3.91060193e+01],
                  [1.28992762e-04, 2.62027346e-03, 1.00000000e+00]]
                 )
    return np.dot(H, point)