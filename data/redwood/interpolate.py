import sys
import numpy as np
import math


def smooth(x, y, xp, yp, t):
    a = 2 * x - 2 * y + xp + yp
    b = -3 * x + 3 * y - 2 * xp - yp
    c = xp
    d = x
    return a * t * t * t + b * t * t + c * t + d


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


def interpolate(filename, trans_per_sec = 0.5, rot_per_sec = 25):

    with open(filename) as f:
        lines = []
        for line in f:
            lines.append( [float(x) for x in line.split()] )

    frames = []

    for k in range(0, len(lines) - 1):
        if k == 0:
            ii = [a * 2 - b for a, b in zip(lines[k], lines[k + 1])]
        else:
            ii = lines[k - 1]

        i = lines[k]    
        j = lines[k + 1]

        if k == len(lines) - 2:
            jj = [b * 2 - a for a, b in zip(lines[k], lines[k + 1])]
        else:
            jj = lines[k + 2]

        dist = math.sqrt((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2 + (i[2] - j[2]) ** 2)
        angle = max(abs(i[3] - j[3]), abs(i[4] - j[4]))

        duration = max(dist / trans_per_sec, angle / rot_per_sec)

        for t in frange (0, 1, 1 / (duration * 30)):
            frames.append([smooth(a, b, (b - c) * 0.5, (d - a) * 0.5, t) for a, b, c, d in zip(i, j, ii, jj)])

    print('frames:', len(frames))
    return frames


def main(argv):
    frames = interpolate(argv[1])

    with open(argv[2], mode='w') as ff:

        for i in range(0, len(frames)):
            tx, ty, tz, rx, ry = frames[i]

            dx = math.sin(math.radians(rx))*math.cos(math.radians(ry))
            dz = -math.cos(math.radians(rx))*math.cos(math.radians(ry))
            dy = math.sin(math.radians(ry))

            eye = np.array([tx, ty, tz])
            center = np.array([tx + dx, ty + dy, tz + dz])
            up = np.array([0, -1, 0])

            f =  eye - center
            f = f / np.linalg.norm(f)

            side = np.cross(up, f)
            side = side / np.linalg.norm(side)

            up = np.cross(f, side)

            rmat = np.eye(4)

            rmat[0, 0:3] = side
            rmat[1, 0:3] = up
            rmat[2, 0:3] = f

            tmat = np.eye(4)
            tmat[0:3, 3] = -eye

            zz = np.eye(4)
            zz[0, 0] = -1
            zz[2, 2] = -1

            m = np.dot(np.linalg.inv(np.dot(rmat, tmat)), zz)

            ff.write('{0} {1} {2}\n'.format(i, i, i + 1))
            ff.write('{0} {1} {2} {3}\n'.format(m[0][0], m[0][1], m[0][2], m[0][3]))
            ff.write('{0} {1} {2} {3}\n'.format(m[1][0], m[1][1], m[1][2], m[1][3]))
            ff.write('{0} {1} {2} {3}\n'.format(m[2][0], m[2][1], m[2][2], m[2][3]))
            ff.write('{0} {1} {2} {3}\n'.format(m[3][0], m[3][1], m[3][2], m[3][3]))

if __name__ == "__main__":

    from sys import argv

    if (len(argv) < 3):
        print('Usage: {0} <input key poses> <output>'.format(argv[0]))
        exit(0)

    main( argv )
