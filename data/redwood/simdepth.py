import math
import numpy as np
import glob
import ntpath
from PIL import Image
from sys import argv
from multiprocessing import Pool

class simulator:

	def loaddistmodel(self, fname):

		data = np.loadtxt(fname, comments='%', skiprows = 5)
		dist = np.empty([80, 80, 5])

		for y in range(0, 80):
			for x in range(0, 80):
				idx = (y * 80 + x) * 23 + 3
				if (data[idx:idx + 5] < 8000).all():
					dist[y, x, :] = 0
				else:
					dist[y, x, :] = data[idx + 15 : idx + 20]

		self.model = dist

	def undistort(self, x, y, z):

		i2 = int((z + 1) / 2)
		i1 = i2 - 1
		a = (z - (i1 * 2 + 1)) / 2
		x = int(x / 8)
		y = int(y / 6)
		f = (1 - a) * self.model[y, x, min(max(i1, 0), 4)] + a * self.model[y, x, min(i2, 4)]

		if f == 0:
			return 0
		else:
			return z / f

	def simulate(self, inputpng, outputpng):

		a = np.array(Image.open(inputpng)).astype(np.float32) / 1000.0
		b = np.copy(a)
		it = np.nditer(a, flags=['multi_index'], op_flags=['writeonly'])

		while not it.finished:

			# pixel shuffle
			x = min(max(round(it.multi_index[1] + np.random.normal(0, 0.25)), 0), 639)
			y = min(max(round(it.multi_index[0] + np.random.normal(0, 0.25)), 0), 479)

			# downsample
			d = b[y - y % 2, x - x % 2]

			# distortion
			d = self.undistort(x, y, d)

			# quantization and high freq noise
			if d == 0:
				it[0] = 0
			else:
				it[0] = 35.130 * 8 / round((35.130 / d + np.random.normal(0, 0.027778)) * 8)

			it.iternext()

		Image.fromarray((a * 1000).astype(np.int32)).save(outputpng)


if __name__ == "__main__":

	if (len(argv) < 4):
		print('Usage: {0} <input png dir> <output png dir> <distortion model>'.format(argv[0]))
		exit(0)

	s = simulator()
	s.loaddistmodel(argv[3])

	ifiles = glob.glob(argv[1] + r'\*.png')
	ofiles = [argv[2] + '\\' + ntpath.basename(f) for f in ifiles]
	param = zip(ifiles, ofiles)

	p = Pool(8)
	p.starmap(s.simulate, param)
