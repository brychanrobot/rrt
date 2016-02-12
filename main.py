from cv2 import *
from numpy import *
# from scipy.spatial import KDTree
import kdtree
import time
from scipy.stats import *


class Rectangle:
	@staticmethod
	def create_from_points(top_left, bottom_right):
		return Rectangle(top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])

	def __init__(self, x, y, width, height):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.top_left = (x, y)
		self.bottom_right = (x + width, y + height)
		self.center = (x + width/2, y + height/2)
		self.area = width * height

	def contains(self, item):
		if type(item) is Rectangle:
			return self.contains_rect(item)

		if type(item) is tuple:
			return self.contains_point(item)

	def contains_point(self, p):
		return (self.top_left[0] < p[0] < self.bottom_right[0]) and (self.top_left[1] < p[1] < self.bottom_right[1])

	def contains_rect(self, rect):
		return self.contains_point(rect.top_left) and self.contains_point(rect.bottom_right)

	def __str__(self):
		return "x:%d, y:%d, br: %s" % (self.x, self.y, self.bottom_right)


def line_has_intersection(obstacle_hash, p1, p2):
	line_hash = zeros(obstacle_hash.shape, dtype=uint8)
	line(line_hash, p1, p2, 255, 2)
	intersection = bitwise_and(obstacle_hash, line_hash)
	return any(intersection)


def rect_has_intersection(obstacle_hash, rect):
	rect_hash = zeros(obstacle_hash.shape, dtype=uint8)
	# line(rect_hash, p1, p2, 255, 2)
	rectangle(rect_hash, rect.top_left, rect.bottom_right, 255, 20)
	rectangle(rect_hash, rect.top_left, rect.bottom_right, 255, FILLED)
	intersection = bitwise_and(obstacle_hash, rect_hash)

	"""
	imshow('rect_hash', rect_hash)
	imshow('obstacle_hash', obstacle_hash)
	imshow('intersection', intersection)
	waitKey(0)
	"""
	return any(intersection)


def random_point(bounding_rect):
	x = random.randint(0, bounding_rect.width - 1)
	y = random.randint(0, bounding_rect.height - 1)

	return (x, y)


def random_point_with_probability(probabilities, options):
	prob_ravel = probabilities.ravel()
	prob_ravel /= prob_ravel.sum()
	choice_indices = arange(options.shape[0])
	choice = random.choice(choice_indices, p=prob_ravel)
	return options[choice]


def is_in_obstacles(obstacles, rect):
	for o in obstacles:
		if o.contains(rect) or rect.contains(o):
			return True
	return False


def gaussian(size=50, mean=array([0.0, 0.0]), sigma=array([.25, .25])):
	"""Returns a 2D Gaussian kernel array."""

	x, y = mgrid[-1.0:1.0:size * 1j, -1.0:1.0:size * 1j]
	# Need an (N, 2) array of (x, y) pairs.
	xy = column_stack([x.flat, y.flat])

	#mu = np.array([0.0, 0.0])

	#sigma = np.array([.025, .025])
	covariance = diag(sigma**2)

	z = multivariate_normal.pdf(xy, mean=mean, cov=covariance)

	# Reshape back to a (30, 30) grid.
	z = z.reshape(x.shape) / (z.max() * 6)

	return z


def main():
	map_rect = Rectangle(0, 0, 1200, 1080)

	options = zeros((map_rect.height, map_rect.width, 2), dtype=uint32)

	row = arange(0, options.shape[1])
	col = arange(0, options.shape[0])
	options[:, :, 0] = row
	for c in row:
		options[:, c, 1] = col

	options = options.reshape((options.shape[0] * options.shape[1], 2))

	#options = options.ravel()
	# print(options)
	"""
	for r in range(options.shape[0]):
		for c in range(options.shape[1]):
			options[r, c] = [r, c]
	"""
	# map_rect = Rectangle(0, 0, 300, 300)
	map = zeros((map_rect.height, map_rect.width, 3), dtype=uint8)

	start = [50, 50]
	end = [750, 750]
	max_segment = 20

	"""
	obstacles = [
		Rectangle(700, 500, 800, 100),
		Rectangle(100, 500, 400, 100)
	]
	"""
	num_obstacles = 10
	obstacles = []

	obstacle_hash = zeros(map.shape[:2], dtype=uint8)

	for i in range(num_obstacles):
		while True:
			top_left = random_point(map_rect)
			bottom_right = random_point(map_rect)
			obstacle = Rectangle.create_from_points(top_left, bottom_right)

			if not rect_has_intersection(obstacle_hash, obstacle):
				obstacles.append(obstacle)
				break

		rectangle(obstacle_hash, obstacle.top_left, obstacle.bottom_right, 255, 20)
		rectangle(obstacle_hash, obstacle.top_left, obstacle.bottom_right, 255, FILLED)
		rectangle(map, obstacle.top_left, obstacle.bottom_right, [255, 100, 0], 2)

	gauss = gaussian(size=map_rect.area / 10000)
	g_offset = gauss.shape[0] / 2 + max_segment
	unsearched_area = ones((map.shape[0] + 2 * g_offset, map.shape[1] + 2 * g_offset), dtype=float64)
	unsearched_area[g_offset:g_offset + map.shape[0], g_offset:g_offset + map.shape[1]] -= obstacle_hash / 255

	# imshow('obstacles', obstacle_hash)

	tree = kdtree.create([start])

	for i in range(100000):
		x, y = random_point(map_rect)
		unsearched_probabilities = unsearched_area[g_offset:g_offset + map.shape[0], g_offset:g_offset + map.shape[1]]
		x, y = random_point_with_probability(unsearched_probabilities, options)


		# circle(map, (x, y), 4, [255, 0, 255], 1)

		nn, dist = tree.search_nn((x, y))
		xnn, ynn = nn.data

		if dist > max_segment:
			angle = math.atan2((xnn - x), (y - ynn)) + pi / 2.0
			# print(angle * 180 / math.pi)
			x = int(max_segment * cos(angle) + xnn)
			y = int(max_segment * sin(angle) + ynn)

		new_point = (x, y)
		nn_point = (xnn, ynn)

		if not line_has_intersection(obstacle_hash, nn_point, new_point):
			tree.add(new_point)
			line(map, nn_point, new_point, [100, 0, 255], 2)
			rect = Rectangle(new_point[0] - gauss.shape[1] / 2, new_point[1] - gauss.shape[0] / 2, gauss.shape[1], gauss.shape[0])
			#print(rect)
			unsearched_area[rect.y + g_offset:rect.bottom_right[1] + g_offset, rect.x + g_offset:rect.bottom_right[0] + g_offset] -= gauss
			unsearched_area = unsearched_area.clip(min=0)
			#imshow('unsearched', unsearched_area)# /unsearched_area.max())

		imshow('map', map)
		if i % 10 == 0:
			waitKey(1)
		# tree.rebalance()

		if i % 100 == 0:
			tree.rebalance()


main()
