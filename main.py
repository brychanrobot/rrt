from cv2 import *
from numpy import *
# from scipy.spatial import KDTree
import kdtree
import time
from scipy.stats import *


class Node:
	def __init__(self, parent, location):
		self.parent = parent
		self.location = location
		self.children = []

	def addChild(self, location):
		child = Node(self, location)
		self.children.append(child)
		return child


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
		return (self.top_left[0] < p[0] < self.bottom_right[0]) and (self.top_left[1] > p[1] > self.bottom_right[1])

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
	#choice = random.multinomial(1, prob_ravel, 1)
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


def draw_all_lines(map, node):
	for child in node.children:
		line(map, node.location, child.location, [100, 0, 255], 2)
		draw_all_lines(map, child)

	return map


def main():
	map_rect = Rectangle(0, 0, 1000, 1000)

	options = zeros((map_rect.height, map_rect.width, 2), dtype=uint32)

	row = arange(0, options.shape[1])
	col = arange(0, options.shape[0])
	options[:, :, 0] = row
	for c in row:
		options[:, c, 1] = col

	options = options.reshape((options.shape[0] * options.shape[1], 2))


	start = (50, 50)
	end = (750, 750)
	max_segment = 10

	map = zeros((map_rect.height, map_rect.width, 3), dtype=uint8)
	circle(map, start, 5, (255, 255, 0), 3)
	circle(map, end, 5, (255, 255, 0), 3)

	root = Node(None, start)
	node_hash = {start:root}

	num_obstacles = 20
	obstacles = []

	obstacle_hash = zeros(map.shape[:2], dtype=uint8)

	for i in range(num_obstacles):
		while True:
			top_left = random_point(map_rect)
			bottom_right = random_point(map_rect)
			obstacle = Rectangle.create_from_points(top_left, bottom_right)

			if not rect_has_intersection(obstacle_hash, obstacle) and not obstacle.contains_point(start) and not obstacle.contains_point(end):
				obstacles.append(obstacle)
				break

		rectangle(obstacle_hash, obstacle.top_left, obstacle.bottom_right, 255, 20)
		rectangle(obstacle_hash, obstacle.top_left, obstacle.bottom_right, 255, FILLED)
		rectangle(map, obstacle.top_left, obstacle.bottom_right, [255, 100, 0], 2)

	just_obstacles = map.copy()

	gauss = gaussian(size=map_rect.area / 10000)
	g_offset = gauss.shape[0] / 2 + max_segment
	unsearched_area = ones((map.shape[0] + 2 * g_offset, map.shape[1] + 2 * g_offset), dtype=float64)
	unsearched_area[g_offset:g_offset + map.shape[0], g_offset:g_offset + map.shape[1]] -= obstacle_hash / 255

	# imshow('obstacles', obstacle_hash)

	tree = kdtree.create([start])

	for i in range(100000):
		#x, y = random_point(map_rect)
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
			#line(map, nn_point, new_point, [100, 0, 255], 2)

			parent = node_hash[nn_point]
			node_hash[new_point] = parent.addChild(new_point)

			distance_to_end = math.sqrt(square(asarray(end) - asarray(new_point)).sum())
			#print("distance: %d" % distance_to_end)
			#distance_to_end = math.sqrt((ndarray(end) - ndarray(new_point)) ** 2).sum())
			#print("distance: %d" % distance_to_end)

			if distance_to_end < 50:
				end_node = Node(node_hash[new_point], end)
				node_hash[new_point].children.append(end_node)
				break

			rect = Rectangle(new_point[0] - gauss.shape[1] / 2, new_point[1] - gauss.shape[0] / 2, gauss.shape[1], gauss.shape[0])
			unsearched_area[rect.y + g_offset:rect.bottom_right[1] + g_offset, rect.x + g_offset:rect.bottom_right[0] + g_offset] -= gauss
			unsearched_area = unsearched_area.clip(min=0)
			#imshow('unsearched', 1-unsearched_area)# /unsearched_area.max())

		if i % 10 == 0:
			map = draw_all_lines(just_obstacles.copy(), root)
			imshow('map', map)
			waitKey(1)
		# tree.rebalance()

		if i % 100 == 0:
			tree.rebalance()

	map = draw_all_lines(just_obstacles.copy(), root)
	current_node = end_node
	while not current_node.parent is None:
		line(map, current_node.location, current_node.parent.location, (255, 255, 0), 2)
		current_node = current_node.parent
		imshow('map', map)
		waitKey(50)

	#imshow('map', map)
	waitKey(0)


main()
