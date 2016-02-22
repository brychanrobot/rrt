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

def is_in_obstacles(obstacles, rect):
	for o in obstacles:
		if o.contains(rect) or rect.contains(o):
			return True
	return False


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
	node_list = [root]

	num_obstacles = 5
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

	rectangle(obstacle_hash, map_rect.top_left, map_rect.bottom_right, 255, 4)

	just_obstacles = map.copy()

	for i in range(100000):

		print("\riteration: %4d" % i, end="")

		seed_node = node_list[random.randint(maximum(0, len(node_list) - 50), len(node_list))]

		if seed_node.parent is None:
			angle_to_end = math.atan2((end[0] - seed_node.location[0]), (seed_node.location[1] - end[1])) - pi / 2.0
		else:
			parent_point = seed_node.parent.location;
			angle_to_end = math.atan2((seed_node.location[0] - parent_point[0]), (parent_point[1] - seed_node.location[1])) - pi / 2.0

		rand_angle = angle_to_end + random.uniform(-math.pi/1.5, math.pi/1.5)

		x = int(max_segment * cos(rand_angle) + seed_node.location[0])
		y = int(max_segment * sin(rand_angle) + seed_node.location[1])

		new_point = (x, y)

		if not line_has_intersection(obstacle_hash, seed_node.location, new_point):
			new_node = seed_node.addChild(new_point)
			node_list.append(new_node)

			distance_to_end = math.sqrt(square(asarray(end) - asarray(new_point)).sum())
			#print("distance: %d" % distance_to_end)
			#distance_to_end = math.sqrt((ndarray(end) - ndarray(new_point)) ** 2).sum())
			#print("distance: %d" % distance_to_end)

			if distance_to_end < 50:
				end_node = Node(new_node, end)
				new_node.children.append(end_node)
				break

		if i % 10 == 0:
			map = draw_all_lines(just_obstacles.copy(), root)
			imshow('map', map)
			waitKey(1)

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
