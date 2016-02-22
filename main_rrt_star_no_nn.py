from cv2 import *
from numpy import *
import time
import datetime

created_nodes = 0


class Node:
	node_count = 0
	def __init__(self, parent, location, cumulative_cost):
		self.parent = parent
		self.location = location
		self.children = []
		self.cumulative_cost = cumulative_cost
		self.node_num = Node.node_count
		Node.node_count += 1
		print("nc: %d" % Node.node_count)

	def addChild(self, location, cumulative_cost):
		child = Node(self, location, cumulative_cost)
		self.children.append(child)
		return child

	def __str__(self):
		return "%d: %s" % (self.node_num, self.location)


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
	rectangle(rect_hash, rect.top_left, rect.bottom_right, 255, 20 * 2)
	rectangle(rect_hash, rect.top_left, rect.bottom_right, 255, FILLED)
	intersection = bitwise_and(obstacle_hash, rect_hash)
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
	circle(map, node.location, 2, [100, 0, 255], 1)
	for child in node.children:
		line(map, node.location, child.location, [50, 255, 50], 1)
		draw_all_lines(map, child)

	#putText(map, "%d" % node.cumulative_cost, node.location, FONT_HERSHEY_SIMPLEX, .25, (255, 255, 255))

	return map


def get_circle_indices(center, radius, clip_shape):
	map = zeros(clip_shape, dtype=uint8)
	circle(map, center, radius, 255, FILLED)
	return where(map == 255)

def update_cumulative_cost(node, new_cumulative_cost):
	old_cumulative_cost = node.cumulative_cost
	node.cumulative_cost = new_cumulative_cost
	for child in node.children:
		cost_to_parent = child.cumulative_cost - old_cumulative_cost
		update_cumulative_cost(child, new_cumulative_cost + cost_to_parent)


def rewire(new_parent, child, cost_to_parent):
	child.parent.children.remove(child)
	new_parent.children.append(child)
	child.parent = new_parent
	update_cumulative_cost(child, new_parent.cumulative_cost + cost_to_parent)


def main():
	map_rect = Rectangle(0, 0, 200, 200)
	namedWindow('map')
	moveWindow('map', 20, 20)


	start = (100, 100)
	end = (150, 150)
	max_segment = 20

	map = zeros((map_rect.height, map_rect.width, 3), dtype=uint8)
	circle(map, start, 5, (255, 255, 0), 3)
	circle(map, end, 5, (255, 255, 0), 3)

	root = Node(None, start, 0)
	node_list = [root]
	neighbor_lists = []
	for row in range(map_rect.height):
		neighbor_lists.append([])
		for col in range(map_rect.width):
			neighbor_lists[row].append([])

	neighbor_locations = get_circle_indices(root.location, max_segment * 3, map.shape)
	for y, x in zip(neighbor_locations[0], neighbor_locations[1]):
		neighbor_lists[y][x].append(root)

	num_obstacles = 0
	obstacles = []

	obstacle_hash = zeros(map.shape[:2], dtype=uint8)

	rectangle(obstacle_hash, map_rect.top_left, map_rect.bottom_right, 255, 4)

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

	video_filename = "no_nn_videos/vid%s.avi" % datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
	video = VideoWriter(video_filename, VideoWriter_fourcc(*'IYUV'), 10, (map.shape[1], map.shape[0]))

	for i in range(5000):
		print("\riteration: %4d" % i)

		seed_node = node_list[random.randint(0, len(node_list))]
		rand_angle = random.uniform(0, 2 * math.pi)

		print(rand_angle * 180 / math.pi)
		x = int(max_segment * cos(rand_angle) + seed_node.location[0])
		y = int(max_segment * sin(rand_angle) + seed_node.location[1])

		new_point = (x, y)
		#nn_point = (xnn, ynn)

		print(seed_node)

		neighbors = neighbor_lists[y][x]
		best_cumulative_cost = inf
		best_neighbor = None
		#print(neighbors)
		for neighbor in neighbors:
			distance = math.sqrt(((asarray(new_point) - asarray(neighbor.location)) ** 2).sum())
			cumulative_cost = neighbor.cumulative_cost + distance
			if cumulative_cost < best_cumulative_cost:
				best_neighbor = neighbor

		#neighbors.remove(neighbors[best_neighbor_index])
		if not best_neighbor is None and not line_has_intersection(obstacle_hash, best_neighbor.location, new_point):
			new_node = best_neighbor.addChild(new_point, best_cumulative_cost)
			node_list.append(new_node)

			neighbor_locations = get_circle_indices(new_node.location, max_segment * 3, map.shape)
			for y, x in zip(neighbor_locations[0], neighbor_locations[1]):
				neighbor_lists[y][x].append(new_node)
			"""
			for neighbor, distance in neighbors:
				neighbor_node = node_hash[neighbor.data]
				distance = math.sqrt(distance)
				if distance < max_segment * 3 and (new_node.cumulative_cost + distance) < neighbor_node.cumulative_cost:
					if not line_has_intersection(obstacle_hash, neighbor_node.location, new_node.location):
						rewire(new_node, neighbor_node, distance)
						#print("rewired")
			"""




			distance_to_end = math.sqrt(square(asarray(end) - asarray(new_point)).sum())

			"""
			if distance_to_end < 50:
				#end_node = Node(node_hash[new_point], end)
				end_node = node_hash[new_point].addChild(end, node_hash[new_point].cumulative_cost + distance_to_end)
				break
			"""

			#rect = Rectangle(new_point[0] - gauss.shape[1] / 2, new_point[1] - gauss.shape[0] / 2, gauss.shape[1], gauss.shape[0])
			#unsearched_area[rect.y + g_offset:rect.bottom_right[1] + g_offset, rect.x + g_offset:rect.bottom_right[0] + g_offset] -= gauss
			#unsearched_area = unsearched_area.clip(min=0)
			#imshow('unsearched', 1-unsearched_area)# /unsearched_area.max())

		if i % 1 == 0:
			map = draw_all_lines(just_obstacles.copy(), root)
			imshow('map', map)
			#video.write(map)
			waitKey(1)
		# tree.rebalance()

	map = draw_all_lines(just_obstacles.copy(), root)

	neighbors = neighbor_lists[end[1]][end[0]]
	closest_node = None
	best_cost = inf
	for neighbor in neighbors:
		distance = math.sqrt(((asarray(end) - asarray(neighbor.location)) ** 2).sum())
		if neighbor.cumulative_cost + distance < best_cost:
			closest_node = neighbor
			best_cost = neighbor.cumulative_cost + distance

	current_node = closest_node.addChild(end, 0)

	while not current_node.parent is None:
		line(map, current_node.location, current_node.parent.location, (255, 255, 0), 2)
		current_node = current_node.parent
		imshow('map', map)
		video.write(map)
		waitKey(50)

	#imshow('map', map)
	waitKey(0)

	video.write(map)

	video.release()


main()
