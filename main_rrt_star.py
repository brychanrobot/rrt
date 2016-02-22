from cv2 import *
from numpy import *
# from scipy.spatial import KDTree
import kdtree
import time
from scipy.stats import *
import glob
import datetime
from screeninfo import get_monitors


class Node:
	node_count = 0
	def __init__(self, parent, location, cumulative_cost):
		self.parent = parent
		self.location = location
		self.children = []
		self.cumulative_cost = cumulative_cost
		self.node_num = Node.node_count
		Node.node_count += 1

	def addChild(self, location, cumulative_cost):
		child = Node(self, location, cumulative_cost)
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

	def contains_point(self, p, padding=0):
		print("p: %s, tl: %s, br: %s" % (p, self.top_left, self.bottom_right))
		return (self.top_left[0] - padding < p[0] < self.bottom_right[0] + padding) and (self.top_left[1] - padding < p[1] < self.bottom_right[1] + padding)

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
	x = random.randint(bounding_rect.x, bounding_rect.bottom_right[0])
	y = random.randint(bounding_rect.y, bounding_rect.bottom_right[1])

	return x, y


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
	circle(map, node.location, 2, [100, 0, 255], 1)
	for child in node.children:
		line(map, node.location, child.location, [0, 255, 255], 1)
		draw_all_lines(map, child)

	#putText(map, "%d" % node.cumulative_cost, node.location, FONT_HERSHEY_SIMPLEX, .25, (255, 255, 255))

	return map


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
	while(True):
		namedWindow('map', WND_PROP_FULLSCREEN)
		setWindowProperty('map', WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN)

		monitor = get_monitors()[0]
		map_rect = Rectangle(0, 0, monitor.width, monitor.height)
		#map_rect = Rectangle(0, 0, 800, 800)

		options = zeros((map_rect.height, map_rect.width, 2), dtype=uint32)

		row = arange(0, options.shape[1])
		col = arange(0, options.shape[0])
		options[:, :, 0] = row
		for c in row:
			options[:, c, 1] = col

		options = options.reshape((options.shape[0] * options.shape[1], 2))


		#start = (100, 100)
		#end = (750, 750)
		start = random_point(map_rect)
		end = random_point(map_rect)
		max_segment = 20

		map = zeros((map_rect.height, map_rect.width, 3), dtype=uint8)
		circle(map, start, 5, (10, 255, 10), 3)
		circle(map, end, 5, (100, 100, 255), 3)

		root = Node(None, start, 0)
		node_hash = {start:root}

		num_obstacles = 12
		obstacles = []

		obstacle_hash = zeros(map.shape[:2], dtype=uint8)

		for i in range(num_obstacles):
			while True:
				top_left = random_point(map_rect)
				bottom_right = random_point(Rectangle.create_from_points(top_left, map_rect.bottom_right))
				obstacle = Rectangle.create_from_points(top_left, bottom_right)

				if not rect_has_intersection(obstacle_hash, obstacle) and not obstacle.contains_point(start, 20) and not obstacle.contains_point(end, 20):
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

		video_filename = "videos/vid%s.avi" % datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
		video = VideoWriter(video_filename, VideoWriter_fourcc(*'IYUV'), 60, (map.shape[1], map.shape[0]))

		for i in range(3000):
			print("\r%4d" % i, end="")
			x, y = random_point(map_rect)
			#unsearched_probabilities = unsearched_area[g_offset:g_offset + map.shape[0], g_offset:g_offset + map.shape[1]]
			#x, y = random_point_with_probability(unsearched_probabilities, options)


			# circle(map, (x, y), 4, [255, 0, 255], 1)

			nn, dist = tree.search_nn((x, y))
			dist = math.sqrt(dist)
			xnn, ynn = nn.data

			if dist > max_segment:
				angle = math.atan2((xnn - x), (y - ynn)) + pi / 2.0
				# print(angle * 180 / math.pi)
				x = int(max_segment * cos(angle) + xnn)
				y = int(max_segment * sin(angle) + ynn)

			new_point = (x, y)
			#nn_point = (xnn, ynn)

			neighbors = tree.search_knn(new_point, 50)
			cumulative_costs = []
			for neighbor, distance in neighbors:
				#print("1: %s, 2: %s"%(new_point, neighbor.data))
				#square_dist = ((asarray(new_point) - asarray(neighbor.data)) ** 2).sum()
				#man_dist = math.sqrt(((asarray(new_point) - asarray(neighbor.data)) ** 2).sum())
				#print("d1: %d, d2: %d, d3: %d" % (distance, square_dist, man_dist))
				distance = math.sqrt(distance)
				distance = distance if distance < max_segment * 3 else 10000
				cumulative_costs.append(node_hash[neighbor.data].cumulative_cost + distance)

			best_neighbor_index = argmin(cumulative_costs)
			best_neighbor = node_hash[neighbors[best_neighbor_index][0].data]
			best_cumulative_cost = cumulative_costs[best_neighbor_index]

			neighbors.remove(neighbors[best_neighbor_index])

			if not line_has_intersection(obstacle_hash, best_neighbor.location, new_point):
				tree.add(new_point)
				#line(map, nn_point, new_point, [100, 0, 255], 2)

				#parent = node_hash[nn_point]
				new_node = best_neighbor.addChild(new_point, best_cumulative_cost)
				node_hash[new_point] = new_node


				for neighbor, distance in neighbors:
					neighbor_node = node_hash[neighbor.data]
					distance = math.sqrt(distance)
					if distance < max_segment * 3 and (new_node.cumulative_cost + distance) < neighbor_node.cumulative_cost:
						if not line_has_intersection(obstacle_hash, neighbor_node.location, new_node.location):
							rewire(new_node, neighbor_node, distance)
							#print("rewired")





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
				putText(map, "Iteration: %d" % i, (20, 20), FONT_HERSHEY_SIMPLEX, .7, (50, 255, 50))
				imshow('map', map)
				video.write(map)
				waitKey(1)
			# tree.rebalance()

			if i % 100 == 0:
				tree.rebalance()

		map = draw_all_lines(just_obstacles.copy(), root)

		neighbors = tree.search_knn(end, 30)
		closest_node = None
		best_cost = inf
		for neighbor, dist in neighbors:
			dist = math.sqrt(dist)
			node = node_hash[neighbor.data]
			if node.cumulative_cost + dist < best_cost:
				closest_node = node
				best_cost = node.cumulative_cost + dist

		current_node = closest_node.addChild(end, 0)

		while not current_node.parent is None:
			line(map, current_node.location, current_node.parent.location, (255, 255, 0), 2)
			current_node = current_node.parent
			imshow('map', map)
			video.write(map)
			waitKey(50)

		#imshow('map', map)
		#waitKey(0)

		video.write(map)

		video.release()

		time.sleep(2)


main()
