# Uncomment this line to import some functions that can help
# you debug your algorithm
from plotting import draw_line, draw_hull, circle_point
#import math

class Node:
    def __init__(self, point):
        self.point = point
        self.next = None
        self.prev = None


class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, point):
        node = Node(point)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            node.prev = self.tail
            self.tail = node

    def __iter__(self):
        current = self.head
        while current:
            yield current.point
            current = current.next

def compute_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the subset of provided points that define the convex hull"""
    #using the divide and conquer approach, we will first use linkedlis to store the points, then we will sort the points by x-coordinate, then divide the points into two halves. We will recursively compute the convex hull for each half and merge the two hulls. using finding the upper and lower tangents, we will merge the two hulls.
    
   
    hull = Hull(points)
    return hull.hull


class Hull:
    def __init__(self, points):
        # Initialize the hull with the points provided and sort them by x-coordinate
        self.points = LinkedList()
        for point in points:
            # Append the point to the LinkedList of points in the hull
            self.points.append(point)
        if len(list(self.points)) <= 1:
            self.hull = list(self.points)
        else:
            # Sort the points by x-coordinate and store them in a list for easier access
            self.points_list = list(self.points)
            self.points_list.sort()
            self.hull = self.divide_and_conquer(self.points_list)


    def divide_and_conquer(self, points):
        if not points:  # time complexity = O(1) #space complexity = O(1)
            return []  # time complexity = O(1)  #space complexity = O(1)
        if len(points) <= 2: # time complexity = O(1) #space complexity = O(1)
            # If there are only 3 points, sort them in counter-clockwise order
            if len(points) == 3 and not self.is_counter_clockwise(points[0], points[1], points[2]):
                points[1], points[2] = points[2], points[1] # time complexity = O(1) #space complexity = O(1)
            return points # time complexity = O(1) #space complexity = O(1)
        mid = len(points) // 2 # time complexity = O(1) #space complexity = O(1)
        left = self.divide_and_conquer(points[:mid]) # time complexity = O(n/2) #space complexity = O(n/2)
        right = self.divide_and_conquer(points[mid:]) # time complexity = O(n/2) #space complexity = O(n/2)
        return self.merge_hulls(left, right) # time complexity = O(n) #space complexity = O(n)
    

    def is_counter_clockwise(self, point1, point2, point3):
        return (point2[1] - point1[1]) * (point3[0] - point2[0]) > (point2[0] - point1[0]) * (point3[1] - point2[1]) # time complexity = O(1) #space complexity = O(1). this is because 



    def is_clockwise(self, point1, point2, point3):
        return not self.is_counter_clockwise(point1, point2, point3)

    def find_upper_tangent(self, Left, Right):
        intial_point_1 = max(Left, key=lambda x: x[0])
        intial_point_2 = min(Right, key=lambda x: x[0])
        temp = (intial_point_1, intial_point_2)
        done = False
        while not done:
            done = True
            # Find the clockwise neighbor of intial_point_1 in L
            point_indices = {point: i for i, point in enumerate(Left)}
            r = Left[(point_indices[intial_point_1] + 1) % len(Left)]
            while not self.is_lower_tangent(temp, Left):
                temp = (r, intial_point_2)  
                intial_point_1 = r
                r = Left[(point_indices[intial_point_1] + 1) % len(Left)]
                done = False

            # Find the counterclockwise neighbor of intial_point_2 in R
            point_indices = {point: i for i, point in enumerate(Right)}
            r = Right[(point_indices[intial_point_2] - 1) % len(Right)]
            while not self.is_lower_tangent(temp, Right):
                temp = (intial_point_1, r)  
                intial_point_2 = r
                r = Right[(point_indices[intial_point_2] - 1) % len(Right)]
                done = False

        return temp

    def find_lower_tangent(self, Left, Right):
        intial_point_1 = max(Left, key=lambda x: x[0]) 
        intial_point_2 = min(Right, key=lambda x: x[0])
        temp = (intial_point_1, intial_point_2)
        done = False
        while not done:
            done = True
            # Find the counterclockwise neighbor of intial_point_1 in L
            point_indices = {point: i for i, point in enumerate(Left)}
            r = Left[(point_indices[intial_point_1] - 1) % len(Left)]
            while not self.is_upper_tangent(temp, Left):
                temp = (r, intial_point_2)  
                intial_point_1 = r
                r = Left[(point_indices[intial_point_1] - 1) % len(Left)]
                done = False

            # Find the clockwise neighbor of intial_point_2 in R
            point_indices = {point: i for i, point in enumerate(Right)}
            r = Right[(point_indices[intial_point_2] + 1) % len(Right)]
            while not self.is_upper_tangent(temp, Right):
                temp = (intial_point_1, r)  
                intial_point_2 = r
                r = Right[(point_indices[intial_point_2] + 1) % len(Right)]
                done = False

        return temp
    


    def is_upper_tangent(self, line, points):
        point1, point2 = line 
        for p in points:
            if p != point1 and p != point2 and self.is_counter_clockwise(point1, point2, p):
                return False
        return True
    
    def is_lower_tangent(self, line, points):
        point1, point2 = line # time complexity = O(1) #space complexity = O(1)
        for p in points: # time complexity = O(n) #space complexity = O(1)
            if p != point1 and p != point2 and self.is_clockwise(point1, point2, p): # time complexity = O(1) #space complexity = O(1)
                return False
        return True

    def merge_hulls(self, left, right):
        upper_tangent = self.find_upper_tangent(left, right) #time complexity = O(n^2) #space complexity = O(1)
        lower_tangent = self.find_lower_tangent(left, right) #time complexity = O(n^2) #space complexity = O(1)
        
        merged_hull = [] #time complexity = O(1) #space complexity = O(1)
        
        # Add points from left starting from upper_tangent[0] to lower_tangent[0]
        point_indices = {point: i for i, point in enumerate(left)} #time complexity = O(n) #space complexity = O(n)
        slope = point_indices[upper_tangent[0]] #time complexity = O(1) #space complexity = O(1)
        while True:
            merged_hull.append(left[slope]) #time complexity = O(1) #space complexity = O(1)
            if left[slope] == lower_tangent[0]:
                break 
            slope = (slope + 1) % len(left) #time complexity = O(1) #space complexity = O(1)
        
        # Add points from right starting from lower_tangent[1] to upper_tangent[1]
        point_indices = {point: i for i, point in enumerate(right)} #time complexity = O(n) #space complexity = O(n)
        slope = point_indices[lower_tangent[1]] #time complexity = O(1) #space complexity = O(1)
        while True:
            merged_hull.append(right[slope]) #time complexity = O(1) #space complexity = O(1)
            if right[slope] == upper_tangent[1]: 
                break
            slope = (slope + 1) % len(right) #time complexity = O(1) #space complexity = O(1)
        
        return merged_hull #time complexity = O(1) #space complexity = O(1)


