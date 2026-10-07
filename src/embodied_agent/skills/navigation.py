"""Known-map A* with conservative circular robot/carrying footprints.

This planner uses declared geometry, not SLAM or perception. Distances are
metres. Every returned segment is checked, including the exact endpoints.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable


class NavigationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Obstacle:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    source: str

    def distance(self, point: tuple[float, float]) -> float:
        x, y = point
        return math.hypot(max(self.xmin - x, 0, x - self.xmax),
                          max(self.ymin - y, 0, y - self.ymax))


class Navigator:
    def __init__(self, bounds: tuple[float, float, float, float],
                 obstacles: Iterable[Obstacle], radius: float, resolution: float = 0.10):
        if not all(math.isfinite(v) for v in (*bounds, radius, resolution)):
            raise NavigationError("INVALID_PATH", "Navigation geometry must be finite")
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1] or radius <= 0 or resolution <= 0:
            raise NavigationError("INVALID_PATH", "Invalid room bounds, footprint or grid")
        self.bounds, self.obstacles = bounds, tuple(obstacles)
        for obstacle in self.obstacles:
            values = (obstacle.xmin, obstacle.ymin, obstacle.xmax, obstacle.ymax)
            if not all(math.isfinite(v) for v in values) or values[2] < values[0] or values[3] < values[1]:
                raise NavigationError("INVALID_PATH", "Invalid obstacle geometry")
        self.radius, self.resolution = radius, resolution

    def free(self, point: tuple[float, float]) -> bool:
        x, y = point
        xmin, ymin, xmax, ymax = self.bounds
        return (all(math.isfinite(v) for v in point)
                and xmin + self.radius <= x <= xmax - self.radius
                and ymin + self.radius <= y <= ymax - self.radius
                and all(item.distance(point) > self.radius for item in self.obstacles))

    def segment_free(self, start: tuple[float, float], end: tuple[float, float]) -> bool:
        # Sampling alone can miss a small obstacle between samples. Adding half
        # the sample spacing to the footprint makes the check conservative.
        length = math.dist(start, end)
        if not math.isfinite(length):
            return False
        count = max(1, math.ceil(length / (self.resolution / 4)))
        margin = length / count / 2
        xmin, ymin, xmax, ymax = self.bounds
        for i in range(count + 1):
            point = (start[0] + (end[0] - start[0]) * i / count,
                     start[1] + (end[1] - start[1]) * i / count)
            x, y = point
            if not (xmin + self.radius <= x <= xmax - self.radius
                    and ymin + self.radius <= y <= ymax - self.radius):
                return False
            if any(item.distance(point) <= self.radius + margin for item in self.obstacles):
                return False
        return True

    def remaining_path(self, current: tuple[float, float], path: Iterable[Iterable[float]]) -> list[tuple[float, float]]:
        """Trim travelled segments using the measured pose, never reset the robot."""
        points = [tuple(point)[:2] for point in path]
        if not points:
            raise NavigationError("INVALID_PATH", "No route remains to supervise")
        if len(points) == 1:
            return [tuple(current), points[0]]
        def distance(index):
            start, end = points[index], points[index + 1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length_squared = dx * dx + dy * dy
            fraction = 0.0 if not length_squared else max(0.0, min(1.0,
                ((current[0] - start[0]) * dx + (current[1] - start[1]) * dy) / length_squared))
            return math.dist(current, (start[0] + fraction * dx, start[1] + fraction * dy))
        index = min(range(len(points) - 1), key=distance)
        return [tuple(current), *points[index + 1:]]

    def route_free(self, current: tuple[float, float], path: Iterable[Iterable[float]]) -> bool:
        remaining = self.remaining_path(current, path)
        return all(self.segment_free(start, end) for start, end in zip(remaining, remaining[1:]))

    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
        start, goal = tuple(start), tuple(goal)
        if not self.free(start) or not self.free(goal):
            raise NavigationError("PATH_BLOCKED", "Robot footprint intersects an obstacle or room boundary at an endpoint")
        if self.segment_free(start, goal):
            return [start, goal]
        xmin, ymin, xmax, ymax = self.bounds
        nx = math.floor((xmax - xmin) / self.resolution) + 1
        ny = math.floor((ymax - ymin) / self.resolution) + 1
        if nx * ny > 250000:
            raise NavigationError("INVALID_PATH", "Navigation grid exceeds the declared planning budget")

        def point(node):
            return (xmin + node[0] * self.resolution, ymin + node[1] * self.resolution)

        def attach(position):
            near = (round((position[0] - xmin) / self.resolution),
                    round((position[1] - ymin) / self.resolution))
            candidates = [(math.dist(position, point((i, j))), (i, j))
                          for i in range(max(0, near[0] - 2), min(nx, near[0] + 3))
                          for j in range(max(0, near[1] - 2), min(ny, near[1] + 3))
                          if self.free(point((i, j))) and self.segment_free(position, point((i, j)))]
            if not candidates:
                raise NavigationError("PATH_BLOCKED", "Exact endpoint cannot safely connect to the navigation grid")
            return min(candidates)[1]

        origin, target = attach(start), attach(goal)
        frontier = [(math.dist(point(origin), point(target)), 0.0, origin)]
        costs, parents = {origin: 0.0}, {}
        while frontier:
            _, cost, node = heapq.heappop(frontier)
            if cost != costs[node]:
                continue
            if node == target:
                nodes = [node]
                while node in parents:
                    node = parents[node]
                    nodes.append(node)
                path = [start, *(point(n) for n in reversed(nodes)), goal]
                # Greedy line-of-sight compression preserves checked segments.
                simplified, index = [start], 0
                while index < len(path) - 1:
                    next_index = index + 1
                    for candidate in range(index + 2, len(path)):
                        if self.segment_free(path[index], path[candidate]):
                            next_index = candidate
                    simplified.append(path[next_index])
                    index = next_index
                return simplified
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                neighbor = (node[0] + dx, node[1] + dy)
                if not (0 <= neighbor[0] < nx and 0 <= neighbor[1] < ny):
                    continue
                if not self.free(point(neighbor)) or not self.segment_free(point(node), point(neighbor)):
                    continue
                next_cost = cost + math.hypot(dx, dy) * self.resolution
                if next_cost < costs.get(neighbor, math.inf):
                    costs[neighbor], parents[neighbor] = next_cost, node
                    heapq.heappush(frontier, (next_cost + math.dist(point(neighbor), point(target)), next_cost, neighbor))
        raise NavigationError("PATH_BLOCKED", "No route fits the robot and carried-object footprint")
