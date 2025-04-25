import sys
from collections import deque
import heapq

# Directions and their coordinate changes
ACTIONS = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1),
    'V': (0, 0)  # Vacuum
}

def parse_world(filename):
    """
    Parse the world file and return:
      - grid: list of list of chars
      - robot_pos: (row, col)
      - dirt: list of (row, col)
    """
    with open(filename, 'r') as file:
        cols = int(file.readline())
        rows = int(file.readline())
        grid = []
        robot_pos = None
        dirt = []
        for r in range(rows):
            line = list(file.readline().strip())
            for c, ch in enumerate(line):
                if ch == '@':
                    robot_pos = (r, c)
                elif ch == '*':
                    dirt.append((r, c))
            grid.append(line)
    return grid, robot_pos, dirt


def get_successors(state, grid):
    """
    Given a state (robot_pos, remaining_dirt), return list of
    (action, next_state, cost).
    """
    (rpos, dirt) = state
    successors = []
    rows, cols = len(grid), len(grid[0])

    for action, (dr, dc) in ACTIONS.items():
        nr, nc = rpos[0] + dr, rpos[1] + dc
        # Vacuum action
        if action == 'V':
            if rpos in dirt:
                new_dirt = set(dirt)
                new_dirt.remove(rpos)
                successors.append(('V', (rpos, frozenset(new_dirt)), 1))
        # Move actions
        else:
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#':
                successors.append((action, ((nr, nc), dirt), 1))
    return successors

def depth_first_search(start_state, grid):
    """
    Perform DFS based on textbook pseudocode. Prints actions and stats.
    """
    frontier = [(start_state, [])]  # Stack: list with (state, path)
    explored = set()
    nodes_generated = 0
    nodes_expanded = 0

    while frontier:
        state, path = frontier.pop()

        if state in explored:
            continue

        explored.add(state)
        nodes_expanded += 1

        # Goal test: no dirt left
        if not state[1]:
            for a in path:
                print(a)
            print(f"{nodes_generated} nodes generated")
            print(f"{nodes_expanded} nodes expanded")
            return

        for action, child, cost in get_successors(state, grid):
            if child not in explored and all(child != s for s, _ in frontier):
                frontier.append((child, path + [action]))
                nodes_generated += 1

    print("No solution found.")



def uniform_cost_search(start_state, grid):
    """
    Perform UCS, print actions and stats. AKA Breadth First Search.
    """
    frontier = []
    heapq.heappush(frontier, (0, start_state, []))
    visited = {}
    nodes_generated = 0
    nodes_expanded = 0

    while frontier:
        cost_so_far, state, path = heapq.heappop(frontier)
        if state in visited and visited[state] <= cost_so_far:
            continue
        visited[state] = cost_so_far
        nodes_expanded += 1

        # Goal test
        if not state[1]:
            for a in path:
                print(a)
            print(f"{nodes_generated} nodes generated")
            print(f"{nodes_expanded} nodes expanded")
            return

        for action, succ, step_cost in get_successors(state, grid):
            new_cost = cost_so_far + step_cost
            if succ not in visited or new_cost < visited.get(succ, float('inf')):
                nodes_generated += 1
                heapq.heappush(frontier, (new_cost, succ, path + [action]))



def main():
    if len(sys.argv) != 3:
        print("Usage: python3 planner.py [uniform-cost|depth-first] [world-file]")
        return

    algorithm = sys.argv[1]
    world_file = sys.argv[2]

    grid, robot_pos, dirty_cells = parse_world(world_file)
    start_state = (robot_pos, frozenset(dirty_cells))

    if algorithm == 'depth-first':
        depth_first_search(start_state, grid)
    elif algorithm == 'uniform-cost':
        uniform_cost_search(start_state, grid)
    else:
        print("Invalid algorithm. Choose 'depth-first' or 'uniform-cost'.")

if __name__ == "__main__":
    main()
