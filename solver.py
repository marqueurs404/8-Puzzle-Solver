import os
import sys
from heapq import *


class Puzzle(object):
    class Node:
        """Inner class to represent a search node."""

        def __init__(self, state, depth, f_score, action, parent):
            self.state = state
            self.depth = depth
            self.f_score = f_score
            self.action = action  # direction which blank space was moved in
            self.parent = parent

        def __lt__(self, other):
            return self.f_score < other.f_score

        def __repr__(self):
            return '( ' + str(self.state) + ',' + str(self.depth) + ',' + str(self.action) + ' )'

    def __init__(self, init_state, goal_state):
        self.init_state = init_state
        self.goal_state = goal_state
        self.actions = []
        self.solvable = True

        # Uncomment the heuristic to use
        self.heuristic = 'Manhattan'
        # self.heuristic = 'HammingAndManhattan'

        self.explored_states = set()
        self.frontier = []
        self.num_moves = 0
        self.num_nodes_explored = 0
        self.total_nodes_generated = 0
        self.max_frontier_size = 0

    def solve(self):
        is_solvable = self.check_solvable()
        if not is_solvable:
            print('UNSOLVABLE')
            return ['UNSOLVABLE']

        print("Running A* Search with " + self.heuristic + " heuristic...")

        # generate start node
        init_node = self.Node(self.init_state, 0, 0, '', None)
        init_node.f_score = self.calculate_f_score(
            init_node, self.goal_state, self.heuristic)
        print("h(s0): " + str(init_node.f_score))
        heappush(self.frontier, init_node)
        self.total_nodes_generated += 1

        # begin search
        while len(self.frontier) != 0:
            current_node = heappop(self.frontier)
            self.num_nodes_explored += 1

            self.explored_states.add(self.get_state_string(current_node.state))

            # goal test
            if self.check_states_equal(current_node.state, self.goal_state):
                print("found goal!")
                break

            # generate children nodes and put in frontier
            children = self.generate_children(current_node)
            for child in children:
                heappush(self.frontier, child)
            self.total_nodes_generated += len(children)

            # calculate max frontier size
            self.max_frontier_size = max(
                self.max_frontier_size, len(self.frontier))

        # post-search
        self.trace_path(current_node)
        self.print_stats()
        return self.actions

    def print_stats(self):
        print("Total nodes generated: " + str(self.total_nodes_generated))
        print("Number of nodes explored: " + str(self.total_nodes_generated))
        print("Max frontier size: " + str(self.max_frontier_size))
        print("Number of moves: " + str(self.num_moves))

    def trace_path(self, node):
        """Trace the actions taken to reach goal node."""
        self.num_moves = node.depth
        while node.parent is not None:
            self.actions.append(self.reverse_action(node.action))
            node = node.parent
        self.actions.reverse()

    def check_states_equal(self, state_a, state_b):
        """Check if states are equal."""
        for i in range(3):
            for j in range(3):
                if state_a[i][j] != 0 and state_a[i][j] != state_b[i][j]:
                    return False
        return True

    def check_solvable(self):
        """Checks if state is solvable."""
        inversions = 0
        for i in range(0, 9):
            x, y = divmod(i, 3)
            for j in range(i + 1, 9):
                x2, y2 = divmod(j, 3)
                if self.init_state[x][y] > 0 and self.init_state[x2][y2] > 0 and self.init_state[x2][y2] < self.init_state[x][y]:
                    inversions += 1
        return inversions % 2 == 0

    def calculate_f_score(self, current_node, goal_state, heuristic_type):
        """Calculates f-score / cost for current node"""
        if heuristic_type.lower() == 'manhattan':
            h_score = self.calculate_h_manhattan(
                current_node.state, goal_state)
        elif heuristic_type.lower() == 'hammingandmanhattan':
            h_score = (self.calculate_h_manhattan(
                current_node.state, goal_state) + self.calculate_h_hamming(
                current_node.state, goal_state))/2
        g_score = current_node.depth
        return h_score + g_score

    def calculate_h_hamming(self, current_state, goal_state):
        """calculates h-score for the Hamming distance heuristic (number of misplaced tiles)."""
        h = 0
        for i in range(3):
            for j in range(3):
                if current_state[i][j] != goal_state[i][j] and current_state[i][j] != 0:
                    h += 1
        return h

    def calculate_h_manhattan(self, current_state, goal_state):
        """calculates h-score for the Manhattan distance heuristic 
           (sum of vertical & horizontal distance from the tiles to their goal positions)."""
        h = 0
        for i in range(3):
            for j in range(3):
                if current_state[i][j] != 0:
                    x, y = divmod(current_state[i][j]-1, 3)
                    # print("this is x ", x, "this is y ", y, "this is the init state ", self.init_state[i][j], "Adding ", abs(x-i), abs(y-j))
                    h += abs(x-i) + abs(y-j)
        return h

    def find_blank_position(self, state):
        """Returns the row & column index of the blank position on the state."""
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j

    def generate_children(self, parent):
        """"Generate all possible children nodes."""
        children = []
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        for dir in directions:
            # don't revert back to parent state
            if dir == (self.reverse_action(parent.action)):
                continue

            new_state = self.get_new_state(parent.state, dir)
            if new_state is None:  # move leads to invalid state
                continue

            # graph-search version, don't explore previously explored states
            if self.get_state_string(new_state) in self.explored_states:
                continue

            child_node = self.Node(
                new_state, parent.depth + 1, 0, dir, parent)
            child_node.f_score = self.calculate_f_score(
                child_node, self.goal_state, self.heuristic)
            children.append(child_node)
        return children

    def get_new_state(self, state, direction):
        """Get new state after moving the blank position in the given direction. Returns None if move is invalid."""
        x, y = self.find_blank_position(state)

        if direction == 'UP':
            x2, y2 = x-1, y
        elif direction == 'DOWN':
            x2, y2 = x+1, y
        elif direction == 'LEFT':
            x2, y2 = x, y-1
        elif direction == 'RIGHT':
            x2, y2 = x, y+1

        # copy state and swap tiles
        if 0 <= x2 < 3 and 0 <= y2 < 3:
            new_state = [row[:] for row in state]

            temp = new_state[x2][y2]
            new_state[x2][y2] = new_state[x][y]
            new_state[x][y] = temp

            return new_state
        else:
            return None

    def get_state_string(self, state):
        return ''.join([str(item) for sublist in state for item in sublist])

    def reverse_action(self, action):
        new_action = ''
        if action == 'UP':
            new_action = 'DOWN'
        elif action == 'DOWN':
            new_action = 'UP'
        elif action == 'LEFT':
            new_action = 'RIGHT'
        elif action == 'RIGHT':
            new_action = 'LEFT'
        return new_action


if __name__ == "__main__":
    # do NOT modify below
    if len(sys.argv) != 3:
        raise ValueError("Wrong number of arguments!")

    try:
        f = open(sys.argv[1], 'r')
    except IOError:
        raise IOError("Input file not found!")

    init_state = [[0 for i in range(3)] for j in range(3)]
    goal_state = [[0 for i in range(3)] for j in range(3)]
    lines = f.readlines()

    i, j = 0, 0
    for line in lines:
        for number in line:
            if '0' <= number <= '8':
                init_state[i][j] = int(number)
                j += 1
                if j == 3:
                    i += 1
                    j = 0

    for i in range(1, 9):
        goal_state[(i-1)//3][(i-1) % 3] = i
    goal_state[2][2] = 0

    puzzle = Puzzle(init_state, goal_state)
    ans = puzzle.solve()

    with open(sys.argv[2], 'a') as f:
        for answer in ans:
            f.write(answer+'\n')
