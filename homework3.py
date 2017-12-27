import sys
from queue import Queue
import copy
import random
import math
import time

ALGORITHM = None
SIZE = None
INIT_STATE = None
LIZARD_COUNT = None
LIZARD_MOVEMENT_RANGE = {}
start_time = None
end_time = None
TERMINATION_DURATION = 270
TREE_COUNT = 0


def dfs(init_state):
    frontier = list()
    frontier.append(init_state)
    while frontier:
        state = frontier.pop()
        if state.is_goal_state():
            write_output(state)
        if time.time() - start_time >= TERMINATION_DURATION:
            write_output(None)
        for neighbour in reversed(state.get_neighbours()):
            frontier.append(neighbour)
    write_output(None)

def bfs(init_state):
    frontier = Queue()
    explored = set()
    frontier.put(init_state)
    while not frontier.empty():
        if time.time() - start_time >= TERMINATION_DURATION:
            write_output(None)
        state = frontier.get()
        if state not in explored:
            explored.add(state)
            for neighbour in state.get_neighbours():
                if neighbour.is_goal_state():
                    write_output(neighbour)
                frontier.put(neighbour)
    write_output(None)


def simulated_annealing(init_state):
    i = 0
    current_state = generate_initial_random_state(init_state)
    if current_state is None:
        write_output(None)
    if current_state.total_attacks == 0:
        write_output(current_state)
    temperature = 5
    decrement_steps = 0.08
    current_stabilizer = 1
    stabilizing_factor = 1.005
    t = temperature
    while True:
        if time.time() - start_time >= TERMINATION_DURATION:
            write_output(None)
        t -= t*decrement_steps
        if t <= 3e-323:
            write_output(current_state)
        j = 0
        while j < current_stabilizer:
            next_state = current_state.generate_random_neighbour_state()
            if next_state is None:
                write_output(None) # neighbour generation failed
            if next_state.total_attacks == 0:
                write_output(next_state)
            e = next_state.total_attacks - current_state.total_attacks
            if e < 0:
                current_state = next_state
            else:
                probability = math.exp(-e / t)
                r = random.uniform(0, 1)
                if r < probability:
                    current_state = next_state
            j += 1
        i += 1
        current_stabilizer *= stabilizing_factor


class Board:
    def __init__(self, state):
        self.state = state
        self.lizard_count = 0
        self.depth = 0
        self.lizard_positions = {}
        self.total_attacks = 0

    def __eq__(self, other):
        if isinstance(other, Board) and self.state == other.state:
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(self.state))

    def is_goal_state(self):
        return self.lizard_count == LIZARD_COUNT

    def clone(self):
        board = Board(list(self.state))
        board.lizard_count = self.lizard_count
        board.lizard_positions = copy.deepcopy(self.lizard_positions)
        board.total_attacks = self.total_attacks
        board.depth = self.depth
        return board

    def add_lizard(self, idx, identity=None):
        if not self.is_idx_valid_for_add(idx):
            return False
        self.state[idx] = 1
        self.lizard_count += 1
        if identity is None:
            identity = self.lizard_count
        self.lizard_positions[identity] = idx
        attack_counts = self.mark_collisions(idx)
        self.total_attacks += attack_counts
        return identity

    def remove_lizard(self, identity):
        lizard_index = self.lizard_positions.pop(identity, None)
        if lizard_index is None:
            return False
        attack_counts = self.mark_collisions(lizard_index)
        self.state[lizard_index] = 0
        self.lizard_count -= 1
        self.total_attacks -= attack_counts
        return lizard_index

    def move_lizard(self, identity, idx):
        is_removed = self.remove_lizard(identity)
        if is_removed is False:
            return False
        is_added = self.add_lizard(idx, identity)
        if not is_added:
            self.add_lizard(is_removed, identity)
            return False
        return True

    def is_idx_valid_for_add(self, idx):
        if idx < 0 or idx >= SIZE * SIZE:
            return False
        if self.state[idx] == 2 or self.state[idx] == 1:
            return False
        return True

    def generate_random_neighbour_index(self, identity):
        lizard_index = self.lizard_positions[identity]
        next_lizard_row = random.randint(0, SIZE - 1)
        next_lizard_col = random.randint(0, SIZE - 1)
        next_lizard_index = next_lizard_row * SIZE + next_lizard_col
        if next_lizard_index == lizard_index or self.state[next_lizard_index] == 1 or self.state[next_lizard_index] == 2:
            return False
        return next_lizard_index

    def generate_random_neighbour_state(self, tries=500):
        next_state = self.clone()
        lizard_to_move = random.randint(1, LIZARD_COUNT)
        neighbour_index = self.generate_random_neighbour_index(lizard_to_move)
        while neighbour_index is False and tries > 0:
            lizard_to_move = random.randint(1, LIZARD_COUNT)
            neighbour_index = self.generate_random_neighbour_index(lizard_to_move)
            tries -= 1
        if tries == 0:
            return None
        if next_state.move_lizard(lizard_to_move, neighbour_index):
            return next_state
        return None

    def mark_collisions(self, idx: int):
        attacks_count = 0
        row = int(idx / SIZE)
        col = int(idx % SIZE)
        # marking left row collisions
        for i in range(idx, row*SIZE - 1, -1):
            if self.state[i] == 2:
                break
            if self.state[i] == 0:
                self.state[i] = -1
            if self.state[i] == 1 and i != idx:
                attacks_count += 1

        # marking right row collisions
        for i in range(idx + 1, SIZE*(row+1)):
            if self.state[i] == 2:
                break
            if self.state[i] == 0:
                self.state[i] = -1
            if self.state[i] == 1 and i != idx:
                attacks_count += 1

        # marking top column collisions
        for i in range(idx - SIZE, col - 1, -1 * SIZE):
            if self.state[i] == 2:
                break
            if self.state[i] == 0:
                self.state[i] = -1
            if self.state[i] == 1 and i != idx:
                attacks_count += 1

        # marking bottom column collisions
        for i in range(idx + SIZE, SIZE*SIZE, SIZE):
            if self.state[i] == 2:
                break
            if self.state[i] == 0:
                self.state[i] = -1
            if self.state[i] == 1 and i != idx:
                attacks_count += 1

        # marking main diagonal collisions upwards
        for i in range(idx - SIZE - 1, -1, -1 * SIZE - 1):
            if self.state[i] == 2 or i % SIZE == SIZE - 1:
                break
            if self.state[i] == 0:
                self.state[i] = -1
            if self.state[i] == 1 and i != idx:
                attacks_count += 1

        # marking main diagonal collisions downwards
        for i in range(idx + SIZE + 1, SIZE * SIZE, SIZE + 1):
            if self.state[i] == 2 or i % SIZE == 0:
                break
            if self.state[i] == 0:
                self.state[i] = -1
            if self.state[i] == 1 and i != idx:
                attacks_count += 1

        # marking alternate diagonal collisions upwards
        for i in range(idx - SIZE + 1, 0, -1 * SIZE + 1):
            if self.state[i] == 2 or i % SIZE == 0:
                break
            if self.state[i] == 0:
                self.state[i] = -1
            if self.state[i] == 1 and i != idx:
                attacks_count += 1

        # marking alternate diagonal collisions downwards
        for i in range(idx + SIZE - 1, SIZE * SIZE, SIZE - 1):
            if self.state[i] == 2 or i % SIZE == SIZE - 1:
                break
            if self.state[i] == 0:
                self.state[i] = -1
            if self.state[i] == 1 and i != idx:
                attacks_count += 1
        return attacks_count

    def get_neighbours(self):
        neighbours = []
        trees = 0
        if self.depth >= SIZE:
            return neighbours
        for i in range(self.depth * SIZE, (self.depth + 1) * SIZE):
            if self.state[i] == 2:
                trees += 1
            if self.state[i] == 0:
                board = self.clone()
                board.state[i] = 1
                board.mark_collisions(i)
                board.depth += 1
                board.lizard_count += 1
                board.parent = self
                neighbours.append(board)

        for i in range(0, trees):
            for neighbour in neighbours:
                for j in range(self.depth * SIZE, (self.depth + 1) * SIZE):
                    if neighbour.state[j] == 0:
                        board = neighbour.clone()
                        board.state[j] = 1
                        board.mark_collisions(j)
                        board.lizard_count += 1
                        board.parent = self
                        neighbours.append(board)

        if len(neighbours) == 0 and self.depth < SIZE:
            self.depth += 1
            return self.get_neighbours()

        return neighbours

    def print_board(self, output_file):
        for i in range(0, SIZE*SIZE):
            if i != 0 and i % SIZE == 0:
                print(file=output_file)
            print(0 if self.state[i] == -1 else self.state[i], file=output_file, end='')


def initialize_lizard_movement_range(init_state):
    lizards_placed = 0
    all_lizards_placed = False
    for i in range(0, SIZE * SIZE):
        if (i > 0 and init_state.state[i - 1] == 2 and init_state.state[i] != 2) or (i % SIZE == 0 and init_state.state[i] != 2):
            lizards_placed = (lizards_placed % LIZARD_COUNT) + 1
            if LIZARD_MOVEMENT_RANGE.get(lizards_placed) is None:
                LIZARD_MOVEMENT_RANGE[lizards_placed] = list()
            if lizards_placed == LIZARD_COUNT:
                all_lizards_placed = True
        if init_state.state[i] != 2:
            LIZARD_MOVEMENT_RANGE[lizards_placed].append(i)
    if not all_lizards_placed:
        return False
    return True


def generate_initial_random_state(init_state):
    is_initialized = initialize_lizard_movement_range(init_state)
    if is_initialized:
        for key, value in LIZARD_MOVEMENT_RANGE.items():
            init_state.add_lizard(random.choice(value), key)
        return init_state
    else:
        return None


def is_dim_idx_in_range(idx):
    if idx < 0 or idx >= SIZE:
        return False
    return True


def read_input():
    input_file = open('input.txt', 'r')
    algorithm_name = input_file.readline().strip().upper()
    nursery_size = int(input_file.readline().strip())
    total_lizards = int(input_file.readline().strip())
    nursery_details = []
    global TREE_COUNT
    for i in range(nursery_size):
        line = input_file.readline().strip()
        for c in line:
            block_val = int(c)
            if block_val == 2:
                TREE_COUNT += 1
            nursery_details.append(block_val)
    input_file.close()
    global SIZE, LIZARD_COUNT, INIT_STATE, ALGORITHM
    ALGORITHM = algorithm_name
    SIZE = nursery_size
    LIZARD_COUNT = total_lizards
    INIT_STATE = nursery_details


def write_output(search_output):
    output_file = open('output.txt', 'w')
    if search_output is None:
        print("FAIL", file=output_file, end='')
    else:
        if search_output.total_attacks != 0:
            print("FAIL", file=output_file, end='')
        else:
            print("OK", file=output_file)
            search_output.print_board(output_file)
    output_file.close()
    sys.exit(0)


def main():
    read_input()
    init_board = Board(INIT_STATE)
    global start_time, end_time
    start_time = time.time()
    if SIZE + TREE_COUNT < LIZARD_COUNT:
        write_output(None)
    if ALGORITHM == 'DFS':
        dfs(init_board)
    elif ALGORITHM == 'BFS':
        bfs(init_board)
    elif ALGORITHM == 'SA':
        simulated_annealing(init_board)
    else:
        print("Invalid Algorithm")
        write_output(None)
        sys.exit(2)


if __name__ == "__main__":
    main()