import copy
import time
import timeit

from networkx.algorithms.approximation.traveling_salesman import traveling_salesman_problem, simulated_annealing_tsp, \
    greedy_tsp
from networkx.algorithms.shortest_paths.unweighted import single_target_shortest_path_length
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path, single_source_dijkstra, \
    all_pairs_dijkstra
from networkx.utils.misc import pairwise

from layout import Layout
from pacman import Directions, TIME_PENALTY
from game import Agent
import random
import game
import util
from pacman import GameState
from ghostAgents import RandomGhost

import networkx as nx
import matplotlib.pyplot as plt
from itertools import product

def build_graph(layout: Layout) -> nx.Graph:
    g = nx.Graph()

    for x in range(layout.width):
        for y in range(layout.height):
            if not layout.walls[x][y]:
                g.add_node((x, y), danger=1, isolation=1)

    for n in g.nodes:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (n[0] + dx, n[1] + dy)
            if neighbor in g.nodes:
                g.add_edge(n, neighbor, weight=1)
    return g


class TSPAgent(game.Agent):
    def __init__(self, layout: Layout, **kwargs):
        super().__init__()
        print(layout)
        self.layout = layout
        self.graph: nx.Graph = None
        self.cycle: list[tuple[int,int]] = None

        self.offline_planning()

    def offline_planning(self):
        # Compute offline policy and/or value function
        # Time limit: 10 minutes
        self.graph = build_graph(self.layout)

        important_nodes = set(self.layout.food.asList())

        pacman_pos = None
        for is_pacman, pos in self.layout.agentPositions:
            if is_pacman:
                pacman_pos = (int(pos[0]), int(pos[1]))
                important_nodes.add(pacman_pos)

        self.cycle: list[tuple[int,int]] = traveling_salesman_problem(self.graph, nodes=important_nodes)[:-1]
        self.sps = dict(all_pairs_dijkstra(self.graph))

    def heuristic(self, state: GameState):
        # Compute heuristic value for state. Remove all non-important nodes from cycle. Find closest node in self.cycle,
        # compute distance of shortest paths between nodes.
        important_nodes = set(state.getFood().asList())
        new_cycle = [n for n in self.cycle if n in important_nodes]

        pacman_pos = state.getPacmanPosition()
        pacman_pos = (int(pacman_pos[0]), int(pacman_pos[1]))

        closest_node = min(new_cycle, key=lambda p: self.sps[pacman_pos][0][p])
        cni = new_cycle.index(closest_node)
        new_cycle = new_cycle[cni:] + new_cycle[:cni]

        score = state.getScore() + 500 # Assume 500 score for winning
        for p in new_cycle:
            score -= TIME_PENALTY * self.sps[pacman_pos][0][p]
            pacman_pos = p
            score += 10 # 10 score for each food eaten

        return score

    def getAction(self, state : GameState):
        best_action = self.getExpectedScoreNextKSteps(state, 1)[1]
        
        print("Best action: ", best_action)

        return best_action

        

        print(time.time() - t0)
        if diff == (1, 0):
            print("EAST")
            return Directions.EAST
        elif diff == (-1, 0):
            print("WEST")
            return Directions.WEST
        elif diff == (0, 1):
            print("NORTH")
            return Directions.NORTH
        elif diff == (0, -1):
            print("SOUTH")
            return Directions.SOUTH
        else:
            raise ValueError("Invalid diff value: {}".format(diff))
        
    def getStatesProbDistribution(self, state: GameState):
        "Returns the probability of reaching each reachable state in the next k steps."
        num_ghosts = self.layout.getNumGhosts()
        ghost_distributions = {}
        for index in range(num_ghosts):
            ghost = RandomGhost(index + 1)
            ghost_distributions[index] = ghost.getDistribution(state)
    
        # print(ghost_distributions)

        all_ghosts_actions_combinations = list(product(*[actions.keys() for actions in ghost_distributions.values()]))
        state_probabilities = {}

        for combination in all_ghosts_actions_combinations:
            prob = 1.0
            for ghost_index, action in enumerate(combination):
                prob *= ghost_distributions[ghost_index][action]
                state_probabilities[combination] = prob

        # print(state_probabilities)

        #  Generate successor states from applying the actions
        successor_states_probabilities = {}
        for combination in all_ghosts_actions_combinations:
            successor = state.deepCopy()
            for ghost_index, action in enumerate(combination):
                # if action in successor.getLegalActions(ghost_index + 1):
                successor = successor.generateSuccessor(ghost_index + 1, action)
            successor_states_probabilities[successor] = state_probabilities[combination]

        # for successor, prob in successor_states_probabilities.items():
        #     print("Successor state: \n", successor)
        #     print("Probability: ", prob)
            
        
        return successor_states_probabilities
    
    def getExpectedScoreNextKSteps(self, state: GameState, k: int):
        """
        Recursive function that computes the expected score of the next k steps.
        """
        if state.isWin() or state.isLose():
            return state.getScore(), None

        if k == 0:
            print(state)
            print("Heuristic: ", self.heuristic(state))
            return self.heuristic(state), None
        else:
            successor_states_probabilities = self.getStatesProbDistribution(state)

            max_expected_score = -float('inf')
            for action in state.getLegalActions(0):
                expected_score = 0
                for successor, prob in successor_states_probabilities.items():
                    new_state = successor.deepCopy()
                    new_state = new_state.generateSuccessor(0, action)
                    expected_score += prob * self.getExpectedScoreNextKSteps(new_state, k - 1)[0]
                if expected_score > max_expected_score:
                    max_expected_score = expected_score
                    best_action = action

            return max_expected_score, best_action