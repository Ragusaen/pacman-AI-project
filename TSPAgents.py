import copy
import time
import timeit

from networkx.algorithms.approximation.traveling_salesman import traveling_salesman_problem
from networkx.algorithms.shortest_paths.unweighted import single_target_shortest_path_length
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path, single_source_dijkstra
from networkx.utils.misc import pairwise

from layout import Layout
from pacman import Directions
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
    """
    TSP Agent that uses a Traveling Salesman Problem (TSP) and Greedy approach to find the shortest path to collect food.

    The agent precomputes the TSP cycle. The goal is to follow this cycle to collect food.

    Online, it removes the nodes with the ghosts (and the neighbours) from the graph and computes the shortest path to
    the closest food. The weighs on the graph are based on 3 factors:
    1. Isolation: The ratio of the number of neighbours that are important nodes to the total number of neighbours. This
    is to avoid leaving foods behind. The more isolated, the lower weight, i.e. more likely to go there.
    2. Cycle cost: Edges that are in the TSP cycle have their weight lowered. This is to encourage to follow the cycle.
    3. Danger: Edges that lead closer to the ghosts have their weight increased. This is to avoid going near the ghosts.
    """
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

        self.cycle = traveling_salesman_problem(self.graph, nodes=important_nodes)
        pi = self.cycle.index(pacman_pos)
        self.cycle = list(self.cycle[pi + 1:] + self.cycle[:pi])

    def getAction(self, state : GameState):
        # Time limit: approx 1 second
        # Look-up offline policy or online search with MCTS/LRTDP using some pre-computed value function?
        g = copy.deepcopy(self.graph)

        pacman_pos = state.getPacmanPosition()
        pacman_pos = (int(pacman_pos[0]), int(pacman_pos[1]))

        if pacman_pos in self.cycle:
            self.cycle.remove(pacman_pos)
        important_nodes = set(state.getFood().asList())

        for ghost_state in state.getGhostStates():
            pos = ghost_state.configuration.pos
            pos = (int(pos[0]), int(pos[1]))

            if pos in g.nodes:
                g.remove_node(pos)
                if pos in important_nodes:
                    important_nodes.remove(pos)

        assert all(node in g.nodes for node in important_nodes)

        if len(important_nodes) <= 0:
            return Directions.STOP

        for p in important_nodes:
            N = len(list(g.neighbors(p)))
            if N != 0:
                g.nodes[p]['isolation'] = sum(1 for n in g.neighbors(p) if n in important_nodes) / N

        for u, v in g.edges:
            cycle_cost = self.cycle.index(v) / len(self.cycle) if v in self.cycle else 1.0
            g[u][v]['weight'] = 10 * min(g.nodes[u]['isolation'], g.nodes[v]['isolation']) + 5 * cycle_cost

        g.nodes[pacman_pos]['is_pacman']=True
        sp = single_source_dijkstra(g, pacman_pos, weight='weight')

        closest_food = min(important_nodes, key=lambda x: sp[0][x] if x in sp[0] else float('inf'))
        if closest_food not in sp[0]:
            return Directions.STOP

        next = sp[1][closest_food][1]
        diff = (next[0] - pacman_pos[0], next[1] - pacman_pos[1])
      

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
        
