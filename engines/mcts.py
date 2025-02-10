# A player using pure Monte Carlo Tree Search.

from typing import Generator, Tuple
import random
import math
import chess
from engines.player import Player, GreedyPlayer
import logging

class MCTSPlayer(Player):
    """A player using Monte Carlo Tree Search."""

    def __init__(self, logger: logging.Logger | None, max_nodes: int = 1000):
        self.logger = logger
        # Number of explorations.
        self.max_nodes = max_nodes

    def move(self, board: chess.Board) -> chess.Move:
        root = Node(board, 0)
        for _ in range(self.max_nodes):
            root.explore(board)
        if self.logger is not None:
            root.log_policy(self.logger, board)
        return root.best_move(board)

class Node:
    """A node in a game tree."""

    # Maximum depth of a rollout.
    rollout_depth = 100

    # Parameter balancing exploration vs. exploitation.
    # c = math.sqrt(1.5)

    def __init__(self, board: chess.Board, depth: int):
        # Depth of the node in the game tree.
        self.depth = depth
        # Is this a terminal node?
        self.finished = None

        # Check if the game is finished and set the value accordingly.
        # If winner is True (WHITE), the value is 1.0.
        outcome = board.outcome()
        if outcome is None:
            # Number of visits to the node.
            self.visits = 0
            # Number of wins of WHITE from the node.
            self.white_wins = 0.0
        else:
            if outcome.winner is None:
                self.finished = 0.0
            elif outcome.winner == chess.WHITE:
                self.finished = 1.0
            else:
                self.finished = -1.0
            self.visits = 1
            self.white_wins = self.finished

        # Unexplored moves in a random order.
        moves = list(board.legal_moves)
        random.shuffle(moves)
        self.unexplored_moves = iter(moves)

        # Children of the node.
        self.children: dict[Node,chess.Move] = {}

    # Run a game simulation from the current node and record its result
    def rollout(self, board: chess.Board) -> float:
        if self.finished is not None:
            value = self.finished
        else:
            value = GreedyPlayer().rollout(board.copy(), self.rollout_depth)
        self.white_wins += value
        self.visits += 1
        return value

    #
    def explore(self, board: chess.Board) -> float:
        if self.finished is not None:
            value = self.finished
        else:
            # Try an unexplored move.
            move = next(self.unexplored_moves, None)
            if move is not None:
                # The reward in case this move wins for us.
                value = 1.0 if board.turn else -1.0
                # Expand the new child.
                board.push(move)
                child = Node(board, self.depth+1)
                # If the move wins for us, we consider the present node finished.
                # We do not need to explore it further.
                if child.finished == value:
                    self.finished = value
                else:
                    value = child.rollout(board)
                board.pop()
                self.children[child] = move

            else:
                # Pick an existing child to explore using UCB.
                # If I am WHITE, I consider the wins of WHITE as positive.
                sign = 1.0 if board.turn else -1.0
                logN = math.log(self.visits)
                qs = [ (node, sign * node.white_wins / node.visits + 2.0 * math.sqrt(logN / node.visits))
                        for node in self.children.keys() ]
                (child, _quality) = max(qs, key=lambda x: x[1])

                # Recurse into the child.
                move = self.children[child]
                board.push(move)
                value = child.explore(board)
                board.pop()

        # Backpropagate
        self.white_wins += value
        self.visits += 1
        return value

    def best_child(self, board: chess.Board) -> 'Node':
        # Pick the child with the highest number of visits.
        def measure(node: Node) -> Tuple[float, int]:
            # If the child is losing, we are winning, so pick it.
            value = node.finished or 0.0
            return (value if board.turn else -value, node.visits)
        # child = max([node for node in self.children.keys()], key=lambda node: node.visits)
        child = max(self.children.keys(), key=measure)
        return child

    # Pick the most visited child.
    def best_move(self, board: chess.Board) -> chess.Move:
        return self.children[self.best_child(board)]

    # Iterate best_move() until we reach a leaf node.
    def best_move_sequence(self, board: chess.Board) -> list[chess.Move]:
        if not self.children:
            return []
        # Iterate best_child() until we reach a leaf node.
        child = self.best_child(board)
        board.push(self.children[child])
        seq = child.best_move_sequence(board)
        board.pop()
        return [self.children[child]] + seq

    def variants(self, board: chess.Board) -> Generator[Tuple['Node', list[chess.Move]]]:
        for node, move in self.children.items():
            board.push(move)
            seq = [move] + node.best_move_sequence(board)
            board.pop()
            yield (node, seq)

    def log_policy(self, logger: logging.Logger, board: chess.Board):
        logger.info(f"Depth: {self.depth}, Rating: {self.white_wins / self.visits}")
        variants = list(self.variants(board))
        variants.sort(key=lambda p: p[0].visits, reverse=True)

        for node, seq in variants:
            logger.info(f"  {node.visits:3d}, {node.white_wins / node.visits:+1.2f}, {board.variation_san(seq)}")

