"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
from typing import Generator, Tuple
import chess
from chess.engine import PlayResult, Limit
import math
import random
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE
import logging


# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)

class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""

class TimeOutException(Exception):
    """Exception raised for a timeout."""
    def __init__(self, message : str = "Operation timed out"):
        self.message = message
        super().__init__(self.message)

class RatedMoveSequence:
    """A sequence of moves with rating of the sequence."""

    def __init__(self, rating: float, moves: list[chess.Move] = []):
        self.moves = moves
        self.rating = rating

    def __str__(self):
        return f"RatedMoveSequence({self.rating}, {self.moves})"

    def __repr__(self):
        return str(self)

# Base player: returns the first legal move.
class Player:
    """A player picking a move given a board."""

    # The number of nodes visited in the search.
    visited = 0

    def move(self, board: chess.Board) -> chess.Move:
        return next(board.generate_legal_moves())

    def rollout(self, board: chess.Board, max_depth: int) -> float:
        """Simulate a game from the current position, destroying the board.
        The returned value is from the perspective of WHITE.
        So it is 1.0 if WHITE wins, 0.0 if it is a draw, and -1.0 if BLACK wins."""
        outcome = board.outcome()
        depth = 0
        while outcome is None and depth < max_depth:
            board.push(self.move(board))
            outcome = board.outcome()
            depth += 1
        if outcome is None or outcome.winner is None:
            return 0.0
        if outcome.winner: # WHITE wins
            return 1.0
        return -1.0

# Random move engine.
#####################

class RandomPlayer(Player):
    """A player picking a random move."""

    def move(self, board: chess.Board) -> chess.Move:
        return random.choice(list(board.legal_moves))

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a random move."""
        return PlayResult(RandomPlayer().move(board), None)

# Greedy move engine.
#####################

class GreedyPlayer(Player):
    """A player picking a move that directly wins, or a move that does not directly lose."""

    def move(self, board: chess.Board) -> chess.Move:
        """Choose a move that directly wins, or a move that does not directly lose.
        Restores the board state."""
        non_losing_moves: list[chess.Move] = []
        moves = list(board.legal_moves)
        for m in moves:
            self.visited += 1
            board.push(m)
            outcome = board.outcome()
            board.pop()
            # Search for a non-losing move.
            if outcome is None or outcome.winner is None:
                non_losing_moves.append(m)
            # If we find a winning move, commit to it and stop the search.
            elif outcome.winner == board.turn:
                return m
        # If we did not find a winning move, randomly choose a non-losing move.
        if non_losing_moves:
            return random.choice(non_losing_moves)
        # If there are no only losing moves, return one of then randomly.
        return random.choice(moves)

class GreedyMove(ExampleEngine):
    """Get a move that directly wins, or a move that does not directly lose."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a move that directly wins, or a move that does not directly lose."""
        return PlayResult(GreedyPlayer().move(board), None)


class Node:
    """A node in a game tree."""

    # Maximum depth of a rollout.
    rollout_depth = 100

    # Parameter balancing exploration vs. exploitation.
    # c = math.sqrt(1.5)

    def __init__(self, board: chess.Board, depth: int):
        # Depth of the node in the game tree.
        self.depth = depth
        self.finished = None

        # Check if the game is finished and set the value accordingly.
        # If winner is True (WHITE), the value is 1.0.
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                self.finished = 0.0
            elif outcome.winner == chess.WHITE:
                self.finished = 1.0
            else:
                self.finished = -1.0

        # Number of visits to the node.
        self.visits = 0
        # Number of wins of WHITE from the node.
        self.white_wins = 0.0

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
                # Expand the new child.
                board.push(move)
                child = Node(board, self.depth+1)
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

    # Pick the most visited child.
    def best_move(self, board: chess.Board) -> chess.Move:
        # Pick the child with the highest number of visits.
        (child, _quality) = max([(node, node.visits) for node in self.children.keys()], key=lambda x: x[1])
        return self.children[child]

    def best_move_sequence(self, board: chess.Board) -> list[chess.Move]:
        if not self.children:
            return []
        # Pick the child with the highest number of visits.
        (child, _quality) = max([(node, node.visits) for node in self.children.keys()], key=lambda x: x[1])
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

    def log_policy(self, board: chess.Board):
        logger.info(f"Depth: {self.depth}, Rating: {self.white_wins / self.visits}")
        variants = list(self.variants(board))
        variants.sort(key=lambda p: p[0].visits, reverse=True)

        for node, seq in variants:
            logger.info(f"  {node.visits:3d}, {node.white_wins / node.visits:+1.2f}, {board.variation_san(seq)}")



class MCTSPlayer(Player):
    """A player using Monte Carlo Tree Search."""

    def __init__(self, max_nodes: int = 1000):
        # Number of explorations.
        self.max_nodes = max_nodes

    def move(self, board: chess.Board) -> chess.Move:
        root = Node(board, 0)
        for _ in range(self.max_nodes):
            root.explore(board)
        root.log_policy(board)
        return root.best_move(board)


# Monte Carlo Tree Search.
class MonteCarloTreeSearch(ExampleEngine):
    """Explores the game tree with Monte Carlo Tree Search.

    We do not need to store the board at each node, because we can always reconstruct it from the root.

    """
    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a move using Monte Carlo Tree Search."""
        return PlayResult(MCTSPlayer().move(board), None)


# Iterative deepening with alpha-beta pruning.
class IterativeDeepening(ExampleEngine):
    """Explores the game tree to increaing depth, pruning irrelevant subtrees,
    until the time limit is reached.

    The leaves are rated 1.0 for a win of the current player, 0.0 for a draw, and -1.0 for a loss.
    An inner node gets the flipped evaluation v_i of each move.
    The evaluation of the node is the maximum value.

    One of the best moves is randomly picked in the end.
    """
    # Count the number of visited positions in a search.
    visited = 0
    # Interrupt search when we have visited a million nodes.
    max_nodes = 1000000
    # Do not attempt the next iteration when we already visited many nodes.
    hopeless  = 75000

    # The decay factor <= 1 for the evaluation of the child nodes.
    # For decay = 1 the bot will not try to prolongue the game if it thinks it is losing.
    decay = 0.95

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a random best move."""

        self.visited = 0
        best_possible_rating = self.decay

        for depth in range(1, 100):
            if self.visited > self.hopeless:
                break
            try:
                policy = list(self.policy(board.copy(), depth))
            except TimeOutException:
                break

            # Find the best rating.
            best_move_sequence = max(policy, key=lambda x: x.rating)
            best_rating = best_move_sequence.rating
            logger.info(f"Depth {depth}, cum. nodes: {self.visited}, rating: {best_rating}, moves: {board.variation_san(best_move_sequence.moves)}")

            if best_rating >= best_possible_rating or best_rating <= -best_possible_rating:
                break
            best_possible_rating = best_possible_rating * self.decay

        # Find the best moves.
        best_move_sequences = [rms.moves for rms in policy if rms.rating >= best_rating] # type: ignore
        logger.info(f"Best move sequences:")
        for ms in best_move_sequences:
            logger.info(f"- {board.variation_san(ms)}")

        # Choose one of the best moves.
        moves = [ms[0] for ms in best_move_sequences]
        move = random.choice(moves)
        logger.info(f"Move: {board.san(move)}")

        return PlayResult(move, None)

    def policy(self, board: chess.Board, depth: int) -> Generator[RatedMoveSequence]:
        for m in board.legal_moves:
            board.push(m)
            rms = self.value(board, depth, -1.0, 1.0)
            board.pop()
            yield RatedMoveSequence(-rms.rating, [m] + rms.moves)

    # Alpha (α) and beta (β) represent lower and upper bounds for child node values at a given tree depth.
    def value(self, board: chess.Board, depth: int, alpha: float, beta: float) -> RatedMoveSequence:
        self.visited += 1
        if self.visited > self.max_nodes:
            raise TimeOutException()

        # If game is finished or depth exhausted, give the immediate evaluation.
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                return RatedMoveSequence(0.0)
            if outcome.winner == board.turn:
                return RatedMoveSequence(1.0)
            return RatedMoveSequence(-1.0)
        if depth <= 0:
            return RatedMoveSequence(0.0)

        # Otherwise, evaluate the child nodes.
        best_move = None
        for m in board.legal_moves:
            board.push(m)
            rms = self.value(board, depth-1, -beta, -alpha)
            board.pop()
            v = -self.decay * rms.rating
            if v > alpha:
                best_move = None
                alpha = v
            if best_move is None:
                best_move = RatedMoveSequence(v, [m] + rms.moves)
            if alpha >= beta:
                break
        return best_move # type: ignore


# Alpha-beta pruning.
class AlphaBeta(ExampleEngine):
    """Explores the game tree to a certain depth, pruning irrelevant subtrees.

    The leaves are rated 1.0 for a win of the current player, 0.0 for a draw, and -1.0 for a loss.
    An inner node gets the flipped evaluation v_i of each move.
    The evaluation of the node is the maximum value.

    One of the best moves is randomly picked in the end.
    """

    visited = 0

    # The decay factor <= 1 for the evaluation of the child nodes.
    # For decay = 1 the bot will not try to prolongue the game if it thinks it is losing.
    decay = 0.95

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a random move according to the distribution generated from iterative evaluation."""
        k = 5.0
        depth = 4

        # Get moves and values for the current board.
        self.visited = 0
        policy = list(self.policy(board, depth))
        logger.info(f"Visited {self.visited} nodes.")

        # Find the best rating.
        best_rating = max(policy, key=lambda x: x.rating).rating
        logger.info(f"Best rating: {best_rating}")

        # Find the best moves.
        best_move_sequences = [rms.moves for rms in policy if rms.rating >= best_rating]
        logger.info(f"Best move sequences:")
        for ms in best_move_sequences:
            logger.info(f"- {board.variation_san(ms)}")

        # Choose one of the best moves.
        moves = [ms[0] for ms in best_move_sequences]
        move = random.choice(moves)
        logger.info(f"Move: {board.lan(move)}")

        return PlayResult(move, None)


        rated_moves = sorted(policy, key=lambda x: x.rating, reverse=True)
        logger.info(f"Rated moves: {rated_moves}")

        # Calculate the policy.
        weighted = [math.exp(k * rms.rating) for rms in policy]
        # total = sum(weighted)
        # policy = sorted([(moves[i], weighted[i] / total) for i in range(len(moves))], key=lambda x: x[1], reverse=True)
        # # policy = {moves[i]: weighted[i] / total for i in range(len(moves))}
        # logger.info(f"Policy: {policy}")

        # Choose a move.
        moves = [rms.moves[0] for rms in policy]
        [move] = random.choices(population=moves, weights=weighted, k=1)
        logger.info(f"Move: {move}")

        return PlayResult(move, None)

    def policy(self, board: chess.Board, depth: int) -> Generator[RatedMoveSequence]:
        for m in board.legal_moves:
            board.push(m)
            rms = self.value(board, depth, -1.0, 1.0)
            board.pop()
            yield RatedMoveSequence(-rms.rating, [m] + rms.moves)


    # Alpha (α) and beta (β) represent lower and upper bounds for child node values at a given tree depth.
    def value(self, board: chess.Board, depth: int, alpha: float, beta: float) -> RatedMoveSequence:
        self.visited += 1

        # If game is finished or depth exhausted, give the immediate evaluation.
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                return RatedMoveSequence(0.0)
            if outcome.winner == board.turn:
                return RatedMoveSequence(1.0)
            return RatedMoveSequence(-1.0)
        if depth <= 0:
            return RatedMoveSequence(0.0)

        # Otherwise, evaluate the child nodes.
        best_move = None
        for m in board.legal_moves:
            board.push(m)
            rms = self.value(board, depth-1, -beta, -alpha)
            board.pop()
            v = -self.decay * rms.rating
            if v > alpha:
                best_move = None
                alpha = v
            if best_move is None:
                best_move = RatedMoveSequence(v, [m] + rms.moves)
            if alpha >= beta:
                break
        return best_move # type: ignore


    # Alpha (α) and beta (β) represent lower and upper bounds for child node values at a given tree depth.
    def value2(self, board: chess.Board, depth: int, alpha: float, beta: float) -> RatedMoveSequence:
        self.visited += 1
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                return RatedMoveSequence(0.0)
            if outcome.winner == board.turn:
                return RatedMoveSequence(1.0)
            return RatedMoveSequence(-1.0)
        if depth <= 0:
            return RatedMoveSequence(0.0)
        values = self.policy2(board, depth, alpha, beta)
        return max(values, key=lambda x: x.rating)

    def policy1(self, board: chess.Board, depth: int, alpha: float, beta: float) -> list[RatedMoveSequence]:
        moves = list(board.legal_moves)
        # values = [self.value(board.move(m), depth) for m in moves]
        values = [RatedMoveSequence(-9.9, [m]) for m in moves]
        for i in range(len(moves)):
            board.push(moves[i])
            rms = self.value(board, depth-1, -beta, -alpha)
            board.pop()
            v = -rms.rating
            values[i] = RatedMoveSequence(v, [moves[i]] + rms.moves)
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return values


    def policy2(self, board: chess.Board, depth: int, alpha: float, beta: float) -> Generator[RatedMoveSequence]:
        for m in board.legal_moves:
            board.push(m)
            rms = self.value(board, depth-1, -beta, -alpha)
            board.pop()
            v = -self.decay * rms.rating
            yield RatedMoveSequence(v, [m] + rms.moves)
            alpha = max(alpha, v)
            if alpha >= beta:
                break


# Classic min-max.
class MinMax(ExampleEngine):
    """Explores the game tree exhaustively to a certain depth.

    The leaves are rated 1.0 for a win of the current player, 0.0 for a draw, and -1.0 for a loss.
    An inner node gets the flipped evaluation v_i of each move.
    The evaluation of the node is the maximum value.

    The frequency of picking move i is h_i = exp(k * v_i), where k is a parameter.
    """

    visited = 0

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a random move according to the distribution generated from iterative evaluation."""
        k = 5.0
        depth = 3

        # Get moves and values for the current board.
        self.visited = 0
        (moves, values) = self.policy(board, depth)
        logger.info(f"Visited {self.visited} nodes.")
        rated_moves = sorted([(moves[i], values[i]) for i in range(len(moves))], key=lambda x: x[1], reverse=True)
        logger.info(f"Rated moves: {rated_moves}")

        # Calculate the policy.
        weighted = [math.exp(k * v) for v in values]
        # total = sum(weighted)
        # policy = sorted([(moves[i], weighted[i] / total) for i in range(len(moves))], key=lambda x: x[1], reverse=True)
        # # policy = {moves[i]: weighted[i] / total for i in range(len(moves))}
        # logger.info(f"Policy: {policy}")

        # Choose a move.
        [move] = random.choices(population=moves, weights=weighted, k=1)
        logger.info(f"Move: {move}")

        return PlayResult(move, None)


    def value(self, board: chess.Board, depth: int) -> float:
        self.visited += 1
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                return 0.0
            if outcome.winner == board.turn:
                return 1.0
            return -1.0
        if depth <= 0:
            return 0.0
        (_moves, values) = self.policy(board, depth-1)
        return max(values)

    def policy(self, board: chess.Board, depth: int) -> Tuple[list[chess.Move], list[float]]:
        moves = list(board.legal_moves)
        # values = [self.value(board.move(m), depth) for m in moves]
        values = [0.0] * len(moves)
        for i in range(len(moves)):
            m = moves[i]
            board.push(m)
            values[i] = -self.value(board, depth)
            board.pop()
        return (moves, values)



# An iterated version of RandomMove.
class IteratedRandomMove(ExampleEngine):
    """Explores the game tree exhaustively to a certain depth.

    The leaves are rated 1.0 for a win of the current player, 0.0 for a draw, and -1.0 for a loss.
    An inner node gets the flipped evaluation v_i of each move.
    The frequency of picking move i is h_i = exp(k * v_i), where k is a parameter.
    The probability of picking move i is h_i / sum(h_i).
    The evaluation of the node is then the expected value, so sum(h_i * v_i) / sum(h_i).
    """

    visited = 0

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a random move according to the distribution generated from iterative evaluation."""
        k = 5.0
        depth = 3

        self.visited = 0
        (moves, _values, weighted, total) = self.policy(board, depth, k)
        logger.info(f"Visited {self.visited} nodes.")

        policy = {moves[i]: weighted[i] / total for i in range(len(moves))}
        logger.info(f"Policy: {policy}")

        [move] = random.choices(population=moves, weights=weighted, k=1)
        logger.info(f"Move: {move}")

        return PlayResult(move, None)


    def value(self, board: chess.Board, depth: int, k: float) -> float:
        self.visited += 1
        outcome = board.outcome()
        if outcome is not None:
            if outcome.winner is None:
                return 0.0
            if outcome.winner == board.turn:
                return 1.0
            return -1.0
        if depth <= 0:
            return 0.0
        (moves, values, weighted, total) = self.policy(board, depth-1, k)
        v = sum(weighted[i] * values[i] for i in range(len(moves))) / total
        return v

    def policy(self, board: chess.Board, depth: int, k: float) -> Tuple[list[chess.Move], list[float], list[float], float]:
        moves = list(board.legal_moves)
        # values = [self.value(board.move(m), depth, k) for m in moves]
        values = [0.0] * len(moves)
        for i in range(len(moves)):
            m = moves[i]
            board.push(m)
            values[i] = self.value(board, depth, k)  ## BUG: should be modified since player changes
            board.pop()
        weighted = [math.exp(k * v) for v in values]
        total = sum(weighted)
        return (moves, values, weighted, total)

# Bot names and ideas from tom7's excellent eloWorld video



class ComboEngine(ExampleEngine):
    """
    Get a move using multiple different methods.

    This engine demonstrates how one can use `time_limit`, `draw_offered`, and `root_moves`.
    """

    def search(self,
               board: chess.Board,
               time_limit: Limit,
               ponder: bool,  # noqa: ARG002
               draw_offered: bool,
               root_moves: MOVE) -> PlayResult:
        """
        Choose a move using multiple different methods.

        :param board: The current position.
        :param time_limit: Conditions for how long the engine can search (e.g. we have 10 seconds and search up to depth 10).
        :param ponder: Whether the engine can ponder after playing a move.
        :param draw_offered: Whether the bot was offered a draw.
        :param root_moves: If it is a list, the engine should only play a move that is in `root_moves`.
        :return: The move to play.
        """
        if isinstance(time_limit.time, int):
            my_time = time_limit.time
            my_inc = 0
        elif board.turn == chess.WHITE:
            my_time = time_limit.white_clock if isinstance(time_limit.white_clock, int) else 0
            my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, int) else 0
        else:
            my_time = time_limit.black_clock if isinstance(time_limit.black_clock, int) else 0
            my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, int) else 0

        possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)

        if my_time / 60 + my_inc > 10:
            # Choose a random move.
            move = random.choice(possible_moves)
        else:
            # Choose the first move alphabetically in uci representation.
            possible_moves.sort(key=str)
            move = possible_moves[0]
        return PlayResult(move, None, draw_offered=draw_offered)
