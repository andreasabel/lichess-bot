# Play a tournament between two players.

from datetime import datetime
import logging
from rich.logging import RichHandler
from pathlib import Path

import chess
from chess.variant import AtomicBoard

from chessboard import display
from chessboard.board import Color


from engines.player import *
from engines.alphabeta import *
from engines.mcts import MCTSPlayer, TimedMCTSPlayer

# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)


def main():

    level = logging.DEBUG

    # Set up pretty logging to the console.
    # Code from lib/lichess-bot.py
    console_handler = RichHandler()
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    all_handlers: list[logging.Handler] = [console_handler]
    logging.basicConfig(level=level, handlers=all_handlers, force=True)

    # (wins, losses, draws) = tournament(GreedyPlayer(), RandomPlayer(), games=100)
    # print(f"GreedyPlayer vs. RandomPlayer: {wins} wins, {losses} losses, {draws} draws")

    # (wins, losses, draws) = tournament(AlphaBetaPlayer(), GreedyPlayer(), games=100)
    # print(f"AlphaBetaPlayer vs. GreedyPlayer: {wins} wins, {losses} losses, {draws} draws")

    # (wins, losses, draws) = tournament(
    #     IterativeDeepeningAlphaBetaPlayer(decay=0.95),
    #     GreedyPlayer(),
    #     games=10)
    # print(f"IterativeDeepeningAlphaBetaPlayer vs. GreedyPlayer: {wins} wins, {losses} losses, {draws} draws")

    # (wins, losses, draws) = tournament(
    #     MCTSPlayer(logger=None, max_nodes=1000),
    #     IterativeDeepeningAlphaBetaPlayer(decay=0.95),
    #     games=10)
    # print(f"MCTSPlayer vs. IterativeDeepeningAlphaBetaPlayer: {wins} wins, {losses} losses, {draws} draws")

    (wins, losses, draws) = tournament(
        TimedMCTSPlayer(logger=None, thinking_time=1.0),
        TimedIterativeDeepeningAlphaBetaPlayer(decay=0.95),
        games=10)
    print(f"MCTS vs. IterativeDeepeningAlphaBeta (1.0s): {wins} wins, {losses} losses, {draws} draws")

def tournament(player: Player, opponent: Player, games: int = 100) -> tuple[int, int, int]:
    """Play a tournament between two players,
    alternating the starting player in each game.
    Return the number of wins for each player and the number of draws."""

    players = {0: player, 1: opponent}
    wins = {0: 0, 1: 0}
    draws = 0
    for i in range(games):
        logging.info(f"Game {i+1}/{games}")
        p = i % 2
        o = (i + 1) % 2
        outcome = play(players[p], players[o])
        if outcome.winner is None:
            draws += 1
        elif outcome.winner:
            wins[p] += 1
        else:
            wins[o] += 1
        logging.info(f"Score: {wins[0]}-{wins[1]}-{draws}")

    return wins[0], wins[1], draws

def play(player1: Player, player2: Player) -> chess.Outcome:
    """Play a game between two players."""
    players = {0: player1, 1: player2}
    title = f"{player1.name} vs. {player2.name}"
    logging.info(title)
    start = AtomicBoard()
    board = start.copy()
    outcome: chess.Outcome | None  = None
    p = 0

    # Set up chessboard display.
    game_board = display.start(fen=board.fen(), bg_color=Color.WHITE, caption=title)

    while outcome is None:
        m: chess.Move = players[p].move(board)
        logging.info(f"Move: {board.variation_san([m])}  {players[p].move_info}")
        board.push(m)
        outcome = board.outcome()
        p = 1 - p
        # Update display.
        display.update(board.fen(), game_board)

    game = start.variation_san(board.move_stack)
    logging.info(f"Game: {game}")
    logging.info(f"Outcome: {outcome.result()}")

    # Ensure the tournament directory exists.
    Path("tournament").mkdir(exist_ok=True)
    # Make file name from the names of the two players and the current date and time.
    now = datetime.now().replace(microsecond=0)
    filename = f"{player1.name}-vs-{player2.name}-{now.isoformat()}"
    # Save the game to a file in the tournament directory.
    with open(f"tournament/{filename}.pgn", "w") as f:
        f.write("[Variant \"Atomic\"]\n")
        f.write("[Event \"Tournament\"]\n")
        f.write("[White \"{}\"]\n".format(player1.name))
        f.write("[Black \"{}\"]\n".format(player2.name))
        # The date in YYYY.MM.DD format.
        f.write("[Date \"{}\"]\n".format(now.date().isoformat()))
        f.write("[UTCDate \"{}\"]\n".format(now.date().isoformat()))
        f.write("[UTCTime \"{}\"]\n".format(now.time().isoformat()))
        f.write(f"[Result \"{outcome.result()}\"]\n")
        f.write(game)

    return outcome

if __name__ == "__main__":
    main()