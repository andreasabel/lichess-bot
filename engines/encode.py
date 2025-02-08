"""
Encode a chess board as bitboards suitable for input to a neural network.
Decode the output of a neural network to a move on the board.
"""

from typing import Iterator, Literal
import chess
import numpy as np

# type InputLayer = np.ndarray[np.uint8, shape=(13, 8, 8)]
type InputLayer = np.ndarray[tuple[Literal[13], Literal[8], Literal[8]], np.dtype[np.uint8]]
type OutputLayer = np.ndarray[tuple[Literal[8], Literal[8], Literal[8], Literal[8]], np.dtype[np.float32]]

def encode_board(board: chess.Board) -> InputLayer:
    """
    Encode a chess board as a set of bitboards from the perspective of the player.

    The bitboards are ordered as follows:
    0-5: Pieces of the player (P, N, B, R, Q, K)
    6-11: Pieces of the opponent (p, n, b, r, q, k)
    12: Castling rights and en passant square
    """
    # Initialize empty bitboards
    bitboards: InputLayer = np.zeros((13, 8, 8), dtype=np.uint8)

    # Fill in bitboards for each piece type
    # The pieces are numbered as follows:
    # 1: Pawn, 2: Knight, 3: Bishop, 4: Rook, 5: Queen, 6: King
    # The files are numbered from 0 to 7, with 0 being the a-file.
    # The ranks are numbered from 0 to 7, with 0 being the 1st rank.
    # We view the board from the perspective of the player to move.
    # So for black, we flip the board vertically, meaning that we invert the ranks.
    player = board.turn
    opponent = not player
    def rank(square: int) -> int:
        return 7 - chess.square_rank(square) if player == chess.BLACK else chess.square_rank(square)

    for piece in chess.PIECE_TYPES:
        # Pieces of the player
        for square in board.pieces(piece, player):
            bitboards[piece - 1][rank(square)][chess.square_file(square)] = 1
        # Pieces of the opponent
        for square in board.pieces(piece, opponent):
            bitboards[piece + 5][rank(square)][chess.square_file(square)] = 1

    # We add a single bitboard for castling rights and en passant square.
    # The castling rights are are placed on the target square of the king.
    # The en passant square is placed on the square where the pawn would move to.

    # Add bitboard entries for castling rights.
    if board.has_kingside_castling_rights(player):
        bitboards[12][0][6] = 1
    if board.has_queenside_castling_rights(player):
        bitboards[12][0][2] = 1
    if board.has_kingside_castling_rights(opponent):
        bitboards[12][7][6] = 1
    if board.has_queenside_castling_rights(opponent):
        bitboards[12][7][2] = 1

    # Add bitboard entry for en passant square.
    if board.ep_square is not None:
        bitboards[12][rank(board.ep_square)][chess.square_file(board.ep_square)] = 1

    return bitboards

def absolute_move(board: chess.Board, move: chess.Move) -> chess.Move:
    """
    Convert a relative move to an absolute move and remove an invalid promotion.

    The input move is from the perspective of the player to move, which defaults to white.
    So for black, we flip the move vertically, meaning that we invert the ranks.
    """
    # Remove an invalid promotion.
    # If the target is not on the 8th rank (from the perspective of white),
    # then there cannot be a promotion
    promotion = move.promotion if chess.square_rank(move.to_square) == 7 else None

    # Convert the move to an absolute move
    if board.turn == chess.BLACK:
        move = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), promotion)
    return move

def move_with_queen_promotion(src_file: np.intp, src_rank: np.intp, dst_file: np.intp, dst_rank: np.intp) -> chess.Move:
    """
    Create a move with a queen promotion.
    """
    sf = int(src_file); assert 0 <= sf < 8
    sr = int(src_rank); assert 0 <= sr < 8
    tf = int(dst_file); assert 0 <= tf < 8
    tr = int(dst_rank); assert 0 <= tr < 8
    return chess.Move(chess.square(sf, sr), chess.square(tf, tr), chess.QUEEN)

def decode_move(output: OutputLayer) -> chess.Move:
    """
    Decode the output of the neural network to a move on the board.
    The output is a 8x8x8x8 tensor, where the first two dimensions represent the source square.
    The last two dimensions represent the target square.
    Pawn promotion is always to a queen.
    """
    # Find the move with the highest probability
    move_index : np.intp = np.argmax(output)
    (sf, sr, tf, tr) = np.unravel_index(move_index, shape=(8, 8, 8, 8))
    return move_with_queen_promotion(sf, sr, tf, tr)

def encode_moves(board: chess.Board, moves: Iterator[chess.Move]) -> OutputLayer:
    """
    Encode a move or a list of moves as a tensor.
    The output is a 8x8x8x8 tensor, where the first two dimensions represent the source square.
    The last two dimensions represent the target square.

    This function can be used to encode a list of legal moves.
    """
    # Initialize empty tensor
    output: OutputLayer = np.zeros((8, 8, 8, 8), dtype=np.float32)

    # Fill in the tensor for each move
    for move in moves:

        # Convert the move to an absolute move
        move = absolute_move(board, move)

        # Fill in the tensor
        output[chess.square_file(move.from_square)][chess.square_rank(move.from_square)][chess.square_file(move.to_square)][chess.square_rank(move.to_square)] = 1

    return output