"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
#import random
#import time
#import copy
from collections import defaultdict
import logging
from sample_players import improved_score

logging.basicConfig(filename='tournament.log', filemode='w',\
                    format='%(lineno)d:%(levelname)s: %(message)s', level=logging.DEBUG)

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
#==============================================================================
#     # AVERAGE: (myMoves - 3*opMoves)
#     return float(len(game.get_legal_moves(player)) \
#                  - 3 * len(game.get_legal_moves(game.get_opponent(player))))
#==============================================================================

#==============================================================================
#     # BAD: Possible reach wihin 2 moves with current board
#     return float(reach(game, player) - 3*reach(game, game.get_opponent(player)))
#==============================================================================

#==============================================================================
#     # AVERAGE:
#     # Closer to the center of the board = better (more options)
#     # Enemy further away from center = better
#     # --> use manhattan distance
#     pr, pc = game.get_player_location(player)
#     er, ec = game.get_player_location(game.get_opponent(player))
#     h_center = float(game.height) / 2
#     w_center = float(game.width) / 2
#     return float((abs(h_center - pr) + abs(w_center - pc))) \
#                 - float((abs(h_center - er) + abs(w_center - ec)))
#==============================================================================
    
#==============================================================================
#     # GOOD:
#     # Length of the possible knight's tour in this situation - 
#     # Length of the opponent's KT in this situation
#     return float(knights_tour(0, game.copy(), game.get_player_location(player)) \
#                  - 3 * knights_tour(0, game.copy(), game.get_player_location(game.get_opponent(player))))
#==============================================================================

    # COMBO:
    w1, w2 = (1, 2)
    h_num_legal_moves = (float(w1 * len(game.get_legal_moves(player)) \
                             - w2 * len(game.get_legal_moves(game.get_opponent(player)))))
    h_num_legal_moves = 100 * h_num_legal_moves / max(w1 * 8, abs(-w2 * 8))  # Normalized heuristic
    
    w1, w2 = (1, 2)
    h_knights_tour = float(w1 * thorough_knights_tour(0, game.copy(), game.get_player_location(player)) \
                         - w2 * thorough_knights_tour(0, game.copy(), game.get_player_location(game.get_opponent(player))))
    num_blank_spaces = len(game.get_blank_spaces())
    h_knights_tour = 100 * h_knights_tour / max(w1 * num_blank_spaces, abs(-w2 * num_blank_spaces))  # Normalized heuristic
    
    w1, w2 = (1, 1)
#    logging.info("Heuristic scores: legal moves: %s and knight's tour: %s." % (w1 * h_num_legal_moves, w2 * h_knights_tour))
    return w1 * h_num_legal_moves + w2 * h_knights_tour
    # Strategies:
        # play with different weights
        # use different heuristics in different game stages
        # use different heuristics in different scenarios

def reach(game, player, num_moves=2):
    all_moves = set([m for move in game.get_legal_moves(player) \
                     for m in game.__get_moves__(move) \
                     if game.move_is_legal(move)])    
    blank_spaces = set(game.get_blank_spaces())

    moves = all_moves | blank_spaces
    
    return len(moves)

def greedy_knights_tour(n, game, pos):
    """
    Try finding a path by visiting only the neighbour with the smallest num of
    possible moves (NOT A SEARCH PROBLEM BUT FASTER)
    """ 
    while True:
        neighbours = game.__get_moves__(pos)
#        print("Neighbours: %s at pos %s" %(neighbours, pos))
        if len(neighbours) == 0:
            return n
        else:
            best_nb = None
            min_moves = 9
            for nb in neighbours:
                num_moves = len(game.__get_moves__(nb))
                if num_moves < min_moves:
                    min_moves = num_moves
                    best_nb = nb
            if best_nb:
                pos = best_nb
                n += 1
#                print("Found best nb: %s" % (str(best_nb)))
                game.__board_state__[best_nb[0]][best_nb[1]] = 'KT' + str(n)
#                print(game.to_string())
            else:
                print("ERROR")
                raise("WTF")

def thorough_knights_tour(n, game, pos):
    """
    game.__board_state__ = list of lists containing '|', ' ', '1', '2' or '-'
    
    steps or SEARCH problem (keep looking until a path is found)
    # Get neighbours
    # Get num of possible moves for each NB
    # sort the NB list
    # repeat with first NB
    
    """
    neighbours = game.__get_moves__(pos)
    # current depth is the longest tour for this path since no more legal moves
    if len(neighbours) == 0:
        return n

    # list num_moves for all legal neighbours and sort by num_moves
    sorted_neighbours = []
    for neighbour in neighbours:
        num_moves = len(game.__get_moves__(neighbour))
        sorted_neighbours.append((neighbour, num_moves))
    sorted_neighbours.sort(key=lambda x: x[1])
    
    # visit legal neighbours
    for neighbour in sorted_neighbours:
        game.__board_state__[neighbour[0][0]][neighbour[0][1]] = 'KT' + str(n)
        return thorough_knights_tour(n+1, game, neighbour[0])

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.transposition_table = {}
        self.depths = []

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if not legal_moves:
            return (-1, -1)
        score, best_move = (0, (-1, -1))
        
        if self.method is 'minimax': search = self.minimax
        elif self.method is 'alphabeta': search = self.alphabeta
        
        # TODO: opening book
        
        # initiate empty table here. The table is of the form:
        # dict(key: unique hashed number, value: tuple(moves_sorted_by_'goodness'))
        self.transposition_table = defaultdict(tuple)
        
        # For testing. Whether to use transposition or not is toggled here:
        transposition = True
#        transposition = False
        
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                depth = 1
                while True:
                    # Try to go forever, the timeout will stop us
                    score, best_move = search(game, depth, transposition)
                    if score == float("-inf") or score == float("+inf"):
                        break
                    depth += 1
                    
            else:
                score, best_move = search(game, self.search_depth, transposition)
        
        except Timeout:
            # Handle any actions required at timeout, if necessary
            if self.iterative: self.depths.append(depth)
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, transposition=False, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: this can be optimized, doesn't need to be calculated if the board state is
        # in the transposition table
        legal_moves = game.get_legal_moves()
        
        if depth == 0 or not legal_moves:
            return self.score(game, self), (-1, -1)
        
        if transposition:
            # Create the hash key for this game state
            game_hash = game.to_string()       
            # Create dict with all moves for this node sorted from good to bad
            moves = list()
            if self.transposition_table[game_hash]:
                selected_moves = [m for m, v in self.transposition_table[game_hash]]
            else:
                selected_moves = legal_moves
        else:
            selected_moves = legal_moves
        
        if maximizing_player:
            max_v = float("-inf")
            best_move = None
            
            for m in selected_moves:
                v, _ = self.minimax(game.forecast_move(m), depth - 1, transposition)
                if v > max_v:
                    max_v = v
                    best_move = m
                if transposition:
                    #Add move to dict
                    moves.append((m, v))
            if transposition:
                # Update transposition table with sorted list of moves (good -> bad)
                moves.sort(key=lambda x: x[1])
                self.transposition_table[game_hash] = moves
            
            return max_v, best_move
        else:
            min_v = float("+inf")
            for m in selected_moves:
                v, _ =self.minimax(game.forecast_move(m), depth - 1, transposition)
                if v < min_v:
                    min_v = v
                    best_move = m
                if transposition:
                    #Add move to dict
                    moves.append((m, v))
            if transposition:
                # Update transposition table with sorted list of moves (good -> bad)
                moves.sort(key=lambda x: x[1])
                self.transposition_table[game_hash] = moves
            
            return min_v, best_move

    def alphabeta(self, game, depth, transposition=False, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()
        
        if depth == 0 or not legal_moves:
            return self.score(game, self), (-1, -1)
        
        if transposition:
            # Create the hash key for this game state
            game_hash = game.to_string()       
            # Create dict with all moves for this node sorted from good to bad
            moves = list()
            if self.transposition_table[game_hash]:
                #logging.info("TRANSPOSITION!")
                selected_moves = [m for m, v in self.transposition_table[game_hash]]
            else:
                selected_moves = legal_moves
        else:
            selected_moves = legal_moves
        
        if maximizing_player:
            max_v = float("-inf")
            best_move = None
            for m in selected_moves:
                v, _ = self.alphabeta(game.forecast_move(m), depth-1, transposition, alpha, beta, False)
                if v > max_v:
                    max_v = v
                    best_move = m
                if transposition:
                    #Add move to dict
                    moves.append((m, v))
                if v >= beta:
                    return v, m
                alpha = max(v, alpha)
            if transposition:
                # Update transposition table with sorted list of moves (good -> bad)
                moves.sort(key=lambda x: x[1])
                self.transposition_table[game_hash] = moves
            return max_v, best_move
        else:
            min_v = float("+inf")
            best_move = None
            for m in selected_moves:
                v, _ = self.alphabeta(game.forecast_move(m), depth-1, transposition, alpha, beta, True)
                if v < min_v:
                    min_v = v
                    best_move = m
                if transposition:
                    #Add move to dict
                    moves.append((m, v))
                if v <= alpha:
                    return v, m
                beta = min(v, beta)
            if transposition:
                # Update transposition table with sorted list of moves (good -> bad)
                moves.sort(key=lambda x: x[1])
                self.transposition_table[game_hash] = moves
            return min_v, best_move