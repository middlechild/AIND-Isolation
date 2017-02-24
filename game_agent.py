"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def heuristic_one(game, player):
    """The "heuristic_one" evaluation function outputs a score equal to
    the difference in the number of moves available to the player and
    two times the number of moves available to the opponent.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - (2 * opp_moves))


def heuristic_two(game, player):
    """The "heuristic_two" evaluation function outputs a score equal to
    the difference in the number of moves available to the player and
    the weighted number of moves available to the opponent. In this heuristic,
    the additional weight applied is greater at the beginning of the game and
    diminishes as the game progresses.


    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    max_blank_spaces = 46
    blank_spaces = len(game.get_blank_spaces())

    return float(own_moves - ((2 + (blank_spaces/max_blank_spaces)) * opp_moves))


def heuristic_three(game, player):
    """The "heuristic_three" evaluation function outputs a score equal to
    the difference in the number of moves available to the two players in the
    next two rounds, with weight added to the opponent's moves for more
    aggressive game play. In this heuristic, the additional weight applied is
    greater at the beginning of the game and diminishes as the game progresses.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    max_blank_spaces = 46
    blank_spaces = len(game.get_blank_spaces())

    for move in game.get_legal_moves(player):
        own_moves += len(game.__get_moves__(move))

    for move in game.get_legal_moves(game.get_opponent(player)):
        opp_moves += len(game.__get_moves__(move))

    return float(own_moves - ((2 + (blank_spaces/max_blank_spaces)) * opp_moves))


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

    return heuristic_three(game, player)

    raise NotImplementedError


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
        depth of one (1) would only explore the immediate successors of the
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
                 iterative=True, method='minimax', timeout=15.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.count = 0

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

        # Have something ready to be returned in case of timeout
        current_score, current_move = float("-inf"), legal_moves[0]

        # Just for clarity
        argmax = max

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            if self.method == 'minimax':
                if self.iterative is True:
                    self.search_depth = 1
                    while current_score is not float("inf"):
                        current_score, current_move = argmax(self.minimax(game, self.search_depth), (current_score, current_move))
                        self.search_depth += 1
                else:
                    current_score, current_move = argmax(self.minimax(game, self.search_depth), (current_score, current_move))

            elif self.method == 'alphabeta':
                if self.iterative is True:
                    self.search_depth = 1
                    while current_score is not float("inf"):
                        current_score, current_move = argmax(self.alphabeta(game, self.search_depth), (current_score, current_move))
                        self.search_depth += 1
                else:
                    current_score, current_move = argmax(self.alphabeta(game, self.search_depth), (current_score, current_move))

            pass

        except Timeout:
            # Handle any actions required at timeout
            # logging.warning('TIMEOUT - Result Returned: %s', current_move)
            return current_move
            pass

        # Return the best move from the last completed search iteration
        return current_move

        raise NotImplementedError

    def minimax(self, game, depth, maximizing_player=True):
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

        best_move = (-1, -1)

        if depth == 0:
            return self.score(game, self), best_move

        if maximizing_player is True:
            # Checking for time left here allow us to do it more often and avoid timeout issues
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            best_value = float("-inf")
            for move in game.get_legal_moves():
                value, _ = self.minimax(game.forecast_move(move), depth-1, False)
                best_value, best_move = max((best_value, best_move), (value, move))
        else:
            # Checking for time left here allow us to do it more often and avoid timeout issues
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            best_value = float("inf")
            for move in game.get_legal_moves():
                value, _ = self.minimax(game.forecast_move(move), depth-1, True)
                best_value, best_move = min((best_value, best_move), (value, move))

        return best_value, best_move

        raise NotImplementedError

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
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

        # Function to find best score & move for Max player
        def max_value(self, game, depth, alpha, beta):
            # Checking for time left inside this function allow us to do it more often and avoid timeout issues
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            if depth == 0:
                return self.score(game, self), (-1, -1)

            best_score, best_move = float("-inf"), (-1, -1)

            for move in game.get_legal_moves():
                score, _ = min_value(self, game.forecast_move(move), depth-1, alpha, beta)
                best_score, best_move = max((best_score, best_move), (score, move))
                if best_score >= beta:
                    return best_score, best_move
                alpha = max(alpha, best_score)

            return best_score, best_move

        # Function to find best score & move for Min player
        def min_value(self, game, depth, alpha, beta):
            # Checking for time left inside this function allow us to do it more often and avoid timeout issues
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            if depth == 0:
                return self.score(game, self), (-1, -1)

            best_score, best_move = float("inf"), (-1, -1)

            for move in game.get_legal_moves():
                score, _ = max_value(self, game.forecast_move(move), depth-1, alpha, beta)
                best_score, best_move = min((best_score, best_move), (score, move))
                if best_score <= alpha:
                    return best_score, best_move
                beta = min(beta, best_score)

            return best_score, best_move

        # Start by calling the appropriate function based on the "maximizing_player" parameter
        if maximizing_player is True:
            return max_value(self, game, depth, alpha, beta)
        else:
            return min_value(self, game, depth, alpha, beta)

        raise NotImplementedError
