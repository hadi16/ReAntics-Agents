from AIPlayerUtils import *
from GameState import GameState
from Player import *

import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple


class Items:
    """
    Items
    Helper class that abstracts away GameState calls for the game.

    Author: Alex Hadi
    Version: December 7, 2018
    """

    def __init__(self, current_state: GameState):
        """
        __init__
        Creates a new Items object from a GameState object.
        :param current_state: The current state of the game.
        """

        self._current_state = current_state
        self._my_inventory = getCurrPlayerInventory(current_state)
        self._me = current_state.whoseTurn

    @property
    def my_queen(self) -> Ant:
        """
        my_queen
        :return: My queen.
        """

        return getCurrPlayerQueen(self._current_state)

    @property
    def my_ants(self) -> List[Ant]:
        """
        my_ants
        :return: All of my ants.
        """

        return self._my_inventory.ants

    @property
    def enemy_queen(self) -> Ant:
        """
        enemy_queen
        :return: The enemy's queen.
        """

        return getAntList(self._current_state, 1 - self._me, (QUEEN,))[0]

    @property
    def all_food(self) -> List[Construction]:
        """
        all_food
        :return: All of the food as a list.
        """

        return getConstrList(self._current_state, None, (FOOD,))

    @property
    def my_food_count(self) -> int:
        """
        my_food_count
        :return: My food count.
        """

        return self._my_inventory.foodCount

    @property
    def my_workers(self) -> List[Ant]:
        """
        my_workers
        :return: All of my workers as a list.
        """

        return getAntList(self._current_state, self._me, (WORKER,))

    @property
    def my_constructions(self) -> List[Construction]:
        """
        my_constructions
        :return: My anthill and tunnel as a list.
        """

        return getConstrList(self._current_state, self._me, (ANTHILL, TUNNEL))

    @property
    def has_won(self) -> Optional[bool]:
        """
        has_won
        :return: True if I have won, False if opponent has won. Otherwise, False.
        """

        win_status = getWinner(self._current_state)
        if win_status is None:
            return None
        return bool(win_status)


class ConsolidatedState:
    """
    ConsolidatedState
    The consolidated representation of a given state in the game.

    Author: Alex Hadi
    Version: December 7, 2018
    """

    def __init__(self, items: Items = None, json_dict: dict = None):
        """
        __init__
        Creates a new ConsolidatedState, either from the loaded JSON file or a given GameState.

        :param items: The Items object that abstracts away calls to certain game features.
        :param json_dict: The dictionary representing a valid object in JSON format.
        """

        # Create a new consolidated state from a state in the game.
        if items:
            self.carrying_workers_min_dist = self._carrying_workers_min_dist(items)
            self.not_carrying_workers_min_dist = self._not_carrying_workers_min_dist(items)
            self.my_food_count = items.my_food_count
            self.my_worker_count = len(items.my_workers)
            self.my_queen_on_construction: bool = self._my_queen_on_construction(items)

        '''
        Create a new consolidated state from the loaded JSON file.
        Citation:
        https://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
        '''
        if json_dict:
            for key in json_dict:
                setattr(self, key, json_dict[key])

    def __eq__(self, other) -> bool:
        """
        __eq__
        Overrides the equality comparison for the class.
        Overridden so ConsolidatedState can be used as keys in a dictionary.

        :param other: The other object to compare it to.
        :return: True (equal), otherwise False.
        """

        if isinstance(other, ConsolidatedState):
            return self.__dict__ == other.__dict__
        return False

    def __hash__(self):
        """
        __hash__
        Overrides the hash operation for the class.
        Overridden so ConsolidatedState can be used as keys in a dictionary.

        Citation:
        https://stackoverflow.com/questions/390250/
        elegant-ways-to-support-equivalence-equality-in-python-classes
        """
        return hash(tuple(sorted(self.__dict__.items())))

    def _my_queen_on_construction(self, items: Items) -> bool:
        """
        _my_queen_on_construction
        Determines whether my queen is on my anthill or my tunnel.

        :param items: The current items in the game (abstracts away GameState calls).
        :return: True if queen on my anthill or tunnel (otherwise False).
        """

        return any(
            items.my_queen.coords == construction.coords
            for construction in items.my_constructions
        )

    def _carrying_workers_min_dist(self, items: Items) -> int:
        """
        _carrying_workers_min_dist
        Finds the minimum distance that any carrying worker is to the anthill or tunnel.

        :param items: The current items in the game (abstracts away GameState calls).
        :return: The minimum distance to the anthill or tunnel as an integer.
        """

        carrying_workers = [
            worker for worker in items.my_workers if worker.carrying
        ]
        # -1 deliberately chosen, as it is not a valid distance for any worker.
        if not carrying_workers:
            return -1

        return min(
            approxDist(worker.coords, construction.coords)
            for construction in items.my_constructions
            for worker in carrying_workers
        )

    def _not_carrying_workers_min_dist(self, items: Items) -> int:
        """
        _not_carrying_workers_min_dist
        Finds the minimum distance that any non-carrying worker is to any food.

        :param items: The current items in the game (abstracts away GameState calls).
        :return: The minimum distance to any food as an integer.
        """

        not_carrying_workers = [
            worker for worker in items.my_workers if not worker.carrying
        ]
        # -1 deliberately chosen, as it is not a valid distance for any worker.
        if not not_carrying_workers:
            return -1

        return min(
            approxDist(worker.coords, food.coords)
            for food in items.all_food
            for worker in not_carrying_workers
        )


class AIPlayer(Player):
    """
    AIPlayer
    The TD Learning agent for CS 421 HW #6.

    Author: Alex Hadi
    Version: December 7, 2018
    """

    def __init__(self, input_player_id: int):
        """
        __init__
        Creates the agent.

        :param input_player_id: The ID to initialize the agent to.
        """

        super(AIPlayer, self).__init__(input_player_id, "Ted")

        # Constants
        self.DISCOUNT_FACTOR = 0.90
        self.MAX_NUM_ANTS = 8  # Prevents the agent from spawning too many ants (similar to Random)
        self.TRAINING = False
        self.EXPLOITATION_PROBABILITY = 0.67 if self.TRAINING else 1.00
        self.OUTPUT_JSON_FILE = 'state_utilities_hadi20.json'  # The output data file.

        self.games_played = 0
        self.learning_rate: float = 1.00
        self.state_utilities: Dict[ConsolidatedState, float] = self.load_states_from_file()

    def getPlacement(self, current_state: GameState) -> List[Tuple[int, int]]:
        """
        getPlacement
        Called during setup phase for each Construction that must be placed by the player.
        These items are:
        1 Anthill on the player's side, 1 tunnel on player's side, 9 grass on the player's side,
        and 2 food on the enemy's side.

        :param current_state: the state of the game at this point in time.
        :return: The coordinates of where the construction is to be placed.
        """

        # Get the placements for my anthill, tunnel, and grass.
        if current_state.phase == SETUP_PHASE_1:
            return [
                (0, 0), (5, 1), (0, 3), (1, 2), (2, 1), (3, 0),
                (0, 2), (1, 1), (2, 0), (0, 1), (1, 0)
            ]

        # Get the placements for the enemy's food.
        elif current_state.phase == SETUP_PHASE_2:
            moves = []
            for i in range(2):
                move = None
                while move is None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    # Set the move if this space is empty
                    if current_state.board[x][y].constr is None and (x, y) not in moves:
                        move = (x, y)
                moves.append(move)
            return moves

    def load_states_from_file(self) -> dict:
        """
        load_states_from_file
        If the data file exists in the AI/ folder (same directory as this file), load them.

        :return: The dictionary representing each consolidated state and its given utility.
        """

        if not os.path.exists(self.OUTPUT_JSON_FILE):
            return {}

        with open(self.OUTPUT_JSON_FILE, 'r') as file:
            # Load JSON file & initialize ConsolidatedState using json_dict constructor parameter.
            json_list = json.load(file)
            return {
                ConsolidatedState(json_dict=state_utility['state']): state_utility['utility']
                for state_utility in json_list
            }

    def save_states_to_file(self) -> None:
        """
        save_states_to_file
        Saves all the states in self.state_utilities to a JSON file.
        """

        # To make the data file be in the proper folder
        # (the current working directory is ReAntics/ at this point).
        with open('AI/' + self.OUTPUT_JSON_FILE, 'w') as output_file:
            json.dump([
                {
                    # Use the dictionary representation of each consolidated state.
                    "state": consolidated_state.__dict__,
                    "utility": state_utility
                }
                for consolidated_state, state_utility in self.state_utilities.items()
            ], output_file, indent=4)

    def get_move_with_highest_utility(self, current_state: GameState) -> Move:
        """
        get_move_with_highest_utility
        Finds the move that will result in the highest utility.
        Used when the agent exploits its knowledge (rather than explore).

        :param current_state: The current GameState.
        :return: The best Move given the utilities in self.state_utilities.
        """

        legal_moves = listAllLegalMoves(current_state)

        # To prevent the agent from spawning too many ants
        # (similar to how Random prevents this from happening).
        num_my_ants = len(Items(current_state).my_ants)
        if num_my_ants >= self.MAX_NUM_ANTS:
            legal_moves = [
                move for move in legal_moves if move.moveType != BUILD
            ]

        all_utilities = {}
        for move in legal_moves:
            # Finds the next state based on this move and update its utility.
            next_state = getNextState(current_state, move)
            consolidated_next_state = ConsolidatedState(items=Items(next_state))

            all_utilities[move] = self.add_simple_state_and_return_utility(
                current_state, consolidated_next_state
            )

        # Checks to make sure at least one element is in all_utilities
        if all_utilities:
            return max(all_utilities, key=all_utilities.get)

    def add_simple_state_and_return_utility(self, current_state: GameState,
                                            next_simple_state: ConsolidatedState = None) -> float:
        """
        add_simple_state_and_return_utility
        Updates the utility of the current_state (as an equivalent ConsolidatedState object)
        and returns the utility for this state.

        :param current_state: The current GameState.
        :param next_simple_state: The next ConsolidatedState that will be reached.
        :return: The utility of either the current state or the next state.
        """

        current_state_items = Items(current_state)
        current_simple_state = ConsolidatedState(items=current_state_items)

        # If no next_simple_state was passed,
        # return the current utility of current_simple_state (or just 0.0)
        if not next_simple_state:
            if current_simple_state in self.state_utilities:
                return self.state_utilities[current_simple_state]
            else:
                self.state_utilities[current_simple_state] = 0.0
                return self.state_utilities[current_simple_state]

        # If next_simple_state doesn't have a utility assigned to it,
        # set it to 0.0 and return it.
        if next_simple_state not in self.state_utilities:
            self.state_utilities[next_simple_state] = 0.0
            return self.state_utilities[next_simple_state]

        # Determines the reward of the current state based on
        # if the current_state is winning or not.
        current_state_has_won = current_state_items.has_won
        if current_state_has_won is None:
            reward = -0.01
        else:
            reward = 1.0 if current_state_has_won else -1.0

        # Uses the TD Learning equation to set the utility of the current state & return it.
        self.state_utilities[current_simple_state] += self.learning_rate * (
                reward + (
                    self.DISCOUNT_FACTOR * self.state_utilities[next_simple_state]
                ) - self.state_utilities[current_simple_state]
        )
        return self.state_utilities[current_simple_state]

    def getMove(self, current_state: GameState) -> Move:
        """
        getMove
        Gets the next move from the Player.

        :param current_state: The state of the current game waiting for the player's move.
        :return: The move to be made.
        """

        # Either returns the current utility of current_state or sets it to 0.0 and returns it.
        self.add_simple_state_and_return_utility(current_state)

        best_move = None
        # If the agent randomly chooses to exploit,
        # find the move that results in the highest-utility state.
        if random.random() < self.EXPLOITATION_PROBABILITY:
            best_move = self.get_move_with_highest_utility(current_state)
        # If the agent didn't exploit, randomly choose a move.
        if not best_move:
            legal_moves = listAllLegalMoves(current_state)

            # Prevents the agent from spawning too many ants.
            num_my_ants = len(Items(current_state).my_ants)
            if num_my_ants >= self.MAX_NUM_ANTS:
                legal_moves = [
                    move for move in legal_moves if move.moveType != BUILD
                ]

            # Get one of these legal moves at random.
            best_move = random.choice(legal_moves)

        return best_move

    def getAttack(self, current_state: GameState, attacking_ant: Ant,
                  enemy_locations: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        getAttack
        Gets the attack to be made from the Player.

        :param current_state: The current state of the game.
        :param attacking_ant: The ant currently making the attack.
        :param enemy_locations: The locations of the enemies that can be attacked.
        :return: The location of where to attack (a tuple of two integers).
        """

        return enemy_locations[0]

    def registerWin(self, has_won: bool) -> None:
        """
        registerWin
        When the agent has won or lost a game, update the learning rate.
        Also save the state utilities to the JSON file every 50 games.

        :param has_won: True if the agent has won (otherwise False).
        """

        self.games_played += 1
        # Learning rate is incremented by an increasing negative rate exponential equation.
        self.learning_rate += .1 / math.exp(self.learning_rate ** 4) - .1

        # Only write to the file if still training the agent.
        if self.TRAINING and self.games_played % 50 == 0:
            self.save_states_to_file()
