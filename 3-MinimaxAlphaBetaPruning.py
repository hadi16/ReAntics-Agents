from AIPlayerUtils import *
from Constants import *
from GameState import GameState
from Player import Player
from math import inf
from typing import Dict, List


class Node:
    def __init__(self, move: Move, state: GameState, state_evaluation: float, parent_node=None):
        """
        Node

        Class that represents a single node in the search tree.

        :param move: The move that is taken from the parent node to the current node.
        :param state: The resulting state of the move.
        :param state_evaluation: The state evaluation score for the node.
        :param parent_node: The parent node of the node.
        """
        self.move = move
        self.state = state
        self.state_evaluation = state_evaluation
        self.parent_node = parent_node


class AIPlayer(Player):
    """
    Class: AIPlayer
    The Minimax and Alpha Beta Pruning Agent for CS 421.

    Authors: Alex Hadi and Cole French
    Version: October 7, 2018
    """

    def __init__(self, input_player_id: int):
        """
        __init__

        The constructor for AIPlayer (creates a new player).

        :param input_player_id: The player's ID as an integer.
        """
        super(AIPlayer, self).__init__(input_player_id, "MinimaxAlphaBetaPruning")

    def getPlacement(self, current_state: GameState) -> List[tuple]:
        """
        Called during the setup phase for each Construction that must be placed by the player.
        These items are: 1 Anthill on the player's side; 1 tunnel on player's side; 9 grass on the
        player's side; and 2 food on the enemy's side.

        :param current_state: The state of the game at this point in time.
        :return: The coordinates of where the construction items should be placed.
        """

        # implemented by students to return their next move
        if current_state.phase == SETUP_PHASE_1:    # stuff on my side
            num_to_place = 11
            moves = []
            for i in range(0, num_to_place):
                move = None
                while move is None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    # Set the move if this space is empty
                    if current_state.board[x][y].constr is None and (x, y) not in moves:
                        move = (x, y)
                moves.append(move)
            return moves
        elif current_state.phase == SETUP_PHASE_2:   # stuff on foe's side
            num_to_place = 2
            moves = []
            for i in range(0, num_to_place):
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
        else:
            return [(0, 0)]

    def getMove(self, current_state: GameState) -> Move:
        """
        getMove

        Gets the next move from the player. The search tree is used to make this decision.

        :param current_state: The state of the current game (GameState).
        :return: The move to be made.
        """
        root_node = Node(None, current_state, 0.0, None)
        return self.find_best_move_minimax(root_node, 0, -inf, inf, True)

    def getAttack(self, current_state: GameState, attacking_ant: Ant, enemy_locations):
        """
        getAttack

        Gets the attack to be made from the player.
        Just attacks a random enemy.

        :param current_state: A clone of the current state (GameState).
        :param attacking_ant: The ant currently making the attack (Ant).
        :param enemy_locations: The locations of the enemies that can be attacked
        :return:
        """
        return enemy_locations[random.randint(0, len(enemy_locations) - 1)]

    def registerWin(self, has_won):
        """
        registerWin

        This agent doesn't learn.

        :param has_won: Whether the agent has won or not.
        """
        pass

    def _evaluate_worker_count(self, current_state: GameState) -> float:
        """
        _evaluate_worker_count

        Evaluation function for the number of workers that the AI controls.

        :param current_state: The current GameState.
        :return: The evaluation score for the worker count.
        """
        worker_count = len(getAntList(current_state, current_state.whoseTurn, (WORKER,)))
        if worker_count > 1:
            return -1.0
        else:
            return worker_count

    def _evaluate_soldier_count(self, current_state: GameState) -> float:
        """
        _evaluate_soldier_count

        Evaluation function for the number of soldier that the AI controls.

        :param current_state: The current GameState
        :return: The evaluation score for the soldier count.
        """
        soldier_count = len(getAntList(current_state, current_state.whoseTurn, (SOLDIER,)))
        # Reward for having more than 10 soldiers
        if soldier_count > 10:
            return 1.0
        else:
            return 0.1 * soldier_count

    def _evaluate_ant_difference(self, current_state: GameState) -> float:
        """
        _evaluate_ant_difference

        Evaluation function for the difference in ants between the player and opponent.

        :param current_state: The current GameState.
        :return: The evaluation score for the ant difference.
        """
        my_ant_count = len(getAntList(current_state, current_state.whoseTurn))
        enemy_ant_count = len(getAntList(current_state, 1 - current_state.whoseTurn))

        # Evaluation score is the ratio of our AI's ants vs the total number of ants on the board
        return (my_ant_count - enemy_ant_count) / (my_ant_count + enemy_ant_count)

    def _evaluate_health_difference(self, current_state: GameState) -> float:
        """
        _evaluate_health_difference

        Evaluation function for the difference in health of the ants between player and opponent.

        :param current_state: The current GameState.
        :return: The health difference evaluation score.
        """
        my_ants = getAntList(current_state, current_state.whoseTurn)
        my_total_health = sum(ant.health for ant in my_ants)

        enemy_ants = getAntList(current_state, 1 - current_state.whoseTurn)
        enemy_total_health = sum(ant.health for ant in enemy_ants)

        return (my_total_health - enemy_total_health) / (my_total_health + enemy_total_health)

    def _evaluate_worker_positions(self, current_state: GameState) -> float:
        """
        _evaluate_worker_positions

        Evaluation function for the position of the worker.
        Rewards AI for collection food and bring the food back to the anthill/tunnel.

        :param current_state: The current GameState.
        :return: The worker position evaluation score.
        """
        me = current_state.whoseTurn
        my_workers = getAntList(current_state, me, (WORKER,))
        if not my_workers:
            return -1.0

        # 16 steps is around the furthest distance one worker could theoretically be
        # from a food source. The actual step amounts should never be close to this number.
        MAX_STEPS_FROM_FOOD = 16

        my_anthill_and_tunnel = getConstrList(current_state, me, (ANTHILL, TUNNEL))
        building_coords = self._get_coordinate_list_of_game_elements(my_anthill_and_tunnel)

        food_list = getConstrList(current_state, None, (FOOD,))
        food_coords = self._get_coordinate_list_of_game_elements(food_list)

        # Calculate the total steps each worker is from its nearest destination.
        total_steps_to_dest = 0
        for worker in my_workers:
            if worker.carrying:
                total_steps_to_dest += self._min_steps_to_target(worker.coords, building_coords)
            else:
                steps = self._min_steps_to_target(worker.coords, food_coords)
                total_steps_to_dest += steps + MAX_STEPS_FROM_FOOD

        my_inv = getCurrPlayerInventory(current_state)
        total_steps_to_dest += (11 - my_inv.foodCount) * 2 * MAX_STEPS_FROM_FOOD * len(my_workers)
        score_ceiling = 12 * 2 * MAX_STEPS_FROM_FOOD * len(my_workers)
        eval_score = score_ceiling - total_steps_to_dest

        # Max possible score is 1.0, where all workers are at their destination.
        return eval_score / score_ceiling

    def _evaluate_soldier_positions(self, current_state: GameState) -> float:
        """
        _evaluate_soldier_positions

        Evaluation function for the position of the soldier.
        Rewards for being closer to enemy ants resulting in attack.

        :param current_state: The current GameState.
        :return: The soldier position evaluation score.
        """
        me = current_state.whoseTurn
        my_soldiers = getAntList(current_state, me, (SOLDIER,))
        if not my_soldiers:
            return 0.0

        # Save the coordinates of all the enemy's ants.
        enemy_ants = getAntList(current_state, 1 - me)
        enemy_ant_coords = self._get_coordinate_list_of_game_elements(enemy_ants)

        total_steps_to_enemy = 0
        for soldier in my_soldiers:
            total_steps_to_enemy += self._min_steps_to_target(soldier.coords, enemy_ant_coords)

        # 30 steps is around the furthest distance one soldier could theoretically be
        # from an enemy ant. The actual step amounts should never be close to this number.
        MAX_STEPS_FROM_ENEMY = 30
        score_ceiling = MAX_STEPS_FROM_ENEMY * len(my_soldiers)
        eval_score = score_ceiling - total_steps_to_enemy

        # Max possible score is 1.0, where all soldiers are at their destination.
        return eval_score / score_ceiling

    def _evaluate_queen_position(self, current_state: GameState) -> float:
        """
        _evaluate_queen_position

        Evaluation function for the position of the queen.
        Rewards AI for moving away from the closest enemy and not on the anthill/tunnel.

        :param current_state: The current GameState.
        :return: The evaluation score for the queen position.
        """
        me = current_state.whoseTurn
        queen = getCurrPlayerQueen(current_state)
        enemy_fighters = getAntList(current_state, 1 - me, (DRONE, SOLDIER, R_SOLDIER))
        enemy_ant_coords = self._get_coordinate_list_of_game_elements(enemy_fighters)

        total_distance = sum(approxDist(queen.coords, coords) for coords in enemy_ant_coords)

        if enemy_ant_coords:
            MAX_STEPS_FROM_ENEMY = 30
            score_ceiling = MAX_STEPS_FROM_ENEMY * len(enemy_ant_coords)
            buildings = getConstrList(current_state, me, (ANTHILL, TUNNEL, FOOD))
            if queen.coords in self._get_coordinate_list_of_game_elements(buildings):
                return -1.0
            return total_distance / score_ceiling
        else:
            buildings = getConstrList(current_state, me, (ANTHILL, TUNNEL, FOOD))
            if queen.coords in self._get_coordinate_list_of_game_elements(buildings):
                return -1.0
            return 1.0

    def _min_steps_to_target(self, target_coordinate: tuple, coordinates_list: List[tuple]) -> int:
        """
        _min_steps_to_target

        Helper function to get the minimum steps to the given target.

        :param target_coordinate: The target coordinate as a tuple.
        :param coordinates_list: The list of coordinates to search as a list of tuples.
        :return:
        """
        return min([approxDist(target_coordinate, coordinates) for coordinates in coordinates_list])

    def _get_coordinate_list_of_game_elements(self, element_list: list) -> List[tuple]:
        """
        _get_coordinate_list_of_game_elements

        Helper function to get the coordinate list of the given list of Ants or Constructions.

        :param element_list: The list of Ant or Construction objects.
        :return: The list of coordinates as a List of tuples.
        """
        return [element.coords for element in element_list]

    def evaluate_game_state(self, current_state):
        """
        evaluate_game_state

        Calls all of the evaluation scores and multiplies them by a weight.
        This allows the AI to fine tune the evaluation scores to better suit
        favorable moves and strategies.

        :param current_state: A clone of the current game state that will be evaluated.
        :return: A score between [-1.0, 1.0] such that + is good & - is bad for the current player.
        """
        # Determine if the game has ended and who won
        win_result = getWinner(current_state)
        if win_result == 1:
            return 1.0
        elif win_result == 0:
            return -1.0
        # else neither player has won this state.

        # All of the helpers for the evaluation function.
        eval_helpers_with_given_weights: Dict[function, int] = {
            self._evaluate_worker_count:        2,
            self._evaluate_soldier_count:       3,
            self._evaluate_ant_difference:      1,
            self._evaluate_health_difference:   2,
            self._evaluate_worker_positions:    1,
            self._evaluate_soldier_positions:   1,
            self._evaluate_queen_position:      1
        }

        # Determine evaluation scores multiplied by it weights
        total_score = 0
        for eval_helper_func, weight in eval_helpers_with_given_weights.items():
            total_score += eval_helper_func(current_state) * weight

        # OVERALL WEIGHTED AVERAGE
        # Takes the weighted average of all of the scores
        # Only the game ending scores should be 1 or -1.
        return 0.99 * total_score / sum(eval_helpers_with_given_weights.values())

    def _get_final_node_greedy(self, node: Node) -> Node:
        """
        _get_final_node_greedy                  <!-- RECURSIVE -->

        Finds the best node that has an END turn as its move.

        :param node: The Node to find an END move for.
        :return: The Node that has an END turn.
        """
        if node.move.moveType == END:
            return node

        # Create the children nodes and find the best node from this list.
        child_nodes = self._create_child_nodes_from_state(node.state)
        best_node = max(child_nodes, key=lambda x: x.state_evaluation)
        best_node.parent_node = node
        return self._get_final_node_greedy(best_node)

    def _create_child_nodes_from_state(self, current_state: GameState) -> List[Node]:
        """
        _create_child_nodes_from_state

        Gets all the legal children nodes from the given state.

        :param current_state: The current GameState.
        :return: The children nodes as a list of Node objects.
        """
        all_moves = listAllLegalMoves(current_state)

        # Create the list of child nodes.
        child_nodes: List[Node] = []
        for move in all_moves:
            next_state = self.get_next_state_adversarial(current_state, move)
            next_state_evaluation = self.evaluate_game_state(next_state)
            child_nodes.append(Node(move, next_state, next_state_evaluation))

        # Sorts in descending order based on the state evaluation.
        child_nodes = sorted(child_nodes, key=lambda x: x.state_evaluation, reverse=True)

        # Take the highest 2 scoring nodes.
        return child_nodes[:2]

    def find_best_move_minimax(self, current_node: Node, current_depth: int, alpha: float,
                               beta: float, my_turn: bool):
        """
        find_best_move_minimax                      <!-- RECURSIVE -->

        Recursive function that performs minimax and alpha beta pruning.

        :param current_node: The current Node being evaluated.
        :param current_depth: The current depth in the tree.
        :param alpha: The alpha value (minimum).
        :param beta: The beta value (maximum).
        :param my_turn: True if the maximizing player, otherwise False.
        :return: The best move (Move object).
        """
        DEPTH_LIMIT = 4
        if current_depth == DEPTH_LIMIT or abs(current_node.state_evaluation) == 1.0:
            score_multiplier = 1 if my_turn else -1
            return current_node.state_evaluation * score_multiplier

        # Get the children END moves.
        child_nodes = self._create_child_nodes_from_state(current_node.state)
        final_nodes = [self._get_final_node_greedy(node) for node in child_nodes]

        if current_depth == 0:
            for node in final_nodes:
                node.state_evaluation = self.find_best_move_minimax(node, 1, -inf, inf, False)

            best_node = max(final_nodes, key=lambda x: x.state_evaluation)
            # Go up tree to find first move that leads to move with a given end state.
            while best_node.parent_node is not None:
                best_node = best_node.parent_node
            return best_node.move
        else:
            # If it is my turn, want to maximize the score.
            if my_turn:
                max_eval = -inf
                for node in final_nodes:
                    score = self.find_best_move_minimax(node, current_depth + 1, alpha, beta, False)
                    max_eval = max(max_eval, score)
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break
                return max_eval
            # Otherwise, minimize the score.
            else:
                min_eval = inf
                for node in final_nodes:
                    score = self.find_best_move_minimax(node, current_depth + 1, alpha, beta, True)
                    min_eval = min(min_eval, score)
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
                return min_eval

    def get_next_state_adversarial(self, current_state: GameState, move):
        """
        get_next_state_adversarial

        Citation: Made modification that Nux suggested via email (copied from this email).

        This is the same as getNextState (above) except that it properly
        updates the hasMoved property on ants and the END move is processed correctly.

        :param current_state: A clone of the current state (GameState)
        :param move: The move that the agent would take (Move).
        :return: A clone of what the state would look like if the move was made.
        """
        # variables I will need
        next_state = getNextState(current_state, move)
        my_inventory = getCurrPlayerInventory(next_state)
        my_ants = my_inventory.ants

        # If an ant is moved update their coordinates and has moved
        if move.moveType == MOVE_ANT:
            # startingCoord = move.coordList[0]
            starting_coord = move.coordList[len(move.coordList) - 1]
            for ant in my_ants:
                if ant.coords == starting_coord:
                    ant.hasMoved = True
        elif move.moveType == END:
            for ant in my_ants:
                ant.hasMoved = False
            next_state.whoseTurn = 1 - current_state.whoseTurn
        return next_state
