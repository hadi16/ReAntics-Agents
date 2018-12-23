from AIPlayerUtils import *
from Constants import *
from GameState import GameState
from Inventory import Inventory
from Player import Player
from typing import Dict, List


class AIPlayer(Player):
    """
    Class: AIPlayer
    The Heuristic Search Agent for CS 421.

    Authors: Alex Hadi and Reeca Bardon
    Version: September 24, 2018
    """

    def __init__(self, input_player_id: int):
        """
        __init__

        The constructor for AIPlayer (creates a new player).

        :param input_player_id: The player's ID as an integer.
        """
        super(AIPlayer, self).__init__(input_player_id, "InformedSearch")

    def getPlacement(self, current_state):
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

    def getMove(self, current_state) -> Move:
        """
        getMove

        Gets the next move from the player. The search tree is used to make this decision.

        :param current_state: The state of the current game (GameState).
        :return: The move to be made.
        """
        return self.find_best_move(current_state, 0)

    def getAttack(self, current_state, attacking_ant, enemy_locations):
        """
        getAttack

        Gets the attack to be made from the player.
        Just attacks a random enemy.

        :param current_state: A clone of the current state (GameState).
        :param attacking_ant: The ant currently making the attack (Ant).
        :param enemy_locations: The locations of the enemies that can be attacked (Location[])
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

    def _has_unwanted_conditions(self, my_workers: List[Ant], my_drones: List[Ant],
                                 my_soldiers: List[Ant], my_r_soldiers: List[Ant]) -> bool:
        """
        Helper function that checks for unwanted conditions in the GameState.
        If these exist, my agent will evaluate the game state at -0.99.

        :param my_workers: My tunnel.
        :param my_drones: The list of my drones.
        :param my_soldiers: The list of my soldiers.
        :param my_r_soldiers: The list of my ranged soldiers.
        :return: True (unwanted condition exist), otherwise False.
        """

        # Want one worker no matter what.
        if len(my_workers) != 1:
            return True

        # Want no ranged soldiers or soldiers.
        # Also want only 1 drone (although 0 initially is okay).
        if my_r_soldiers or my_soldiers or len(my_drones) not in [0, 1]:
            return True
        return False

    def _gather_food(self, dist_rewards: Dict[int, float], current_state: GameState,
                     my_closest_food: Construction, my_workers: List[Ant],
                     my_anthill: Construction, my_tunnel: Construction) -> float:
        """
        _gather_food

        Helper function that rewards (or punishes) the agent if my workers get (or don't get) food.

        :param dist_rewards: The dictionary of rewards and punishments for given distances.
        :param current_state: The current GameState object.
        :param my_closest_food: The closest food to my anthill or tunnel.
        :param my_workers: The list of my workers.
        :param my_anthill: My anthill.
        :param my_tunnel: My tunnel.
        :return: The delta value for the evaluation score.
        """

        evaluation_score_delta = 0.0

        # To get the queen (and anything else besides the worker) off my anthill.
        ant_at_anthill = getAntAt(current_state, my_anthill.coords)
        if ant_at_anthill and ant_at_anthill.type != WORKER:
            evaluation_score_delta -= 1.00

        # The worker doesn't get rewarded or punished by default.
        DEFAULT_WORKER_REWARD = 0.00
        if my_workers:
            for worker in my_workers:
                # If the worker is carrying food, need to get to my anthill or tunnel.
                if worker.carrying:
                    dist_to_anthill = approxDist(worker.coords, my_anthill.coords)
                    dist_to_tunnel = approxDist(worker.coords, my_tunnel.coords)
                    min_construction_dist = min(dist_to_anthill, dist_to_tunnel)
                    evaluation_score_delta += dist_rewards.get(min_construction_dist,
                                                               DEFAULT_WORKER_REWARD)
                # If the worker is not carrying food, need to get to the food source.
                else:
                    dist_to_closest_food = approxDist(worker.coords, my_closest_food.coords)
                    evaluation_score_delta += dist_rewards.get(dist_to_closest_food,
                                                               DEFAULT_WORKER_REWARD)
        return evaluation_score_delta

    def _kill_enemy_workers(self, dist_rewards: Dict[int, float], my_drones: List[Ant],
                            enemy_workers: List[Ant]) -> float:
        """
        _kill_enemy_workers

        Helper function that rewards (or punishes) my agent for killing the enemy workers.

        :param dist_rewards: The dictionary of rewards and punishments for given distances.
        :param my_drones: The list of my drones.
        :param enemy_workers: The list of the enemy workers.
        :return: The delta value for the evaluation score.
        """

        evaluation_score_delta = 0.0

        # Punish the drone by default (otherwise use dist_rewards dictionary).
        DEFAULT_DRONE_REWARD = -0.60

        # Punish the agent if the enemy has one worker
        if len(enemy_workers) == 1:
            evaluation_score_delta -= 0.85

        # Rewards (or punishes) the agent for getting my drone close to the enemy worker.
        # Only cares if there is exactly one worker.
        if len(enemy_workers) == 1:
            for drone in my_drones:
                dist_to_worker = approxDist(drone.coords, enemy_workers[0].coords)
                evaluation_score_delta += dist_rewards.get(dist_to_worker, DEFAULT_DRONE_REWARD)
        return evaluation_score_delta

    def evaluate_game_state(self, current_state) -> float:
        """
        evaluate_game_state

        Given a game state, calculates an evaluation score between -1.0 and 1.0.
        The agent's main objectives are to gather food and kill the enemy worker.

        :param current_state: The current game state.
        :return: The evaluation score (float)
        """

        evaluation_score = 0.0

        # Get all the relevant items I need.
        items = Items(current_state)
        my_closest_food = items.my_closest_food
        my_anthill = items.my_anthill
        my_tunnel = items.my_tunnel
        my_workers = items.my_workers
        my_drones = items.my_drones
        my_soldiers = items.my_soldiers
        my_r_soldiers = items.my_r_soldiers
        enemy_workers = items.enemy_workers

        # All the distance costs for the drone and worker.
        dist_rewards: Dict[int, float] = {
            0: 0.70,
            1: 0.55,
            2: 0.40,
            3: 0.30,
            4: 0.20,
            5: 0.00,
            6: -0.15,
            7: -0.30,
            8: -0.40,
            9: -0.45
        }

        # Just return -0.99 if an unwanted condition exists in the GameState.
        if self._has_unwanted_conditions(my_workers, my_drones, my_soldiers, my_r_soldiers):
            return -0.99

        # Agent is rewarded for gathering food and killing the enemy workers.
        evaluation_score += self._gather_food(dist_rewards, current_state, my_closest_food,
                                              my_workers, my_anthill, my_tunnel)
        evaluation_score += self._kill_enemy_workers(dist_rewards, my_drones, enemy_workers)

        # Check if there is a winner
        winner = getWinner(current_state)
        if winner == 1:
            return 1.0
        elif winner == 0:
            return -1.0

        # Make sure to return valid number
        if evaluation_score >= 1.0:
            return 0.99
        elif evaluation_score <= -1.0:
            return -0.99
        return evaluation_score

    def find_best_move(self, current_state, current_depth):
        """
        find_best_move                      <!-- RECURSIVE -->

        The best move is found by recursively traversing the search tree.
        An average of the evaluation scores is used to determine an overall score.

        :param current_state: The current GameState.
        :param current_depth: The current depth level in the tree.
        :return: The Move that the agent wishes to perform.
        """

        DEPTH_LIMIT = 2
        all_legal_moves = listAllLegalMoves(current_state)
        all_nodes = []

        for move in all_legal_moves:
            # Ignore the END_TURN move.
            if move.moveType == "END_TURN":
                continue

            next_state_reached = self.getNextState(current_state, move)
            node = Node(move, next_state_reached, self.evaluate_game_state(next_state_reached))
            all_nodes.append(node)

        best_nodes = self._get_best_nodes(all_nodes)
        if current_depth < DEPTH_LIMIT:
            for i, node in enumerate(best_nodes):
                best_nodes[i].state_evaluation = self.find_best_move(node.state, current_depth + 1)

        if current_depth > 0:
            return self.average_evaluation_score(best_nodes)
        else:
            # Citation: https://stackoverflow.com/questions/13067615/
            # python-getting-the-max-value-of-y-from-a-list-of-objects
            return max(best_nodes, key=lambda x: x.state_evaluation).move

    def _get_best_nodes(self, nodes: list) -> list:
        """
        _get_best_nodes

        Helper function used for finding the best nodes to prune the tree properly.

        :param nodes: The list of nodes to check.
        :return: The best nodes (number determined by NUM_BEST_NODES constant).
        """
        NUM_BEST_NODES = 5
        sorted_nodes = sorted(nodes, key=lambda node: node.state_evaluation, reverse=True)
        return sorted_nodes[:NUM_BEST_NODES]

    def average_evaluation_score(self, nodes: list) -> float:
        """
        Helper method to determine the overall evaluation score of a list of nodes.
        The average method is used.

        :param nodes: The list of nodes to check.
        :return: The average evaluation score of all the checked nodes.
        """
        return sum(node.state_evaluation for node in nodes) / len(nodes)

    def getNextState(self, currentState, move):
        """
        Revised version of getNextState from AIPlayerUtils.
        Copied from Nux's email to the class.

        :param currentState: The current GameState.
        :param move: The move to be performed.
        :return: The next GameState from the specified move.
        """

        # variables I will need
        myGameState = currentState.fastclone()
        myInv = getCurrPlayerInventory(myGameState)
        me = myGameState.whoseTurn
        myAnts = myInv.ants
        myTunnels = myInv.getTunnels()
        myAntHill = myInv.getAnthill()

        # If enemy ant is on my anthill or tunnel update capture health
        ant = getAntAt(myGameState, myAntHill.coords)
        if ant is not None:
            if ant.player != me:
                myAntHill.captureHealth -= 1

        # If an ant is built update list of ants
        antTypes = [WORKER, DRONE, SOLDIER, R_SOLDIER]
        if move.moveType == BUILD:
            if move.buildType in antTypes:
                ant = Ant(myInv.getAnthill().coords, move.buildType, me)
                myInv.ants.append(ant)
                # Update food count depending on ant built
                if move.buildType == WORKER:
                    myInv.foodCount -= 1
                elif move.buildType == DRONE or move.buildType == R_SOLDIER:
                    myInv.foodCount -= 2
                elif move.buildType == SOLDIER:
                    myInv.foodCount -= 3
            # ants are no longer allowed to build tunnels, so this is an error
            elif move.buildType == TUNNEL:
                print("Attempted tunnel build in getNextState()")
                return currentState

        # If an ant is moved update their coordinates and has moved
        elif move.moveType == MOVE_ANT:
            newCoord = move.coordList[-1]
            startingCoord = move.coordList[0]
            for ant in myAnts:
                if ant.coords == startingCoord:
                    ant.coords = newCoord
                    # TODO: should this be set true? Design decision
                    ant.hasMoved = False
                    attackable = listAttackable(ant.coords, UNIT_STATS[ant.type][RANGE])
                    for coord in attackable:
                        foundAnt = getAntAt(myGameState, coord)
                        if foundAnt is not None:  # If ant is adjacent my ant
                            if foundAnt.player != me:  # if the ant is not me
                                foundAnt.health = foundAnt.health - UNIT_STATS[ant.type][
                                    ATTACK]  # attack
                                # If an enemy is attacked and loses all its health
                                # remove it from the other players
                                # inventory
                                if foundAnt.health <= 0:
                                    myGameState.inventories[1 - me].ants.remove(foundAnt)
                                # If attacked an ant already don't attack any more
                                break
        return myGameState


class Node:
    def __init__(self, move: Move, state: GameState, state_evaluation: float):
        """
        Node

        Class that represents a single node in the search tree.

        :param move: The move that is taken from the parent node to the current node.
        :param state: The resulting state of the move.
        :param state_evaluation: The state evaluation score for the node.
        """
        self.move = move
        self.state = state
        self.state_evaluation = state_evaluation


class Items:
    """
    Items

    Helper class that serves three primary purposes.
    First, it handles calls to getAntList, getConstrList, etc.
    so that the main AIPlayer class doesn't have to do this.
    Second, it provides type hints so that the main AIPlayer class doesn't get cluttered with them.
    Third, it handles the logic for getting the inventory and me/enemy,
    so these lines of code aren't repeated needlessly in the main AIPlayer class.
    """
    def __init__(self, current_state: GameState):
        """
        __init__

        Creates a new Items object.

        :param current_state: The current GameState.
        """
        self._current_state = current_state

        # I should either be 0 or 1 (enemy is just 1 or 0, respectively)
        self._me: int = current_state.whoseTurn
        self._enemy = 1 - current_state.whoseTurn

        self._my_inventory: Inventory = current_state.inventories[self._me]
        self._enemy_inventory: Inventory = current_state.inventories[self._enemy]

    @property
    def my_food_count(self) -> int:
        """
        my_food_count

        :return: The amount of food I currently have.
        """
        return self._my_inventory.foodCount

    @property
    def enemy_food_count(self) -> int:
        """
        enemy_food_count

        :return: The amount of food that the enemy currently has.
        """
        return self._enemy_inventory.foodCount

    @property
    def my_food(self) -> List[Construction]:
        return getConstrList(self._current_state, None, (FOOD,))

    @property
    def my_closest_food(self) -> Construction:
        """
        my_closest_food

        :return: My food that is the closest to my tunnel.
        """
        # Distance to food and the corresponding food.
        food_distances_dict: Dict[int, Construction] = {}
        foods = getConstrList(self._current_state, None, (FOOD,))
        for food in foods:
            # Want the food closest to either the tunnel or anthill.
            food_dist_to_tunnel = approxDist(self.my_tunnel.coords, food.coords)
            food_dist_to_anthill = approxDist(self.my_anthill.coords, food.coords)
            food_distances_dict[min(food_dist_to_anthill, food_dist_to_tunnel)] = food

        # Return the food that has the minimum cost to get to.
        return food_distances_dict[min(food_distances_dict)]

    @property
    def my_ants(self) -> List[Ant]:
        """
        my_ants

        :return: A list of all of my ants.
        """
        return getAntList(self._current_state, self._me)

    @property
    def my_workers(self) -> List[Ant]:
        """
        my_workers

        :return: A list of my workers.
        """
        return getAntList(self._current_state, self._me, (WORKER,))

    @property
    def my_drones(self) -> List[Ant]:
        """
        my_drones

        :return: A list of my drones.
        """
        return getAntList(self._current_state, self._me, (DRONE,))

    @property
    def my_soldiers(self) -> List[Ant]:
        """
        my_soldiers

        :return: A list of my soldiers.
        """
        return getAntList(self._current_state, self._me, (SOLDIER,))

    @property
    def my_r_soldiers(self) -> List[Ant]:
        """
        my_r_soldiers

        :return: A list of my ranged soldiers.
        """
        return getAntList(self._current_state, self._me, (R_SOLDIER,))

    @property
    def my_anthill(self) -> Construction:
        """
        my_anthill

        :return: My anthill from my inventory.
        """
        return self._my_inventory.getAnthill()

    @property
    def my_tunnel(self) -> Construction:
        """
        my_tunnel

        :return: My tunnel.
        """
        return getConstrList(self._current_state, self._me, (TUNNEL,))[0]

    @property
    def enemy_workers(self) -> List[Ant]:
        """
        enemy_workers

        :return: A list of the enemy's workers.
        """
        return getAntList(self._current_state, self._enemy, (WORKER,))


def create_test_game_state() -> GameState:
    """
    create_test_game_state

    Creates the test game state to use with the unit tests.
    :return: The test game state.
    """
    state = GameState.getBasicState()

    # Set the food counts
    state.inventories[0].foodCount = 3
    state.inventories[1].foodCount = 2

    # Set my grass in a row on top
    my_grass_list = [Construction((x, 0), GRASS) for x in range(0, 9)]
    state.inventories[0].constrs.extend(my_grass_list)
    state.inventories[1].constrs.extend(my_grass_list)
    for grass in my_grass_list:
        state.board[grass.coords[0]][grass.coords[1]].constr = grass

    # Set the enemy grass in a row on the bottom.
    enemy_grass_list = [Construction((x, 9), GRASS) for x in range(0, 9)]
    state.inventories[1].constrs.extend(enemy_grass_list)
    state.inventories[1].constrs.extend(enemy_grass_list)
    for grass in enemy_grass_list:
        state.board[grass.coords[0]][grass.coords[1]].constr = enemy_grass_list

    # Create one food for me
    my_food = Construction((9, 1), FOOD)
    state.board[my_food.coords[0]][my_food.coords[1]].constr = my_food
    state.inventories[0].constrs.append(my_food)

    # Create a worker for me that is carrying
    my_worker = Ant((8, 0), WORKER, 0)
    my_worker.carrying = True
    state.board[my_worker.coords[0]][my_worker.coords[1]].ant = my_worker
    state.inventories[0].ants.append(my_worker)

    # Create a drone for me
    my_drone = Ant((2, 8), DRONE, 0)
    state.board[my_drone.coords[0]][my_drone.coords[1]].ant = my_drone
    state.inventories[0].ants.append(my_drone)

    return state


def run_unit_tests() -> None:
    """
    run_unit_tests

    Runs the unit tests for the agent.
    """
    test_game_state = create_test_game_state()
    my_player = AIPlayer(0)

    # Test the evaluate_game_state function. It should evaluate to about -0.45
    evaluation_score = my_player.evaluate_game_state(test_game_state)
    if round(evaluation_score, 2) != -0.45:
        print("Test for evaluate_game_state failed!")

    # Test the find_best_move method.
    # It should want to move the queen off the anthill.
    actual_best_move = my_player.find_best_move(test_game_state, 0)
    expected_best_move = Move(MOVE_ANT, [(0, 0), (1, 0)])
    if actual_best_move.moveType != expected_best_move.moveType or\
            actual_best_move.coordList != expected_best_move.coordList:
        print("Test for find_best_move failed!")

    # Test the average_evaluation_score method.
    # It should average to 0.0 with the given nodes.
    node_list = [Node(Move(None), test_game_state, 1.0), Node(Move(None), test_game_state, -1.0)]
    average_eval_score = my_player.average_evaluation_score(node_list)
    if average_eval_score != 0.0:
        print("Test for average_evaluation_score failed!")


# Run the unit tests
run_unit_tests()
