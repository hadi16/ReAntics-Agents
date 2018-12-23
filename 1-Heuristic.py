from typing import Dict, List

from AIPlayerUtils import *
from GameState import GameState
from Inventory import Inventory
from Player import *


class AIPlayer(Player):
    """
    The Heuristic AI for Dr. Nuxoll's CS 421 course.
    Overall strategy:
    Kill the enemy's worker, create a defense, and collect food in order to win.

    Authors: Alex Hadi and Noah Schutte
    Version: September 10, 2018
    """

    def __init__(self, input_player_id: int):
        """
        __init__

        Creates a new Player object.

        :param input_player_id: The ID to give the new player
        """
        super(AIPlayer, self).__init__(input_player_id, "Heuristic")

    def _setup_phase_one_placement(self) -> List[tuple]:
        """
        _setup_phase_one_placement

        Helper method that handles the setup phase one placement
        (placing anthill, tunnel, and grass).
        This agent uses hardcoded placements.

        :return: A list of tuples with the positions for the anthill, tunnel, and grass.
        """
        return [(1, 1), (7, 1),
                # Grass placements
                (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3)]

    def _setup_phase_two_placement(self, current_state: GameState) -> List[tuple]:
        """
        _setup_phase_two_placement

        Helper method that handles the setup phase two placement (placing enemy's food).

        :param current_state: The current GameState.
        :return: A list of tuples of where to place the enemy's food.
        """
        items = Items(current_state)
        enemy_tunnel = items.enemy_tunnel
        enemy_anthill = items.enemy_anthill

        # x coordinates: 0 - 9, y coordinates: 6 - 9
        ENEMY_SQUARES = [(x, y) for x in range(10) for y in range(6, 10)]

        # Keys:   average of approximate distance to anthill and tunnel.
        # Values: list of tuples representing the corresponding (x, y) placements.
        legal_food_placements: Dict[float, List[tuple]] = {}
        for x, y in ENEMY_SQUARES:
            if current_state.board[x][y].constr is None:
                enemy_tunnel_dist = approxDist(enemy_tunnel.coords, (x, y))
                enemy_anthill_dist = approxDist(enemy_anthill.coords, (x, y))
                min_dist = min(enemy_tunnel_dist, enemy_anthill_dist)

                # Extend list for key in the dictionary if key exists. Otherwise, create that key.
                if min_dist in legal_food_placements:
                    legal_food_placements[min_dist].append((x, y))
                else:
                    legal_food_placements[min_dist] = [(x, y)]

        # Gets the max dist placements and deletes that key from the dictionary.
        max_min_dist = max(legal_food_placements)
        max_min_dist_placements = legal_food_placements.pop(max_min_dist)

        # Only takes two tuples from the list (if there are that many)
        food_placements = max_min_dist_placements[:2]

        # If only one position was taken from the dictionary so far.
        if len(food_placements) != 2:
            max_min_dist = max(legal_food_placements)
            max_min_dist_placements = legal_food_placements[max_min_dist]

            # Only want one more position.
            food_placements.append(max_min_dist_placements[0])

        return food_placements

    def getPlacement(self, current_state: GameState) -> List[tuple]:
        """
        getPlacement

        The agent uses a hardcoded arrangement for phase 1 for the queen's maximum protection.
        Enemy food is placed the furthest away from both the tunnel and the anthill.

        :param current_state: The current GameState.
        :return: A list of tuples representing where to place items.
        """

        '''
        Citation:
        https://stackoverflow.com/questions/24580993/
        calling-functions-with-parameters-using-a-dictionary-in-python
        
        Dictionary used to mock switch statement in Python
        Key: setup phase (integer)
        Value: corresponding function as a lambda function to helper method
        '''
        setup_phase_function_dict: Dict[int, function] = {
            SETUP_PHASE_1: lambda: self._setup_phase_one_placement(),
            SETUP_PHASE_2: lambda: self._setup_phase_two_placement(current_state)
        }
        # Behavior of get() method: will return None if invalid phase entered
        # (analogous to "default" in switch statement)
        setup_phase_function = setup_phase_function_dict.get(current_state.phase)
        if setup_phase_function is not None:
            return setup_phase_function()

    def _specified_ants_have_moved(self, ant_list: List[Ant]) -> bool:
        """
        _specified_ants_have_moved

        Helper method that determines if all of the given ants have moved.

        :param ant_list: The specified ants to check.
        :return: True if all of those ants have moved. Otherwise, returns False.
        """
        if not ant_list:
            return True
        else:
            ant_moves = 0
            for ant in ant_list:
                if ant.hasMoved:
                    ant_moves += 1
            if ant_moves == len(ant_list):
                return True
        return False

    def _move_worker(self, current_state: GameState, worker: Ant) -> Move:
        """
        _move_worker

        Allows the agent to move the specified worker.

        :param current_state: The current GameState.
        :param worker: The worker to be moved.
        :return: The Move that the worker wishes to complete.
        """
        # If the worker has food, move toward tunnel. Otherwise, move toward food.
        items = Items(current_state)
        worker_target = items.my_tunnel.coords if worker.carrying else items.my_closest_food.coords
        path = createPathToward(current_state, worker.coords, worker_target,
                                UNIT_STATS[WORKER][MOVEMENT])

        # Prevents the workers from going back and forth
        if len(path) < 2:
            reachable_coordinates = listReachableAdjacent(current_state, worker.coords,
                                                          UNIT_STATS[WORKER][MOVEMENT])
            if reachable_coordinates:
                coordinate = random.choice(reachable_coordinates)
                path = createPathToward(current_state, worker.coords, coordinate,
                                        UNIT_STATS[WORKER][MOVEMENT])
        return Move(MOVE_ANT, path, None)

    def _get_path_for_r_soldier(self, current_state: GameState, r_soldier: Ant,
                                my_r_soldiers: List[Ant]) -> List[tuple]:
        """
        _get_path_for_r_soldier
        
        Gets the path for the specified ranged soldier.
        Implicitly returns None if no such path is found.

        :param current_state: The current GameState.
        :param r_soldier: The ranged soldier to find the path for.
        :param my_r_soldiers: A list with all ranged soldiers
        :return: A list of tuples representing the path for the ranged soldier.
        """

        # The coordinates where the ranged soldiers take defense
        R_SOLDIER_TARGET_LIST = [(9, 4), (7, 4), (5, 4), (3, 4), (1, 4)]
        if r_soldier.coords[1] < 4:
            return createPathToward(current_state, r_soldier.coords, R_SOLDIER_TARGET_LIST[4],
                                    UNIT_STATS[R_SOLDIER][MOVEMENT])
        else:
            for target in R_SOLDIER_TARGET_LIST:
                if r_soldier.coords == target:
                    return [r_soldier.coords]
                elif getAntAt(current_state, target) not in my_r_soldiers:
                    return createPathToward(current_state, r_soldier.coords, target,
                                            UNIT_STATS[R_SOLDIER][MOVEMENT])

    def getMove(self, current_state: GameState) -> Move:
        """
        getMove

        This agent creates two workers to collect food.
        It moves drones towards the enemy's workers.
        It creates a defense using ranged soldiers.

        :param current_state: The current GameState.
        :return: The Move that the AI wishes to perform.
        """
        # Get all items needed for me and enemy.
        items = Items(current_state)
        my_food_count = items.my_food_count
        my_queen = items.my_queen
        my_workers = items.my_workers
        my_drones = items.my_drones
        my_r_soldiers = items.my_r_soldiers
        my_anthill = items.my_anthill
        my_ants = items.my_ants
        enemy_workers = items.enemy_workers
        enemy_anthill = items.enemy_anthill

        # If all ants have moved, we're done 
        if self._specified_ants_have_moved(my_ants):
            return Move(END, None, None)

        # If the queen is on the anthill, move her
        if my_queen.coords == my_anthill.coords:
            return Move(MOVE_ANT, [my_queen.coords, (1, 0)], None)

        # If she hasn't moved, have her move in place so she will attack
        if not my_queen.hasMoved:
            return Move(MOVE_ANT, [my_queen.coords], None)

        # If there is no food and there are no workers left, go full force to the enemy's anthill
        if my_food_count < 1 and not my_workers:
            for ant in my_ants:
                if not ant.hasMoved and ant is not my_queen:
                    path = createPathToward(current_state, ant.coords, enemy_anthill.coords,
                                            UNIT_STATS[ant.type][MOVEMENT])
                    return Move(MOVE_ANT, path, None)

        # If I have the food and the anthill is unoccupied, then make a drone (but just 1)
        if my_food_count > 1 and not my_drones:
            if getAntAt(current_state, my_anthill.coords) is None:
                return Move(BUILD, [my_anthill.coords], DRONE)

        # Make sure to have 2 workers at all times. We make at least one drone first.
        if my_food_count > 0 and len(my_workers) < 2 and my_drones:
            if getAntAt(current_state, my_anthill.coords) is None:
                return Move(BUILD, [my_anthill.coords], WORKER)

        # Move the drone to attack the enemy's workers
        for drone in my_drones:
            if not drone.hasMoved:
                if enemy_workers:
                    # Find the closest enemy worker.
                    # Key: approx distance, Value: given worker
                    closest_worker_dict: Dict[int, Ant] = {}
                    for worker in enemy_workers:
                        distance_to_drone = approxDist(drone.coords, worker.coords)
                        closest_worker_dict[distance_to_drone] = worker
                    closest_worker = closest_worker_dict[min(closest_worker_dict)]

                    path = createPathToward(current_state, drone.coords, closest_worker.coords,
                                            UNIT_STATS[DRONE][MOVEMENT])
                    return Move(MOVE_ANT, path, None)
                else:  # If there are no enemy workers take position in defense line
                    path = createPathToward(current_state, drone.coords, (0, 4),
                                            UNIT_STATS[DRONE][MOVEMENT])
                    return Move(MOVE_ANT, path, None)

        # If enough food, make a ranged soldier (but not more than 5)
        if my_food_count > 1 and len(my_r_soldiers) < 5:
            if getAntAt(current_state, my_anthill.coords) is None:
                return Move(BUILD, [my_anthill.coords], R_SOLDIER)

        # Move ranged soldiers to create defense
        for r_soldier in my_r_soldiers:
            if not r_soldier.hasMoved:
                path = self._get_path_for_r_soldier(current_state, r_soldier, my_r_soldiers)
                if path is not None:
                    return Move(MOVE_ANT, path, None)

        # Move the workers around
        for worker in my_workers:
            if not worker.hasMoved:
                return self._move_worker(current_state, worker)

    def getAttack(self, current_state: GameState, attacking_ant: Ant,
                  enemy_locations: List[tuple]) -> tuple:
        """
        getAttack

        The agent never attacks.

        :param current_state: The current GameState.
        :param attacking_ant: The ant that is currently attacking.
        :param enemy_locations: A list of tuples that represent where the enemies are.
        :return: A tuple representing the player to attack.
        """
        return enemy_locations[0]

    def registerWin(self, has_won: bool) -> None:
        """
        registerWin

        The agent doesn't learn.
        :param has_won: A boolean indicating if the AI won or not.
        """
        pass


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
        self._my_inventory: Inventory = getCurrPlayerInventory(current_state)

        # I should either be 0 or 1 (enemy is just 1 or 0, respectively)
        self._me: int = current_state.whoseTurn
        self._enemy = 1 - current_state.whoseTurn

    @property
    def my_food_count(self) -> int:
        """
        my_food_count

        :return: The amount of food I currently have.
        """
        return self._my_inventory.foodCount

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
            food_dist = stepsToReach(self._current_state, self.my_tunnel.coords, food.coords)
            food_distances_dict[food_dist] = food

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
    def my_queen(self) -> Ant:
        """
        my_queen

        :return: My queen from my inventory.
        """
        return self._my_inventory.getQueen()

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

    @property
    def enemy_anthill(self) -> Construction:
        """
        enemy_anthill

        :return: The enemy's anthill.
        """
        return getConstrList(self._current_state, self._enemy, (ANTHILL,))[0]

    @property
    def enemy_tunnel(self) -> Construction:
        """
        enemy_tunnel

        :return: The enemy's tunnel.
        """
        return getConstrList(self._current_state, self._enemy, (TUNNEL,))[0]
