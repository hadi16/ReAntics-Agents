from AIPlayerUtils import *
from Constants import *
from GameState import GameState
from Player import Player

from math import exp, inf
from typing import Dict, List


class Perceptron:
    """
    Perceptron
    Class that represents a single perceptron in the neural network.

    Authors: Alex Hadi and Mitchell Nguyen
    Version: November 26, 2018
    """

    def __init__(self, weights):
        """
        __init__

        The constructor for Perceptron (creates a new Perceptron).

        :param weights: The list of weights to set for the perceptron.
        """

        self.weights: List[float] = weights
        self.error_term: float = 0.0
        self.output: float = 0.0


class Node:
    """
    Class: Node
    Class that represents a single node for the minimax evaluation.

    Authors: Alex Hadi and Mitchell Nguyen
    Version: November 26, 2018
    """

    def __init__(self, move: Move, state: GameState, state_evaluation: float, parent_node=None):
        """
        Node
        Class that represents a single node in the search tree.

        Authors: Alex Hadi and Mitchell Nguyen
        Version: November 26, 2018

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
    The Neural Networks Agent for CS 421.

    Back-propagation code adapted from:
    https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

    Authors: Alex Hadi and Mitchell Nguyen
    Version: November 26, 2018
    """

    def __init__(self, input_player_id: int):
        """
        __init__

        The constructor for AIPlayer (creates a new player).

        :param input_player_id: The player's ID as an integer.
        """

        super(AIPlayer, self).__init__(input_player_id, "Nermal")

        # True allows it to train more, otherwise it will use the hardcoded weights.
        self.TRAINING = False
        if self.TRAINING:
            self.LEARNING_RATE = 0.5
            self.training_set: Dict[GameState, float] = {}
            self.num_within_error_margin = 0

        self.NUM_INPUTS = len(self.map_game_state_to_input_array(GameState.getBasicState()))
        self.HIDDEN_PERCEPTRONS = int(self.NUM_INPUTS * 3)
        self.OUTPUT_PERCEPTRONS = 1

        self.neural_network: List[List[Perceptron]] = self.initialize_network()

    def initialize_network(self) -> List[List[Perceptron]]:
        """
        initialize_network

        Creates a neural network based on either the training set obtained from running the
        heuristic evaluation function, or through the hard-coded weights within the neural
        network once the agent has been fully trained.

        :return: A list of lists of perceptrons within the neural network,
                including the hidden and output layers.
        """

        if self.TRAINING:
            # Initialize hidden layer and output layer with random weights.
            return [
                # Hidden layer
                [
                    Perceptron([random.uniform(-1.0, 1.0) for _ in range(self.NUM_INPUTS + 1)])
                    for _ in range(self.HIDDEN_PERCEPTRONS)
                ],
                # Output layer
                [
                    Perceptron([random.uniform(-1.0, 1.0) for _ in range(self.HIDDEN_PERCEPTRONS+1)])
                    for _ in range(self.OUTPUT_PERCEPTRONS)
                ]
            ]
        else:
            # When the agent is not training, use the hardcoded weights instead.
            hidden_layer_weights = [
                [0.4187218127625196, -0.13378910736653646, 0.6761543435758932, -0.28978280718182425,
                 0.9193720661799243, -0.666450354474997, -0.9323596176587439, -0.4233560993620153],

                [0.0795217195410418, -0.355417984907123, -0.7960050525460557, -0.15327205905807012,
                 -0.0280987916130657, -0.009111911703121858, -0.5222478836336528, 0.6242402815834383],

                [0.3481444557169257, -0.8984728579549985, 0.48540125185432553, -1.0825915913222774,
                 -0.8747979761150072, 0.5860677394855431, -0.03796051025465741, -0.41638900024906517],

                [-0.3411521830933091, -0.8537703880673299, -0.16742385731772247, -0.4245150019280747,
                 0.47539995842224253, -0.3565339365292677, 0.30817183003618753, 0.36968534433821676],

                [0.038216236081534884, 0.2666342381426053, -0.3109864925816561, -0.9103633704854652,
                 -0.4973275856299116, -0.19926564493350163, 0.432996325626982, -0.9403767206673327],

                [0.458341801996665, 0.29380431249395594, 1.0603862788399654, -0.5471856289034723,
                 0.24586782650746242, 0.0557713460293829, -0.6318332581551038, 0.10881261876711548],

                [0.510183454644882, -0.8786141720301937, 0.534487860852592, -0.22838409923726521,
                 -0.8296693332402774, -0.31861993943720524, -0.7707868796867496, 0.3506956688769601],

                [0.2336854487556306, -0.23228541428305438, -0.9558163135072631, 0.5185160688638939,
                 0.03005904209316745, 0.6911241456529154, 0.6235771015362019, 0.517708695439517],

                [0.2607924978851921, 0.613301417643262, -0.5597726858112405, -0.8832933079345481,
                 -0.8598781858894444, 0.28833673943228005, -0.6848647464032613, 0.19059813468245893],

                [0.3001708677450452, 0.07331474060495881, -0.29758213271687234, 0.3779827306368417,
                 -0.9670633671510153, 0.8862649222942719, 0.033512576640676306, -0.41308698903736174],

                [1.1528121334919361, -0.06901899234264275, 0.29717291091385395, -0.7895687051376074,
                 -0.1626175871672342, -0.4996241162155171, 0.30906421593137634, -0.2735924421521647],

                [0.5520069648494649, 0.8102746482674321, 0.7535092970454791, -0.821735237162471,
                 -0.598203450521515, 0.7309398182915433, 0.18722082829233333, -0.4124322708230723],

                [0.025752667337424578, 0.3485986846228904, 0.9280116047420632, 0.23007076704630852,
                 0.06950339309636035, -0.8777153023934622, -0.363913980927127, 0.9019883613629963],

                [0.5994544763705297, 0.09091498659060576, 0.9650409729425854, -0.4776974213108278,
                 0.8194450451351216, 1.1470259851040863, -0.11682918155173416, -0.3036072661384948],

                [-0.2735693536584485, 0.7252553909571253, 0.905529328932912, 0.4996566130737462,
                 -0.562626568863321, 0.460397571554636, -0.23261249845894927, -0.2391497514022697],

                [-0.7108653776156025, 0.09873672109345824, 0.1766893112474018, -0.930342176767011,
                 -0.7575020185805245, -1.0329104402858924, -0.9387096831678379, 0.6417353033017638],

                [-1.2973188878316178, -1.131974259990376, -0.6709994622935425, -0.7005860820743463,
                 -0.9083913657585835, 0.06763873164642259, 0.32163680001861605, -0.8998201475269164],

                [0.9880749920774172, 0.5511671348875808, 0.6463354773002936, 0.7873121653672519,
                 -0.7056057630337138, 0.45347781744186244, -0.5892314699666177, 0.9958011143629564],

                [-0.8799151532387652, -0.13055083223510003, 0.6269146651525843, 0.249634733441537,
                 -0.11265312635112232, -0.25110859342373887, -0.7517508753083064, -0.35201376993023],

                [0.7160845182187194, -0.7817250070438541, -0.5811293303048741, 0.010995532618728978,
                 -0.4627362306039869, -0.7603393336740877, -0.4607729286060669, -0.9356496825221413],

                [0.3711850886295835, -0.5169658950810464, -0.11167605830086097, 0.35831790688899284,
                 -1.0140606243786023, -0.14211054784692329, -0.727057044637526, -0.8839210956806928]
            ]
            output_layer_weights = [
                [-0.7814962467236305, -0.6344946092856676, -0.7197899310210566, -1.01992036306982,
                 0.017759359288513278, 0.511112529628166, -0.7974570575326142, -0.403558312086394,
                 -0.4387294940542308, 0.22464592666486372, 0.8980142424474421, -0.41136530136218397,
                 0.6704722471147585, 0.8924597135173636, 0.5912191321248623, -1.2989181818393203,
                 -1.4385649775993967, 1.305798413374187, -0.7515724416627396, -0.06339108914135484,
                 -0.7390883868876651, -0.7645877910891021]
            ]
            return [
                [Perceptron(i) for i in hidden_layer_weights],
                [Perceptron(i) for i in output_layer_weights]
            ]

    def sum_weights_and_inputs(self, weights: List[float], inputs: List[float]) -> float:
        """
        sum_weights_and_inputs

        Summation of all the inputs multiplied by all their corresponding weights for
        each perceptron, which will be used in forward propagation

        :param weights: A list of all the perceptron's incoming weights
        :param inputs: A list of all the perceptron's inputs (which should be all the elements
                       from the input array)
        :return: The sum of all the inputs multiplied by all their corresponding weights.
        """
        return sum(
            weights[i] * inputs[i]
            for i in range(len(weights) - 1)
        ) + weights[-1]  # includes bias in sum (last element of weights)

    def activation_function(self, input_value: float) -> float:
        """
        activation_function

        The threshold function that implements the sigmoid equation.

        :param input_value: The sum of all the inputs multiplied by all their corresponding weights
                            for a given perceptron.
        :return: A threshold value based on the given sum of inputs & weights
        """

        return 1.0 / (1.0 + exp(-input_value))

    def run_network(self, inputs: List[float]) -> List[float]:
        """
        run_network

        Calculate the outputs for each of the perceptrons by summing up the values of each
        input with their correlating weights, and then use the activation (threshold) function
        to create the inputs for the output layer.

        :param inputs: The array of mapped inputs that correlate with the inputs of the
                       heuristic evaluation function.
        :return: A list of the inputs for the next layer of perceptrons.
        """

        # Go through each layer.
        for layer in self.neural_network:
            new_inputs = []
            # Go through each perceptron in a given layer.
            for perceptron in layer:
                # Sum the weights with inputs and apply the activation function.
                input_value = self.sum_weights_and_inputs(perceptron.weights, inputs)
                perceptron.output = self.activation_function(input_value)
                new_inputs.append(perceptron.output)
            inputs = new_inputs
        # Returns a list containing just the output layer's outputs.
        return inputs

    def perceptron_derivative(self, output_value: float) -> float:
        """
        perceptron_derivative

        The derivative of the threshold function, which will be used for calculating the errors
        of the weights during the process of backward propagation.

        :param output_value: The original output value from a perceptron.
        :return: The slope of the perceptron's output, which will be used to calculate the error
                 for the given perceptron.
        """

        return output_value * (1.0 - output_value)

    def backward_propagate_error(self, expected: List[float]) -> None:
        """
        backward_propagate_error

        Calculate the error for each of the perceptrons starting from the output layer,
        which will give us the error-term to propagate backwards throughout the neural network's
        hidden layer.

        :param expected: The expected output value of the perceptron
        """

        # Goes through the network in reverse (starts from the output layer of perceptrons).
        for i in reversed(range(len(self.neural_network))):
            layer = self.neural_network[i]
            errors: List[float] = []
            if i != len(self.neural_network) - 1:
                errors.extend(
                    # Create a list of errors based on the sum
                    # of the weights multiplied by the error term.
                    sum(
                        (perceptron.weights[j] * perceptron.error_term)
                        for perceptron in self.neural_network[i + 1]
                    )
                    for j in range(len(layer))
                )
            else:
                errors.extend(
                    # Creates a list of errors for each perceptron.
                    (expected[j] - perceptron.output)
                    for j, perceptron in enumerate(layer)
                )

            # Set the error term for each perceptron.
            for j, perceptron in enumerate(layer):
                perceptron.error_term = errors[j] * self.perceptron_derivative(perceptron.output)

    def update_weights(self, input_values: List[float]) -> None:
        """
        update_weights

        Once errors are calculated for each perceptron in the network via the back propagation
        method, these errors can be used to update the weights of each perceptron.

        :param input_values: The inputs for the perceptrons.
        """

        for i, layer in enumerate(self.neural_network):
            if i == 0:
                # inputs are set to all the inputs except the bias if at index 0
                inputs = input_values[:-1]
            else:
                inputs = [perceptron.output for perceptron in self.neural_network[i - 1]]

            for perceptron in layer:
                for j, input_value in enumerate(inputs):
                    perceptron.weights[j] += self.LEARNING_RATE*perceptron.error_term*input_value
                # Bias weight (end of weight list).
                perceptron.weights[-1] += self.LEARNING_RATE * perceptron.error_term

    def train_network(self):
        """
        train_network

        Update the neural network to train the agent based on the weights assigned for each
        perceptron. This method will be called every time a game ends.
        """

        TARGET_ERROR = 0.03
        TARGET_WITHIN_ERROR_MARGIN = 1000

        # Randomize training set.
        training_set = self.randomize_training_set_order()
        for state, target_score in training_set:
            if self.num_within_error_margin == TARGET_WITHIN_ERROR_MARGIN:
                print('Training complete!')

            # Propagate the mapped inputs forward through the network.
            mapped_inputs = self.map_game_state_to_input_array(state)
            actual_score = self.run_network(mapped_inputs)[0]

            # Increment number within error margin if applicable.
            if abs(target_score - actual_score) <= TARGET_ERROR:
                self.num_within_error_margin += 1

            # Perform back-propagation and update weights in the network.
            expected_values = [target_score for _ in range(self.OUTPUT_PERCEPTRONS)]
            self.backward_propagate_error(expected_values)
            self.update_weights(mapped_inputs)

    def predict_evaluation_score(self, current_state: GameState) -> float:
        """
        predict_evaluation_score

        Test an already-trained neural network to predict the evaluation score based
        on the given state of the game, which will determine the next state of the game
        that the agent chooses.

        :param current_state: The current state of the game.
        :return: An output value based on the forward propagation of the hidden layer's perceptrons
        """

        inputs = self.map_game_state_to_input_array(current_state)
        return self.run_network(inputs)[0]

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

        if self.TRAINING:
            self.training_set[current_state] = self.evaluate_game_state(current_state)

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
        """

        return enemy_locations[random.randint(0, len(enemy_locations) - 1)]

    def registerWin(self, has_won):
        """
        registerWin

        This agent doesn't learn.

        :param has_won: Whether the agent has won or not.
        """

        # If training, run the network and print the weights.
        if self.TRAINING:
            self.train_network()
            self.print_weights_to_console()
            self.training_set.clear()

    def print_weights_to_console(self) -> None:
        """
        print_weights_to_console

        Print out the weights that the neural network learned after being trained by each game.
        """

        print("Weights")
        all_weights = ([neuron.weights for neuron in layer] for layer in self.neural_network)
        for weight in all_weights:
            print(weight)
        print()

    def randomize_training_set_order(self) -> list:
        """
        randomize_training_set_order

        Further randomize the neural network's training set to make sure that the agent
        doesn't learn based on chronological order.

        :return: The randomized list of training items.
        """

        training_set = list(self.training_set.items())
        random.shuffle(training_set)
        return training_set

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

        # Get my total health.
        my_ants = getAntList(current_state, current_state.whoseTurn)
        my_total_health = sum(ant.health for ant in my_ants)

        # Get the enemy's total health.
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

        # If there is no queen in the state, return 0.0
        if not queen:
            return 0.0

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
        :return: The minimum steps to the given target.
        """

        # If the coordinates list is empty, return 0.
        if not coordinates_list:
            return 0
        return min([approxDist(target_coordinate, coordinates) for coordinates in coordinates_list])

    def _get_coordinate_list_of_game_elements(self, element_list: list) -> List[tuple]:
        """
        _get_coordinate_list_of_game_elements

        Helper function to get the coordinate list of the given list of Ants or Constructions.

        :param element_list: The list of Ant or Construction objects.
        :return: The list of coordinates as a List of tuples.
        """

        return [element.coords for element in element_list]

    def evaluate_game_state(self, current_state) -> float:
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

    def get_next_state_adversarial(self, current_state: GameState, move) -> GameState:
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

    def map_game_state_to_input_array(self, current_state: GameState) -> List[float]:
        """
        map_game_state_to_input_array

        Map the game state to an input array for the neural network.
        Each element will be a floating point number from -1.0 to 1.0

        :param current_state: The current game state.
        :return: The mapped list of inputs.
        """

        return [
            self._evaluate_worker_count(current_state),
            self._evaluate_soldier_count(current_state),
            self._evaluate_ant_difference(current_state),
            self._evaluate_health_difference(current_state),
            self._evaluate_worker_positions(current_state),
            self._evaluate_soldier_positions(current_state),
            self._evaluate_queen_position(current_state)
        ]

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
            if self.TRAINING:
                next_state_evaluation = self.evaluate_game_state(next_state)
            else:
                next_state_evaluation = self.predict_evaluation_score(next_state)
            child_nodes.append(Node(move, next_state, next_state_evaluation))

        # Sorts in descending order based on the state evaluation.
        child_nodes = sorted(child_nodes, key=lambda x: x.state_evaluation, reverse=True)

        # Take the highest 2 scoring nodes.
        return child_nodes[:2]

    def find_best_move_minimax(self, current_node: Node, current_depth: int, alpha: float,
                               beta: float, my_turn: bool) -> Move:
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
