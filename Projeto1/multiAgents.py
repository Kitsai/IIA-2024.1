# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).



#### ALUNOS ####
# Lucas Rocha dos Santos - 211055325
# Gabriela de Oliveira Henriques - 211055254
################

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food_list = newFood.asList()
        food_dist = [manhattanDistance(newPos, pos) for pos in food_list]

        curr_food_list = currentGameState.getFood().asList()
        curr_food_dist = [manhattanDistance(currentGameState.getPacmanPosition(),pos) for pos in curr_food_list]

        accumulator = 0

        # Get the closest food
        x, y = newPos
        accumulator += 1 if newFood[x][y] else 0

        # Penalty for remaining food
        accumulator -= len(food_list) * 5

        # Scores if pacman gets closer to the food
        if(food_list and min(food_dist) < min(curr_food_dist)):
            accumulator += 100/min(food_dist)

        # Penalty for doing nothing
        if(action == Directions.STOP):
            accumulator -25

        # Distance from ghost.
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            distance = manhattanDistance(newPos, ghostPos)
            if ghostState.scaredTimer > 0:
                accumulator += 100/distance
            else: 
                accumulator += distance

            if ghostPos == newPos:
                if ghostState.scaredTimer > 0:
                    accumulator += 5
                else:
                    accumulator = float('-inf')

        # Bonus for keeping scared times higher
        accumulator += sum(newScaredTimes)/len(newScaredTimes)

        # Gets the diference in score.
        return successorGameState.getScore() - currentGameState.getScore() + accumulator

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Starts by running a different version of max_value that also returns the action instead of just the score
        curr_action = None
        curr_score = float('-inf')
        # Get the legal actions for the pacman because he is the first agent
        legal_actions = gameState.getLegalActions(0)
        for action in  legal_actions:
            score = self.value(gameState.generateSuccessor(0,action), 0, 0) # The first 0 is the depth and the second 0 is the index. Gets the value of each successor.
            if(score > curr_score):
                curr_score = score
                curr_action = action
        return curr_action

    # The max_value and min_value functions are the same as the ones in the slides
    def max_value(self, gameState: GameState, depth, index):
        v = float('-inf')
        legal_actions = gameState.getLegalActions(index)
        for action in legal_actions:
            v = max(v, self.value(gameState.generateSuccessor(index,action), depth, index))
        return v
    
    def min_value(self, gameState: GameState, depth, index):
        v = float('inf')
        legal_actions = gameState.getLegalActions(index)
        for action in legal_actions:
            v = min(v, self.value(gameState.generateSuccessor(index,action), depth, index))
        return v
    
    # The value function treats the details of the implementation.
    def value(self, gameState: GameState, depth, index):
        # If the index is the number of agents, then it is the pacman's turn again. So is cycles between the agents.
        new_index = (index + 1) % gameState.getNumAgents()

        # If the index is 0, then it is the pacman's turn again. So the depth is increased since we are guaranteed to have gone through all the agents.
        if(new_index == 0):
            depth += 1 

        # Terminal test
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # If the index is 0, then it is the pacman's turn again. So it is a max node.
        if new_index == 0:
            return self.max_value(gameState, depth,new_index)
        # If the index is not 0, then it is a ghost's turn. So it is a min node.
        else:
            return self.min_value(gameState, depth,new_index)
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Starts by running a different version of max_value that also returns the action instead of just the score
        curr_action = None
        curr_score = float('-inf')
        alpha = float ("-inf")
        beta = float("inf")
        # Get the legal actions for the pacman because he is the first agent
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            score = self.value(gameState.generateSuccessor(0,action), 0, 0, alpha, beta) # The first 0 is the depth and the second 0 is the index. Gets the value of each successor.
            alpha = max(alpha, score)
            if(score > curr_score):
                curr_score = score
                curr_action = action
            # Never prune on root
        return curr_action

    # The max_value and min_value functions are the same as the ones in the slides
    def max_value(self, gameState: GameState, depth, index, alpha, beta):
        v = float('-inf')
        legal_actions = gameState.getLegalActions(index)
        for action in legal_actions:
            v = max(v, self.value(gameState.generateSuccessor(index,action), depth, index, alpha, beta))
            # maximizes alpha since its a max node
            alpha = max(alpha, v)
            # if prune condition is met stop visiting rest of nodes. 
            if (alpha > beta): break
        return v
    
    def min_value(self, gameState: GameState, depth, index, alpha, beta):
        v = float('inf')
        legal_actions = gameState.getLegalActions(index)
        for action in legal_actions:
            v = min(v, self.value(gameState.generateSuccessor(index,action), depth, index, alpha, beta))
            # minimizes beta since its a min node
            beta = min(beta, v)
            # if prune condition is met stop visiting rest of nodes
            if (alpha > beta): break
        return v
    
    # The value function treats the details of the implementation.
    def value(self, gameState: GameState, depth, index, alpha, beta):
        # If the index is the number of agents, then it is the pacman's turn again. So is cycles between the agents.
        new_index = (index + 1) % gameState.getNumAgents()

        # If the index is 0, then it is the pacman's turn again. So the depth is increased since we are guaranteed to have gone through all the agents.
        if(new_index == 0):
            depth += 1 

        # Terminal test
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # If the index is 0, then it is the pacman's turn again. So it is a max node.
        if new_index == 0:
            return self.max_value(gameState, depth,new_index, alpha, beta)
        # If the index is not 0, then it is a ghost's turn. So it is a min node.
        else:
            return self.min_value(gameState, depth,new_index, alpha, beta)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Starts by running a different version of max_value that also returns the action instead of just the score
        curr_action = None
        curr_score = float('-inf')
        # Get the legal actions for the pacman because he is the first agent
        legal_actions = gameState.getLegalActions(0)
        for action in  legal_actions:
            score = self.value(gameState.generateSuccessor(0,action), 0, 0) # The first 0 is the depth and the second 0 is the index. Gets the value of each successor.
            if(score > curr_score):
                curr_score = score
                curr_action = action
        return curr_action

    # The max_value is the same as the one in the slides
    def max_value(self, gameState: GameState, depth, index):
        v = float('-inf')
        legal_actions = gameState.getLegalActions(index)
        for action in legal_actions:
            v = max(v, self.value(gameState.generateSuccessor(index,action), depth, index))
        return v
    
    # Now is a exp_value function that deals with probability
    def exp_value(self, gameState: GameState, depth, index):
        v = 0
        legal_actions = gameState.getLegalActions(index)
        if(not legal_actions):
                return 0
        # Gives all legal action a equal chance of happening
        p = 1/len(legal_actions)
        for action in legal_actions:
            # Scales points based on likelyhood of happenning
            v += p*self.value(gameState.generateSuccessor(index,action), depth, index)
        return v
    
    # The value function treats the details of the implementation.
    def value(self, gameState: GameState, depth, index):
        # If the index is the number of agents, then it is the pacman's turn again. So is cycles between the agents.
        new_index = (index + 1) % gameState.getNumAgents()

        # If the index is 0, then it is the pacman's turn again. So the depth is increased since we are guaranteed to have gone through all the agents.
        if(new_index == 0):
            depth += 1 

        # Terminal test
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # If the index is 0, then it is the pacman's turn again. So it is a max node.
        if new_index == 0:
            return self.max_value(gameState, depth,new_index)
        # If the index is not 0, then it is a ghost's turn. So it is a min node.
        else:
            return self.exp_value(gameState, depth,new_index)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Variables
    pacman_position = currentGameState.getPacmanPosition()

    ghosts = currentGameState.getGhostStates()
    ghost_positions = currentGameState.getGhostPositions()
    ghost_dist = [manhattanDistance(pacman_position,pos) for pos in ghost_positions]

    scared_time = sum([ghost.scaredTimer for ghost in ghosts])

    food = currentGameState.getFood()
    food_list = food.asList()
    food_dist = [manhattanDistance(pacman_position,pos) for pos in food_list]

    capsules = currentGameState.getCapsules()
    capsules_dist = [manhattanDistance(pacman_position, pos) for pos in capsules]

    game_score = currentGameState.getScore()


    # evaluation
    accumulator = 0

    #adds the current game score
    accumulator += game_score

    # Adds how many spots not have food
    accumulator += len(food.asList(False))

    # based on ghosts changes what pacman wants to be close to
    if(scared_time > 0):
        accumulator += scared_time
        accumulator -= min(ghost_dist) if ghost_dist else 0 
        accumulator -= min(capsules_dist) if capsules_dist else 0
    else:
        accumulator += min(ghost_dist) if ghost_dist else 0
        accumulator -= min(food_dist) if food_dist else 0

    if(currentGameState.isWin()):
        accumulator += 1000
    if(currentGameState.isLose()):
        accumulator -= 1000

    return accumulator

# Abbreviation
better = betterEvaluationFunction
