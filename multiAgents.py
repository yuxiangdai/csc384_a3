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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
        A reflex agent chooses an action at each choice point by examining
        its alternatives via a state evaluation function.

        The code below is provided as a guide.  You are welcome to change
        it in any way you see fit, so long as you don't touch our method
        headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        if legalMoves[chosenIndex] == "Stop":
            # scores[chosenIndex] == float("-inf")
            # newBestScore = max(scores)
            newBestIndices = [index for index in range(len(scores)) if scores[index] != bestScore]
            chosenIndex = random.choice(newBestIndices)
        return legalMoves[chosenIndex]


    def evaluationFunction(self, currentGameState, action):
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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distanceToScaredGhosts = [10000]
        distanceToGhosts = [float("inf")]
        distanceToFood = []
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            if ghost.scaredTimer:
                distanceToScaredGhosts.append(manhattanDistance(ghostPos, newPos))
            else:
                distanceToGhosts.append(manhattanDistance(ghostPos, newPos))
        if min(distanceToGhosts) == 0:
            distanceToGhosts = [float("inf")]

        if(newFood != []):
            distanceToFood = map(lambda x: manhattanDistance(x, newPos), newFood)
        else:
            distanceToFood = [0]
            
        
        return successorGameState.getScore() - 4 * 1.0 / min(distanceToGhosts)  - min(distanceToFood) - min(distanceToScaredGhosts)
        # return successorGameState.getScore() - 2 * min(distanceToFood) - min(distanceToScaredGhosts)


def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
        """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction.

            Here are some method calls that might be useful when implementing minimax.

            gameState.getLegalActions(player):
            Returns a list of legal actions for an agent
            player=0 means Pacman, ghosts are >= 1

            gameState.generateSuccessor(player, action):
            Returns the successor game state after an agent takes an action

            gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        action, score = self.DFMiniMax(gameState, 0, 0) 
        return action

    def DFMiniMax(self, curGameState, player, curDepth):
        if curDepth == self.depth or curGameState.isWin() or curGameState.isLose():
            return "STOP", self.evaluationFunction(curGameState)

        actions = curGameState.getLegalActions(player)

        succs = []
        for action in actions:
            succs.append([curGameState.generateSuccessor(player, action), action])
        
        if player == 0:
            curScore = float("-inf")
            for succ, action in succs:
                if player == curGameState.getNumAgents() - 1:
                    newPlayer = 0
                    newDepth = curDepth + 1
                else:
                    newPlayer = player + 1
                    newDepth = curDepth

                pacAction, score = self.DFMiniMax(succ, newPlayer, newDepth)
                if score > curScore:
                    curAction = action
                    curScore = score

        else:
            curScore = float("inf")
            for succ, action in succs:
                if player == curGameState.getNumAgents() - 1:
                    newPlayer = 0
                    newDepth = curDepth + 1
                else:
                    newPlayer = player + 1
                    newDepth = curDepth

                ghostAction, score = self.DFMiniMax(succ, newPlayer, newDepth)
                if score < curScore:
                    curAction = action
                    curScore = score

        return curAction, curScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
        Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
            Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alphabeta, action, score  = self.AlphaBeta(gameState, 0, 0, float("-inf"), float("inf")) 
        return action


    def AlphaBeta(self, curGameState, player, curDepth, curAlpha, curBeta):
        if curDepth == self.depth or curGameState.isWin() or curGameState.isLose():
            return self.evaluationFunction(curGameState), "STOP", self.evaluationFunction(curGameState)

        actions = curGameState.getLegalActions(player)

        # for some reason this code breaks the entire thing
        # successors = []
        # for action in actions:
        #     successors.append([curGameState.generateSuccessor(player, action), action])
        
        if player == 0:
            curScore = float("-inf")
            for action in actions:
                succ = curGameState.generateSuccessor(player, action)
                if player == curGameState.getNumAgents() - 1:
                    newPlayer = 0
                    newDepth = curDepth + 1
                else:
                    newPlayer = player + 1
                    newDepth = curDepth

                alpha, pacAction, score = self.AlphaBeta(succ, newPlayer, newDepth, curAlpha, curBeta)

                if score > curScore:
                    curAction = action
                    curScore = score
                if alpha >= curAlpha:
                    curAlpha = alpha
                if curBeta <= curAlpha:
                    break
            return curAlpha, curAction, curScore

        else:
            curScore = float("inf")
            for action in actions:
                succ = curGameState.generateSuccessor(player, action)
                if player == curGameState.getNumAgents() - 1:
                    newPlayer = 0
                    newDepth = curDepth + 1
                else:
                    newPlayer = player + 1
                    newDepth = curDepth

                beta, ghostAction, score,  = self.AlphaBeta(succ, newPlayer, newDepth, curAlpha, curBeta)

                if score < curScore:
                    curAction = action
                    curScore = score
                if beta <= curBeta:
                    curBeta = beta
                if curBeta <= curAlpha:
                    break
                    
            return curBeta, curAction, curScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
        Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
            Returns the expectimax action using self.depth and self.evaluationFunction

            All ghosts should be modeled as choosing uniformly at random from their
            legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, score = self.DFMiniMax(gameState, 0, 0) 
        return action


    def DFMiniMax(self, curGameState, player, curDepth):
        if curDepth == self.depth or curGameState.isWin() or curGameState.isLose():
            return "STOP", self.evaluationFunction(curGameState)

        actions = curGameState.getLegalActions(player)

        succs = []
        for action in actions:
            succs.append([curGameState.generateSuccessor(player, action), action])
        
        if player == 0:
            curScore = float("-inf")
            for succ, action in succs:
                if player == curGameState.getNumAgents() - 1:
                    newPlayer = 0
                    newDepth = curDepth + 1
                else:
                    newPlayer = player + 1
                    newDepth = curDepth

                pacAction, score = self.DFMiniMax(succ, newPlayer, newDepth)
                if score > curScore:
                    curAction = action
                    curScore = score

        # The only difference in implementation of Expectimax
        # search and Minimax search, is that at a min node, Expectimax search will return the average value over its
        # children as opposed to the minimum value.
        else:
            curScore = float("inf")
            totalScore = 0
            totalActions = 0
            for succ, action in succs:
                totalActions += 1
                if player == curGameState.getNumAgents() - 1:
                    newPlayer = 0
                    newDepth = curDepth + 1
                else:
                    newPlayer = player + 1
                    newDepth = curDepth

                ghostAction, score = self.DFMiniMax(succ, newPlayer, newDepth)
                totalScore += score
            curScore = totalScore * 1.0 / totalActions

        try:
            curAction
        except (UnboundLocalError, NameError) as e:
            return "STOP", curScore 
        return curAction, curScore 
            

def betterEvaluationFunction(currentGameState):
    """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).

        DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood().asList()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    distanceToScaredGhosts = [10000]
    distanceToGhosts = [float("inf")]
    distanceToFood = []
    for ghost in newGhostStates:
        ghostPos = ghost.getPosition()
        if ghost.scaredTimer:
            distanceToScaredGhosts.append(manhattanDistance(ghostPos, newPos))
        else:
            distanceToGhosts.append(manhattanDistance(ghostPos, newPos))
    if min(distanceToGhosts) == 0:
        distanceToGhosts = [float("inf")]

    if(newFood != []):
        distanceToFood = map(lambda x: manhattanDistance(x, newPos), newFood)
    else:
        distanceToFood = [0]
        
    
    return successorGameState.getScore() - 4 * 1.0 / min(distanceToGhosts)  - min(distanceToFood) - min(distanceToScaredGhosts)
    # return successorGameState.getScore() - 2 * min(distanceToFood) - min(distanceToScaredGhosts)

# Abbreviation
better = betterEvaluationFunction

