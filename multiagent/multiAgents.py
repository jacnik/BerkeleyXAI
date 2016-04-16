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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodScore = 1
        
        distancesToFood = [(manhattanDistance(newPos, foodPos), foodPos) for foodPos in newFood.asList()]
        if distancesToFood:
            closestFoodDist, closestFoodPos = min(distancesToFood)
            distancesToFood.remove((closestFoodDist, closestFoodPos))
        #foodScore = 1.0#/(closestFoodDist + 1) # the closer the better
        #foodScore += 1.0/(len(distancesToFood)+1)
        
            minAggDist = closestFoodDist
            while distancesToFood:
                distancesToFood = [(manhattanDistance(closestFoodPos, foodPos), foodPos) for _, foodPos in distancesToFood]
                closestFoodDist, closestFoodPos = min(distancesToFood)
                minAggDist += closestFoodDist
                distancesToFood.remove((closestFoodDist, closestFoodPos))
                
            foodScore = 1.0/minAggDist
        
        distancesToGhosts = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        closestGhostDist = min(distancesToGhosts)
        ghostScore = closestGhostDist # the further the better
        ghostScore = ghostScore/-foodScore if ghostScore < 3 else 0
        
        #import pdb; pdb.set_trace()
        return foodScore + ghostScore
        #return successorGameState.getScore()

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

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        maxDepth = self.depth * gameState.getNumAgents()
        maxInt = util.sys.maxint
        minInt = -util.sys.maxint - 1

        def maxValue(state, agentIndex, currDepth):
            v = minInt
            act = ''
            actions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                nextVal, nextAct = getValueDispatch(successor, nextAgent, currDepth + 1)
                if nextVal > v:
                    v = nextVal
                    act = action
                    
            return (v, act)

        def minValue(state, agentIndex, currDepth):
            v = maxInt
            act = ''
            actions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                nextVal, nextAct = getValueDispatch(successor, nextAgent, currDepth + 1)
                if nextVal < v:
                    v = nextVal
                    act = action

            return (v, act)
        
        def getValueDispatch(state, agentIndex, currDepth):
            if currDepth == maxDepth or state.isLose() or state.isWin():
                return (self.evaluationFunction(state), '')
            if agentIndex == 0: return maxValue(state, agentIndex, currDepth) # 0 is maximizing agent
            else: return minValue(state, agentIndex, currDepth)
        
        maxVal, maxAct = getValueDispatch(gameState, 0, 0)
        return maxAct


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxDepth = self.depth * gameState.getNumAgents()
        maxInt = util.sys.maxint
        minInt = -util.sys.maxint - 1
        
        # bMax: alpha 
        # bMin: beta
        def maxValue(state, agentIndex, currDepth, bMax, bMin):
            v = minInt
            act = ''
            actions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                nextVal, nextAct = getValueDispatch(successor, nextAgent, currDepth + 1, bMax, bMin)
                if nextVal > v:
                    v = nextVal
                    act = action
                if v > bMin: return (v, action)
                bMax = max(bMax, v) 

            return (v, act)

        def minValue(state, agentIndex, currDepth, bMax, bMin):
            v = maxInt
            act = ''
            actions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                nextVal, nextAct = getValueDispatch(successor, nextAgent, currDepth + 1, bMax, bMin)
                if nextVal < v:
                    v = nextVal
                    act = action
                if v < bMax: return (v, action)
                bMin = min(bMin, v)
 
            return (v, act)
        
        def getValueDispatch(state, agentIndex, currDepth, bMax, bMin):
            if currDepth == maxDepth or state.isLose() or state.isWin():
                return (self.evaluationFunction(state), '')
            if agentIndex == 0: return maxValue(state, agentIndex, currDepth, bMax, bMin) # 0 is maximizing agent
            else: return minValue(state, agentIndex, currDepth, bMax, bMin)
        
        maxVal, maxAct = getValueDispatch(gameState, 0, 0, minInt, maxInt)
        return maxAct


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
        maxDepth = self.depth * gameState.getNumAgents()
        maxInt = util.sys.maxint
        minInt = -util.sys.maxint - 1

        def maxValue(state, agentIndex, currDepth):
            v = minInt
            act = ''
            actions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                nextVal, nextAct = getValueDispatch(successor, nextAgent, currDepth + 1)
                if nextVal > v:
                    v = nextVal
                    act = action

            return (v, act)

        def expValue(state, agentIndex, currDepth):
            v = 0
            act = ''
            actions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            for action in actions:
                p = 1.0/len(actions) # probability of a single action (here uniform distribution)
                successor = state.generateSuccessor(agentIndex, action)
                nextVal, nextAct = getValueDispatch(successor, nextAgent, currDepth + 1)
                v += p * nextVal

            return (v, act)

        def getValueDispatch(state, agentIndex, currDepth):
            if currDepth == maxDepth or state.isLose() or state.isWin():
                return (self.evaluationFunction(state), '')
            if agentIndex == 0: return maxValue(state, agentIndex, currDepth) # 0 is maximizing agent
            else: return expValue(state, agentIndex, currDepth)

        maxVal, maxAct = getValueDispatch(gameState, 0, 0)
        return maxAct

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

