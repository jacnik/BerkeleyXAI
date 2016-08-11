# dataClassifier.py
# -----------------
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


# This file contains feature extraction methods and harness
# code for data classification

import mostFrequent
import naiveBayes
import perceptron
import perceptron_pacman
import mira
import samples
import sys
import util
from pacman import GameState

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    #features =  basicFeatureExtractorDigit(datum)

    "*** YOUR CODE HERE ***"
    #import pdb; pdb.set_trace()         
    #return features
    
    # kernel = (0, 1, 1)
    # features_1 = getFeaturesFromKernelMultiplication(datum, kernel, 1, 3)
    
    # kernel = (0, 1, 1)
    # features_2 = getFeaturesFromKernelMultiplication(datum, kernel, 3, 1)

    # kernel = (0, 0, 1,  0, 1, 1,  1, 1, 1)
    # features_3 = getFeaturesFromKernelMultiplication(datum, kernel, 3, 3)

    # kernel = (1, 1, 1,  1, 1, 0,  1, 0, 0)
    # features_4 = getFeaturesFromKernelMultiplication(datum, kernel, 3, 3)
    
    # kernel = (0, 1)
    # features_5 = getFeaturesFromKernelMultiplication(datum, kernel, 1, 2)
    
    # kernel = (0, 1)
    # features_6 = getFeaturesFromKernelMultiplication(datum, kernel, 2, 1)

    # kernel = (1, 0)
    # features_7 = getFeaturesFromKernelMultiplication(datum, kernel, 1, 2)
    
    # kernel = (1, 0)
    # features_8 = getFeaturesFromKernelMultiplication(datum, kernel, 2, 1)
 
    # kernel = (0, 1, 1,  0, 0, 1,  0, 0, 0)
    # features_9 = getFeaturesFromKernelMultiplication(datum, kernel, 3, 3)
    
    # kernel = (1, 1, 0,  1, 0, 0,  0, 0, 0)
    # features_10 = getFeaturesFromKernelMultiplication(datum, kernel, 3, 3)

    # kernel = (0, 0, 0,  1, 0, 0,  1, 1, 0)
    # features_11 = getFeaturesFromKernelMultiplication(datum, kernel, 3, 3)
    
    # kernel = (0, 0, 0,  0, 0, 1,  0, 1, 1)
    # features_12 = getFeaturesFromKernelMultiplication(datum, kernel, 3, 3)
    
    #res = features + features_1 + features_2 + features_3 # test => 78%
    #res = features_1 + features_2 + features_3 + features_4 # test => 74% 
    #res = features_5 + features_6 + features_7 + features_8 # 72%
    #res = features_9 + features_10 + features_11 + features_12 # 34%
    #res = features + features_5 + features_6 + features_7 + features_8 # 78%
    
    #features_13 = getFeaturesFromKernelAddition(datum, 3, 3)
    #features_14 = getFeaturesFromKernelAddition(datum, 4, 4)
    #features_15 = getFeaturesFromKernelAddition(datum, 5, 5)
    # features_16 = getFeaturesFromKernelAddition(datum, 6, 6)
    #features_17 = getFeaturesFromKernelAddition(datum, 2, 2)
    
    # res = features_13 #73
    # res = features_14 #75
    # res = features_15 #68
    #res = features_17 #80
    
    #features_18 = getFeaturesFromKernelAvg(datum, 2, 2)
    #features_19 = getFeaturesFromKernelAvg(datum, 3, 3)
    #features_20 = getFeaturesFromKernelAvg(datum, 4, 4)
    #features_21 = getFeaturesFromKernelAvg(datum, 5, 5) 
    #features_22 = getFeaturesFromKernelAvg(datum, 8, 8)
    #features_23 = getFeaturesFromKernelAvg(datum, 12, 12)
    #features_24 = getFeaturesFromKernelAvg(datum, 10, 10)
    
    #res = features_18 #78
    #res = features_19 #79
    #res = features_20 #78
    #res = features_21 #75
    #res = features_22 #46
    #res = features_23 #
    #res = features_24 #
    
    #features_25 = getFeaturesFromKernelAddition(datum, 2, 3)
    #features_26 = getFeaturesFromKernelAddition(datum, 3, 2)
    features_27 = getFeaturesFromKernelAddition(datum, 1, 2)
    #features_28 = getFeaturesFromKernelAddition(datum, 2, 1)
    
    #res = features_25 #79
    #res = features_26 #77
    res = features_27 #81
    #res = features_28 #75

    #import pdb; pdb.set_trace()
    return res
    
    
def getFeaturesFromKernelMultiplication(datum, kernel, kernelRows, kernelColumns):
    """
    returns Counter of features
    features are a binary multiple of pixels in kernel and image
    """
    #edges = basicFeatureExtractorFace(datum)
    a = datum.getPixels()
    
    def calculateBoolFromKernel(x, y):
        for dx in range(kernelColumns):
            for dy in range(kernelRows):
                #import pdb; pdb.set_trace()
                pixel = 1 if datum.getPixel(x + dx, y + dy) > 0 else 0
                kernelIndex = dx + dy * kernelColumns
                if kernel[kernelIndex] != pixel:
                    return False
        return True
    
    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH - kernelColumns):
        for y in range(DIGIT_DATUM_HEIGHT - kernelRows):
            for dx in range(kernelColumns):
                features[(x,y)] = calculateBoolFromKernel(x,y)
                
    return features

    
def getFeaturesFromKernelAddition(datum, kernelRows, kernelColumns):
    """
    returns Counter of features
    features are a binary sum of pixels in kernel,
    that means that it will set feature to 1 
    if any of the pixels in kernel range is a 1
    """
    #edges = basicFeatureExtractorFace(datum)
    a = datum.getPixels()
    
    def calculateBoolFromKernel(x, y):
        for dx in range(kernelColumns):
            for dy in range(kernelRows):
                #import pdb; pdb.set_trace()
                pixel = 1 if datum.getPixel(x + dx, y + dy) > 0 else 0
                if pixel == 1:
                    return True
        return False
    
    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH - kernelColumns):
        for y in range(DIGIT_DATUM_HEIGHT - kernelRows):
            for dx in range(kernelColumns):
                features[(x,y)] = calculateBoolFromKernel(x,y)
                
    return features

def getFeaturesFromKernelAvg(datum, kernelRows, kernelColumns):
    """
    returns Counter of features
    features are a binary average of pixels in kernel,
    that means that it will set feature to 1 
    if more than a half of the pixels in kernel range is a 1
    """
    #edges = basicFeatureExtractorFace(datum)
    a = datum.getPixels()
    treshold = kernelRows * kernelColumns / 2
    
    def calculateBoolFromKernel(x, y):
        counter = 0
        for dx in range(kernelColumns):
            for dy in range(kernelRows):
                #import pdb; pdb.set_trace()
                pixel = 1 if datum.getPixel(x + dx, y + dy) > 0 else 0
                counter += pixel
        
        return counter > treshold
    
    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH - kernelColumns):
        for y in range(DIGIT_DATUM_HEIGHT - kernelRows):
            for dx in range(kernelColumns):
                features[(x,y)] = calculateBoolFromKernel(x,y)
                
    return features
    

def basicFeatureExtractorPacman(state):
    """
    A basic feature extraction function.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """
    features = util.Counter()
    for action in state.getLegalActions():
        successor = state.generateSuccessor(0, action)
        foodCount = successor.getFood().count()
        featureCounter = util.Counter()
        featureCounter['foodCount'] = foodCount
        features[action] = featureCounter
    return features, state.getLegalActions()

def enhancedFeatureExtractorPacman(state):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """

    features = basicFeatureExtractorPacman(state)[0]
    for action in state.getLegalActions():
        features[action] = util.Counter(features[action], **enhancedPacmanFeatures(state, action))
    return features, state.getLegalActions()

def enhancedPacmanFeatures(state, action):
    """
    For each state, this function is called with each legal action.
    It should return a counter with { <feature name> : <feature value>, ... }
    """
    features = util.Counter()
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    return features


def contestFeatureExtractorDigit(datum):
    """
    Specify features to use for the minicontest
    """
    features =  basicFeatureExtractorDigit(datum)
    return features

def enhancedFeatureExtractorFace(datum):
    """
    Your feature extraction playground for faces.
    It is your choice to modify this.
    """
    features =  basicFeatureExtractorFace(datum)
    return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the printImage(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(guesses)):
    #     prediction = guesses[i]
    #     truth = testLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print rawTestData[i]
    #         break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
            # This is so that new features that you could define which
            # which are not of the form of (x,y) will not break
            # this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print "new features:", pix
                continue
        print image

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'], default='mostFrequent')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces', 'pacman'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-g', '--agentToClone', help=default("Pacman agent to copy"), default=None, type="str")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print "Doing classification"
    print "--------------------"
    print "data:\t\t" + options.data
    print "classifier:\t\t" + options.classifier
    if not options.classifier == 'minicontest':
        print "using enhanced features?:\t" + str(options.features)
    else:
        print "using minicontest feature extractor"
    print "training set size:\t" + str(options.training)
    if(options.data=="digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
        if (options.classifier == 'minicontest'):
            featureFunction = contestFeatureExtractorDigit
    elif(options.data=="faces"):
        printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorFace
        else:
            featureFunction = basicFeatureExtractorFace
    elif(options.data=="pacman"):
        printImage = None
        if (options.features):
            featureFunction = enhancedFeatureExtractorPacman
        else:
            featureFunction = basicFeatureExtractorPacman
    else:
        print "Unknown dataset", options.data
        print USAGE_STRING
        sys.exit(2)

    if(options.data=="digits"):
        legalLabels = range(10)
    else:
        legalLabels = ['Stop', 'West', 'East', 'North', 'South']

    if options.training <= 0:
        print "Training set size should be a positive integer (you provided: %d)" % options.training
        print USAGE_STRING
        sys.exit(2)

    if options.smoothing <= 0:
        print "Please provide a positive number for smoothing (you provided: %f)" % options.smoothing
        print USAGE_STRING
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print "Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2)
            print USAGE_STRING
            sys.exit(2)

    if(options.classifier == "mostFrequent"):
        classifier = mostFrequent.MostFrequentClassifier(legalLabels)
    elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        classifier.setSmoothing(options.smoothing)
        if (options.autotune):
            print "using automatic tuning for naivebayes"
            classifier.automaticTuning = True
        else:
            print "using smoothing parameter k=%f for naivebayes" %  options.smoothing
    elif(options.classifier == "perceptron"):
        if options.data != 'pacman':
            classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
        else:
            classifier = perceptron_pacman.PerceptronClassifierPacman(legalLabels,options.iterations)
    elif(options.classifier == "mira"):
        if options.data != 'pacman':
            classifier = mira.MiraClassifier(legalLabels, options.iterations)
        if (options.autotune):
            print "using automatic tuning for MIRA"
            classifier.automaticTuning = True
        else:
            print "using default C=0.001 for MIRA"
    elif(options.classifier == 'minicontest'):
        import minicontest
        classifier = minicontest.contestClassifier(legalLabels)
    else:
        print "Unknown classifier:", options.classifier
        print USAGE_STRING

        sys.exit(2)

    args['agentToClone'] = options.agentToClone

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options

# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl','pacmandata/food_validation.pkl','pacmandata/food_test.pkl' ),
    'StopAgent': ('pacmandata/stop_training.pkl','pacmandata/stop_validation.pkl','pacmandata/stop_test.pkl' ),
    'SuicideAgent': ('pacmandata/suicide_training.pkl','pacmandata/suicide_validation.pkl','pacmandata/suicide_test.pkl' ),
    'GoodReflexAgent': ('pacmandata/good_reflex_training.pkl','pacmandata/good_reflex_validation.pkl','pacmandata/good_reflex_test.pkl' ),
    'ContestAgent': ('pacmandata/contest_training.pkl','pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl' )
}
# Main harness code



def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']
    
    # Load data
    numTraining = options.training
    numTest = options.test

    if(options.data=="pacman"):
        agentToClone = args.get('agentToClone', None)
        trainingData, validationData, testData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES.get(agentToClone, (None, None, None))
        trainingData = trainingData or args.get('trainingData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][0]
        validationData = validationData or args.get('validationData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][1]
        testData = testData or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][2]
        rawTrainingData, trainingLabels = samples.loadPacmanData(trainingData, numTraining)
        rawValidationData, validationLabels = samples.loadPacmanData(validationData, numTest)
        rawTestData, testLabels = samples.loadPacmanData(testData, numTest)
    else:
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


    # Extract features
    print "Extracting features..."
    trainingData = map(featureFunction, rawTrainingData)
    validationData = map(featureFunction, rawValidationData)
    testData = map(featureFunction, rawTestData)

    # Conduct training and testing
    print "Training..."
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print "Validating..."
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
    print "Testing..."
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

    # do odds ratio computation if specified at command line
    if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.findHighOddsFeatures(label1,label2)
        if(options.classifier == "naiveBayes" or options.classifier == "nb"):
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

        print string3
        printImage(features_odds)

    if((options.weights) & (options.classifier == "perceptron")):
        for l in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(l)
            print ("=== Features with high weight for label %d ==="%l)
            printImage(features_weights)

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)
