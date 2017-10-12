import svm
import parameters
from os import path

import prepare_data

# folder to temporary store files
TMP_FOLDER = path.join('..', 'tmp')

'compute features for classifying a word as aspect in a sentence'
class AspectTermFeatures:

    'Constructor: compute some basic statistics (such as words in training prepare_data), to be able to calculate the features individually per sentence later on'
    def __init__(self, train, commonFeatures):
        # we use word features and other features that are also used by other feature extractors, so we don't implement the same code here
        self.train = train

        self.comFeatures = commonFeatures

        # TODO features that are unique for this extractor

    'get features for a single sample'
    # index: id of sentence in dataset, since they have to be consecutive.
    # returns a string which matches the format constraints of SVM HMM
    def getFeatures(self, sentence, index):
        # string to return
        ret = ''

        # note that we have to generate a feature vector for each word in the sentence (sentence is treated as a sequence)
        ind = 0
        for word in sentence['tokens']:

            # feature array, which we will return
            # is a array of tuples, each tuple represent an entry in a sparse vector
            features = []

            # current offset in the feature vector, thus the size of the feature vector before considering the current feature
            offset = 0

            # compute word features for sentence
            [wordFeatures, offset] = self.comFeatures.getWordFeatures(word, offset)
            features += wordFeatures

            ### write in to file
            FEATURE_FILE = path.join(TMP_FOLDER, "aspectFeatures_getWordFeatures.txt")
            featuresFile = open(FEATURE_FILE, "w")
            featuresFile.write(str(features))
            featuresFile.close()
            ### writing close

            # compute context features for current word
            [contextFeatures, offset] = self.comFeatures.getContextFeatures(sentence, offset, True, parameters.contextWindow, ind)
            features += contextFeatures

            ### write in to file
            FEATURE_FILE = path.join(TMP_FOLDER, "aspectFeatures_getContextFeatureTrue.txt")
            featuresFile = open(FEATURE_FILE, "w")
            featuresFile.write(str(features))
            featuresFile.close()
            ### writing close

            [contextFeatures, offset] = self.comFeatures.getContextFeatures(sentence, offset, False, parameters.contextWindow, ind)
            features += contextFeatures

            ### write in to file
            FEATURE_FILE = path.join(TMP_FOLDER, "aspectFeatures_getContextFeatureFalse.txt")
            featuresFile = open(FEATURE_FILE, "w")
            featuresFile.write(str(features))
            featuresFile.close()
            ### writing close

            # add POS tag of current word as a feature
            [posFeature, offset] = self.comFeatures.getPoSFeature(sentence, offset, ind)
            features += posFeature

            ### write in to file
            FEATURE_FILE = path.join(TMP_FOLDER, "aspectFeatures_getPosFeature.txt")
            featuresFile = open(FEATURE_FILE, "w")
            featuresFile.write(str(features))
            featuresFile.close()
            ### writing close

            # add word vector of current word as a feature

            if prepare_data.useGlove:
                [wordFeature, offset] = self.comFeatures.getWordVectorFeatures(sentence, offset, ind)
                features += wordFeature

                ### write in to file
                FEATURE_FILE = path.join(TMP_FOLDER, "aspectFeatures_getWordVectorFeature_glove.txt")
                featuresFile = open(FEATURE_FILE, "w")
                featuresFile.write(str(features))
                featuresFile.close()
                ### writing close

            # TODO more features

            # for SVM hmm, the feature indices must be in increasing order
            features.sort(key=lambda tup: tup[0]) # sort by first element of tuples

            isAspect = 2 if sentence['aspects'][ind] > 0 else 1
            ret += str(isAspect) + " qid:" + str(index) + svm.sparseVectorToString(features, "aspectFeature") + '\n'

            ind += 1

            ### write in to file
            FEATURE_FILE = path.join(TMP_FOLDER, "aspectFeatures_functionReturn.txt")
            featuresFile = open(FEATURE_FILE, "w")
            featuresFile.write(str(features))
            featuresFile.close()
            ### writing close

        return ret