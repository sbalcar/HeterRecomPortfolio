#!/usr/bin/python3

from typing import List
from typing import Dict

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from pandas.core.series import Series #class

from datasets.ml.ratings import Ratings #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from portfolio.aPortfolio import APortfolio #class

from simulation.aSequentialSimulation import ASequentialSimulation #class

import pandas as pd
import numpy as np

from pandas.core.frame import DataFrame #class

from history.aHistory import AHistory #class
from evaluationTool.aEvalTool import AEvalTool #class

from datasets.ml.behaviours import Behaviours #class

from simulation.tools.modelOfIndexes import ModelOfIndexes #class


class SimulationML(ASequentialSimulation):

    _ratingClass = Ratings
    _behaviourClass = Behaviours


    @staticmethod
    def divideDataset(dataset:ADataset, behaviourDF:DataFrame,
                      divisionDatasetPercentualSize:int, testDatasetPercentualSize:int,
                      recomRepetitionCount:int):

        ratingsDF:DataFrame = dataset.ratingsDF
        usersDF:DataFrame = dataset.usersDF
        itemsDF:DataFrame = dataset.itemsDF

        # create train Dataset
        ratingsSortedDF:DataFrame = ratingsDF.sort_values(by=Ratings.COL_TIMESTAMP)
        numberOfRatings:int = ratingsSortedDF.shape[0]

        trainSize:int = (int)(numberOfRatings * divisionDatasetPercentualSize / 100)
        #print("trainSize: " + str(trainSize))
        trainRatingsDF:DataFrame = ratingsSortedDF[0:trainSize]

        datasetID:str = "ml1m" + "Div" + str(divisionDatasetPercentualSize)
        trainDataset:ADataset = DatasetML(datasetID, trainRatingsDF, usersDF, itemsDF)


        # create test Rating DataFrame
        testSize:int = (int)(numberOfRatings * testDatasetPercentualSize / 100)
        #print("testSize: " + str(testSize))
        testRatingsDF:DataFrame = ratingsSortedDF[trainSize:(trainSize + testSize)]


        # create test relevant Rating DataFrame
        testRelevantRatingsDF:DataFrame = testRatingsDF.loc[testRatingsDF[Ratings.COL_RATING] >= 4]


        # create behaviour dictionary of DataFrame indexed by recomRepetition
        recomRepetitionCountInDataset:int = behaviourDF[Behaviours.COL_REPETITION].max() +1

        testRepeatedBehaviourDict:dict = {}
        bIndexes:List[int] = list([recomRepetitionCountInDataset*i for i in testRatingsDF.index])
        for repetitionI in range(recomRepetitionCount):
            # indexes of behaviour
            indexes:List[int] = [vI+repetitionI for vI in bIndexes]
            behaviourDFI:DataFrame = DataFrame(behaviourDF.take(indexes).values.tolist(),
                        index=testRatingsDF.index, columns=behaviourDF.keys())
            testRepeatedBehaviourDict[repetitionI] = behaviourDFI


        return (trainDataset, testRatingsDF, testRelevantRatingsDF, testRepeatedBehaviourDict)


    @staticmethod
    def initEvaluationModel(dataset:ADataset):
        usersDF:DataFrame = dataset.usersDF

        # model with clicked ItemIDs for users
        clickedItems:dict[List[int]] = {}
        for indexI, rowI in usersDF.iterrows():
            clickedItems[rowI[Ratings.COL_USERID]] = []
        return clickedItems



    def iterateOverDataset(self, portfolios:List[APortfolio], portfolioDescs:List[APortfolioDescription],
                             portFolioModels:List[pd.DataFrame], evaluatonTools:List[AEvalTool],
                             histories:List[AHistory], testRatingsDF:DataFrame, testRelevantRatingsDF:DataFrame,
                             testBehaviourDict:Dict[int, DataFrame]):

        model:ModelOfIndexes = ModelOfIndexes(testRelevantRatingsDF, Ratings)

        portIds:List[str] = [portDescI.getPortfolioID() for portDescI in portfolioDescs]

        evaluations:List[dict] = [{} for i in range(len(portfolios))]

        counterI:int = 0

        currentDFIndexI:int
        nextIndexDFI:int
        for currentDFIndexI in list(testRatingsDF.index):

            counterI += 1
            if counterI  % 100 == 0:
                print("RatingI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]))
                print("PortfolioIds: " + str(portIds))
                print("Evaluations: " + str(evaluations))

                self.computationFile.write("RatingI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]) + "\n")
                self.computationFile.write("PortfolioIds: " + str(portIds) + "\n")
                self.computationFile.write("Evaluations: " + str(evaluations) + "\n")
                self.computationFile.flush()

            currentItemIdI:int = testRatingsDF.loc[currentDFIndexI][Ratings.COL_MOVIEID]
            currentRatingI:int = testRatingsDF.loc[currentDFIndexI][Ratings.COL_RATING]
            currentUserIdI:int = testRatingsDF.loc[currentDFIndexI][Ratings.COL_USERID]

            if currentRatingI < 4:
                continue

            windowOfItemIDsI:int = self.getWindowOfItemIDs(model, currentUserIdI, currentDFIndexI, testRatingsDF, self._windowSize)

            portfolioI:APortfolio
            for portfolioI in portfolios:

                dfI:DataFrame = DataFrame([testRatingsDF.loc[currentDFIndexI]], columns=testRatingsDF.keys())
                portfolioI.update(dfI)

            repetitionI:int
            for repetitionI in range(self._recomRepetitionCount):
                self.simulateRecommendations(portfolios, portfolioDescs, portFolioModels, evaluatonTools,
                                             histories, evaluations, currentDFIndexI, currentUserIdI, repetitionI,
                                             testRatingsDF, testBehaviourDict, windowOfItemIDsI)

        return evaluations
