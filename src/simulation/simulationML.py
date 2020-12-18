#!/usr/bin/python3

from typing import List

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from pandas.core.series import Series #class

from datasets.ml.ratings import Ratings #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from portfolio.aPortfolio import APortfolio #class

from simulation.aSequentialSimulation import ASequentialSimulation #class

import pandas as pd

from pandas.core.frame import DataFrame #class

from history.aHistory import AHistory #class
from evaluationTool.aEvalTool import AEvalTool #class

from datasets.ml.behaviours import Behaviours #class

from simulation.tools.modelOfIndexes import ModelOfIndexes #class

class SimulationML(ASequentialSimulation):

    _behaviourClass = Behaviours

    @staticmethod
    def divideDataset(dataset:ADataset, divisionDatasetPercentualSize:int,
                        testDatasetPercentualSize:int):

        ratingsDF:DataFrame = dataset.ratingsDF
        usersDF:DataFrame = dataset.usersDF
        itemsDF:DataFrame = dataset.itemsDF

        ratingsSortedDF:DataFrame = ratingsDF.sort_values(by=Ratings.COL_TIMESTAMP)
        numberOfRatings:int = ratingsSortedDF.shape[0]

        trainSize:int = (int)(numberOfRatings * divisionDatasetPercentualSize / 100)
        trainRatingsDF:DataFrame = ratingsSortedDF[0:trainSize]
        trainRatingsDF.reset_index(drop=True, inplace=True)

        testSize:int = (int)(numberOfRatings * testDatasetPercentualSize / 100)
        testRatingsDF:DataFrame = ratingsSortedDF[trainSize:(trainSize + testSize)]
        testRatingsDF.reset_index(drop=True, inplace=True)

        trainDataset:ADataset = DatasetML(trainRatingsDF, usersDF, itemsDF)
        return (trainDataset, testRatingsDF)


    @staticmethod
    def initEvaluationModel(dataset:ADataset):
        usersDF:DataFrame = dataset.usersDF

        # model with clicked ItemIDs for users
        clickedItems:dict[List[int]] = {}
        for indexI, rowI in usersDF.iterrows():
            clickedItems[rowI[Ratings.COL_USERID]] = []
        return clickedItems



    def iterateOverDataset(self, portfolios:List[APortfolio], portfolioDescs:List[APortfolioDescription],
                             portFolioModels:List[pd.DataFrame], evaluatonTools:List[AEvalTool], histories:List[AHistory],
                             testRatingsDF:DataFrame):

        model:ModelOfIndexes = ModelOfIndexes(testRatingsDF, Ratings)

        portIds:List[str] = [portDescI.getPortfolioID() for portDescI in portfolioDescs]

        evaluations:List[dict] = [{} for i in range(len(portfolios))]

        counterI:int = 0

        currentIndexDFI:int
        nextIndexDFI:int
        for currentIndexDFI in list(testRatingsDF.index):

            counterI += 1
            if counterI  % 100 == 0:
                print("RatingI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]))
                print("PortfolioIds: " + str(portIds))
                print("Evaluations: " + str(evaluations))

                self.computationFile.write("RatingI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]) + "\n")
                self.computationFile.write("PortfolioIds: " + str(portIds) + "\n")
                self.computationFile.write("Evaluations: " + str(evaluations) + "\n")
                self.computationFile.flush()

            currentItemIdI:int = testRatingsDF.loc[currentIndexDFI][Ratings.COL_MOVIEID]
            currentRatingI:int = testRatingsDF.loc[currentIndexDFI][Ratings.COL_RATING]
            currentUserIdI:int = testRatingsDF.loc[currentIndexDFI][Ratings.COL_USERID]

            if currentRatingI < 4:
                continue

            nextItemIDsI:int = self.getNextItemIDs(model, currentUserIdI, currentItemIdI, testRatingsDF, self._windowSize)
            if nextItemIDsI == []:
                continue

            portfolioI:APortfolio
            for portfolioI in portfolios:

                dfI:DataFrame = DataFrame([testRatingsDF.loc[currentIndexDFI]],
                    columns=[Ratings.COL_USERID, Ratings.COL_MOVIEID, Ratings.COL_RATING, Ratings.COL_TIMESTAMP])

                portfolioI.update(dfI)

            repetitionI:int
            for repetitionI in range(self._repetitionOfRecommendation):
                self.simulateRecommendations(portfolios, portfolioDescs, portFolioModels, evaluatonTools,
                                               histories, evaluations, currentItemIdI, nextItemIDsI,
                                               currentUserIdI, repetitionI)

        return evaluations


    # model:ModelOfIndexes
    def getNextItemIDs(self, model, userId:int, itemId:int, ratingsDF:DataFrame, windowSize:int):

        selectedItems:List[int] = []

        itemIdI:int = itemId
        for i in range(windowSize):
            nextIndexIdI:int = model.getNextIndex(userId, itemIdI)
            if nextIndexIdI is None:
                break

            nextItemIdI:int = ratingsDF.loc[nextIndexIdI][Ratings.COL_MOVIEID]
            nextUserIdI:int = ratingsDF.loc[nextIndexIdI][Ratings.COL_USERID]
            if nextUserIdI != userId:
                raise ValueError("Error")

            selectedItems.append(nextItemIdI)

            itemIdI = nextItemIdI

        return selectedItems

