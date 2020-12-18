#!/usr/bin/python3

from typing import List

from datasets.aDataset import ADataset #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class

from datasets.retailrocket.events import Events #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class

from simulation.aSequentialSimulation import ASequentialSimulation #class
from simulation.simulationML import SimulationML #class

from history.aHistory import AHistory #class
from evaluationTool.aEvalTool import AEvalTool #class

from datasets.ml.behaviours import Behaviours #class

from portfolio.aPortfolio import APortfolio #class

from simulation.tools.modelOfIndexes import ModelOfIndexes #class


class SimulationRR(SimulationML):

    _behaviourClass = Behaviours

    @staticmethod
    def divideDataset(dataset:ADataset, divisionDatasetPercentualSize:int,
                        testDatasetPercentualSize:int):

        eventsDF:DataFrame = dataset.eventsDF
        categoryTreeDF:DataFrame = dataset.categoryTreeDF
        itemPropertiesDF:DataFrame = dataset.itemPropertiesDF

        eventsSortedDF:DataFrame = eventsDF.sort_values(by=Events.COL_TIME_STAMP)
        numberOfEvents:int = eventsSortedDF.shape[0]

        trainSize:int = (int)(numberOfEvents * divisionDatasetPercentualSize / 100)
        trainEventsDF:DataFrame = eventsSortedDF[0:trainSize]
        trainEventsDF.reset_index(drop=True, inplace=True)

        testSize:int = (int)(numberOfEvents * testDatasetPercentualSize / 100)
        testEventsDF:DataFrame = eventsSortedDF[trainSize:(trainSize + testSize)]
        testEventsDF.reset_index(drop=True, inplace=True)

        #print(testEventsDF.head(30))
        trainDataset:ADataset = DatasetRetailRocket(trainEventsDF, categoryTreeDF, itemPropertiesDF)

        testEventsDF = testEventsDF.loc[testEventsDF[Events.COL_EVENT] == "transaction"]
        print("testEventsDF:")
        print(len(testEventsDF))

        return (trainDataset, testEventsDF)


    @staticmethod
    def initEvaluationModel(dataset:ADataset):
        eventsDF:DataFrame = dataset.eventsDF

        # model with clicked ItemIDs for users
        clickedItems:dict[List[int]] = {}
        for indexI, rowI in eventsDF.iterrows():
            clickedItems[rowI[Events.COL_VISITOR_ID]] = []
        return clickedItems


    def iterateOverDataset(self, portfolios:List[APortfolio], portfolioDescs:List[APortfolioDescription],
                             portFolioModels:List[DataFrame], evaluatonTools:List[AEvalTool], histories:List[AHistory],
                             testRatingsDF:DataFrame):

        model:ModelOfIndexes = ModelOfIndexes(testRatingsDF, Events)

        portIds:List[str] = [portDescI.getPortfolioID() for portDescI in portfolioDescs]

        evaluations:List[dict] = [{} for i in range(len(portfolios))]

        counterI:int = 0

        currentIndexDFI:int
        nextIndexDFI:int
        for currentIndexDFI in list(testRatingsDF.index):

            counterI += 1
            if counterI  % 100 == 0:
                print("EventI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]))
                print("PortfolioIds: " + str(portIds))
                print("Evaluations: " + str(evaluations))

                self.computationFile.write("EventI: " + str(counterI) + " / " + str(testRatingsDF.shape[0]) + "\n")
                self.computationFile.write("PortfolioIds: " + str(portIds) + "\n")
                self.computationFile.write("Evaluations: " + str(evaluations) + "\n")
                self.computationFile.flush()

            currentItemIdI:int = testRatingsDF.loc[currentIndexDFI][Events.COL_ITEM_ID]
            currentEventI:int = testRatingsDF.loc[currentIndexDFI][Events.COL_EVENT]
            currentVisitorIdI:int = testRatingsDF.loc[currentIndexDFI][Events.COL_VISITOR_ID]

            if currentEventI == "view":
                continue

            nextItemIDsI:int = self.getNextItemIDs(model, currentVisitorIdI, currentItemIdI, testRatingsDF, self._windowSize)
            if nextItemIDsI == []:
                continue

            portfolioI:APortfolio
            for portfolioI in portfolios:

                dfI:DataFrame = DataFrame([testRatingsDF.loc[currentIndexDFI]], columns=testRatingsDF.columns)

                portfolioI.update(dfI)

            repetitionI:int
            for repetitionI in range(self._repetitionOfRecommendation):
                self.simulateRecommendations(portfolios, portfolioDescs, portFolioModels, evaluatonTools,
                                               histories, evaluations, currentItemIdI, nextItemIDsI,
                                               currentVisitorIdI, repetitionI)

        return evaluations


    # model:ModelOfIndexes
    def getNextItemIDs(self, model, userId:int, itemId:int, ratingsDF:DataFrame, windowSize:int):

        selectedItems:List[int] = []

        itemIdI:int = itemId
        for i in range(windowSize):
            nextIndexIdI:int = model.getNextIndex(userId, itemIdI)
            if nextIndexIdI is None:
                break

            nextItemIdI:int = ratingsDF.loc[nextIndexIdI][Events.COL_ITEM_ID]
            nextUserIdI:int = ratingsDF.loc[nextIndexIdI][Events.COL_VISITOR_ID]
            if nextUserIdI != userId:
                raise ValueError("Error")

            selectedItems.append(nextItemIdI)

            itemIdI = nextItemIdI

        return selectedItems


