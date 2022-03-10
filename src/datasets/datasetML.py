#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from datasets.aDataset import ADataset #class

from datasets.ml.items import Items #class
from datasets.ml.ratings import Ratings #class
from datasets.ml.users import Users #class

class DatasetML(ADataset):

    def __init__(self, datasetID:str, ratingsDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        if type(datasetID) is not str:
            raise ValueError("Argument datasetID isn't type str.")
        if type(ratingsDF) is not DataFrame:
            raise ValueError("Argument ratingsDF isn't type DataFrame.")
        if type(usersDF) is not DataFrame:
            raise ValueError("Argument usersDF isn't type DataFrame.")
        if type(itemsDF) is not DataFrame:
            raise ValueError("Argument itemsDF isn't type DataFrame.")

        self.datasetID = datasetID
        self.ratingsDF:DataFrame = ratingsDF
        self.usersDF:DataFrame = usersDF
        self.itemsDF:DataFrame = itemsDF

    @staticmethod
    def readDatasets():
        # dataset reading
        ratingsDF:DataFrame = Ratings.readFromFileMl1m()
        usersDF:DataFrame = Users.readFromFileMl1m()
        itemsDF:DataFrame = Items.readFromFileMl1m()

        return DatasetML("ml1mDivAll", ratingsDF, usersDF, itemsDF)


    def getTheMostPopular(self):

        ratingsTrainDF:DataFrame = self.ratingsDF

        # ratingsSum:Dataframe<(userId:int, movieId:int, ratings:int, timestamp:int)>
        ratings5DF:DataFrame = ratingsTrainDF.loc[ratingsTrainDF[Ratings.COL_RATING] >= 4]

        # ratingsSum:Dataframe<(movieId:int, ratings:int)>
        ratings5SumDF:DataFrame = DataFrame(ratings5DF.groupby(Ratings.COL_MOVIEID)[Ratings.COL_RATING].count())

        # sortedAscRatings5CountDF:Dataframe<(movieId:int, ratings:int)>
        sortedAscRatings5CountDF:DataFrame = ratings5SumDF.sort_values(by=Ratings.COL_RATING, ascending=False)
        #print(sortedAscRatings5CountDF)

        return sortedAscRatings5CountDF


    def divideDataset(self, divisionDatasetPercentualSize:int, sortByTimestapmt=True):

        # create train Dataset
        ratingsToSplitDF:DataFrame = self.ratingsDF

        if sortByTimestapmt:
            ratingsToSplitDF = ratingsToSplitDF.sort_values(by=Ratings.COL_TIMESTAMP)

        numberOfRatings:int = ratingsToSplitDF.shape[0]
        trainSize:int = (int)(numberOfRatings * divisionDatasetPercentualSize / 100)

        trainRatingsDF:DataFrame = ratingsToSplitDF[0:trainSize]

        trainDatasetID:str = self.datasetID + "Div0-" + str(divisionDatasetPercentualSize)
        trainDataset:ADataset = DatasetML(trainDatasetID, trainRatingsDF, self.usersDF, self.itemsDF)

        testRatingsDF:DataFrame = ratingsToSplitDF[trainSize:]

        testDatasetID:str = self.datasetID + "Div" + str(divisionDatasetPercentualSize) + "-100"
        testDataset:ADataset = DatasetML(testDatasetID, testRatingsDF, self.usersDF, self.itemsDF)

        return (trainDataset, testDataset)