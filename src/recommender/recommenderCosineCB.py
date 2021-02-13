import pandas as pd
import numpy as np
import scipy.sparse

from pandas.core.frame import DataFrame  #class

from typing import List
from typing import Dict

from sklearn.metrics import *
from sklearn.preprocessing import normalize
from recommender.aRecommender import ARecommender  #class

from datasets.aDataset import ADataset  #class
from datasets.datasetML import DatasetML  #class
from datasets.datasetRetailrocket import DatasetRetailRocket  # lass
from datasets.datasetST import DatasetST  #class

from datasets.ml.ratings import Ratings  #class

from history.aHistory import AHistory  #class

from recommender.tools.toolMMR import ToolMMR #class


class RecommenderCosineCB(ARecommender):
    ARG_CB_DATA_PATH:str = "cbDataPath"

    ARG_USER_PROFILE_STRATEGY:str = "userProfileStrategy"

    ARG_USER_PROFILE_SIZE:str = "userProfileSize"

    ARG_USE_DIVERSITY:str = "useDiversity"

    ARG_MMR_LAMBDA:str = "MMRLambda"

    DEBUG_MODE = False

    def __init__(self, jobID: str, argumentsDict: Dict[str, object]):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        # arguments je dictionary, povinny parametr je cesta k souboru s CB daty
        self._arguments: dict = argumentsDict
        # "../../../../data/cbDataOHE.txt" nebo "../../../../data/cbDataTFIDF.txt"
        self.cbDataPath: str = self._arguments[self.ARG_CB_DATA_PATH]
        # print(argumentsDict)

        if "simMatrixRR.npz" in self.cbDataPath:
            sparseMat = scipy.sparse.load_npz(self.cbDataPath)
            sparseMat.setdiag(0.0)
            sparseMat = sparseMat.transpose()
            itemIDs = np.load(self.cbDataPath.replace("simMatrixRR.npz", "itemsRR.npy"))
            # self.cbData = pd.SparseDataFrame(sparseMat, columns=itemIDs , index=itemIDs)¨#alternativa pro pandas <= 1.0
            self.cbData = pd.DataFrame.sparse.from_spmatrix(sparseMat, columns=itemIDs, index=itemIDs)

        else:
            self.dfCBFeatures = pd.read_csv(self.cbDataPath, sep=",", header=0, index_col=0)
            # self.dfCBFeatures.fillna(self.dfCBFeatures.mean(), inplace=True)
            # print(self.dfCBFeatures)
            dfCBSim = 1 - pairwise_distances(self.dfCBFeatures, metric="cosine")
            np.fill_diagonal(dfCBSim, 0.0)
            self.cbData: DataFrame = DataFrame(data=dfCBSim, index=self.dfCBFeatures.index,
                                               columns=self.dfCBFeatures.index)
            self.cbData = self.cbData.transpose()
        self.userProfiles: dict = {}

        self._useMMR = False
        if self._arguments.get(self.ARG_USE_DIVERSITY, False):
            self._useMMR = True
            self._MMR_lambda = self._arguments.get(self.ARG_MMR_LAMBDA)


    def train(self, history:AHistory, dataset:ADataset):
        if not isinstance(history, AHistory):
            raise ValueError("Argument history isn't type AHistory.")
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")
        self._trainDataset = dataset

        if type(dataset) is DatasetML:
            COL_USERID:str = Ratings.COL_USERID
            COL_ITEMID:str = Ratings.COL_MOVIEID
            ratingsDF:DataFrame = dataset.ratingsDF
            # ratingsSum:Dataframe<(userId:int, movieId:int, ratings:int, timestamp:int)>
            ratingsDF = ratingsDF.loc[ratingsDF[Ratings.COL_RATING] >= 4]

        elif type(dataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events  # class
            COL_USERID:str = Events.COL_VISITOR_ID
            COL_ITEMID:str = Events.COL_ITEM_ID
            ratingsDF:DataFrame = dataset.eventsDF


        elif type(dataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            COL_USERID:str = Events.COL_USER_ID
            COL_ITEMID:str = Events.COL_OBJECT_ID
            ratingsDF:DataFrame = dataset.eventsDF

        self.ratingsGroupDF: DataFrame = ratingsDF.groupby(COL_USERID)[COL_ITEMID]
        # userProfileDF:DataFrame[userID:int, itemIDs:List[int]]
        userProfileDF: DataFrame = self.ratingsGroupDF.aggregate(lambda x: list(x))
        self.userProfiles: dict = userProfileDF.to_dict()
        s = ""


        if self._useMMR:
            #use MMR diversity, prepare input data
            self._toolMMR:ToolMMR = ToolMMR()
            self._toolMMR.init(dataset)


    def update(self, ratingsUpdateDF: DataFrame, argumentsDict: Dict[str, object]):
        if type(ratingsUpdateDF) is not DataFrame:
            raise ValueError("Argument ratingsTrainDF isn't type DataFrame.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        # ratingsUpdateDF has only one row
        row = ratingsUpdateDF.iloc[0]

        if type(self._trainDataset) is DatasetML:
            rating:int = row[Ratings.COL_RATING]
            userID:int = row[Ratings.COL_USERID]
            itemID:int = row[Ratings.COL_MOVIEID]
            if rating < 4:
                return

        elif type(self._trainDataset) is DatasetRetailRocket:
            from datasets.retailrocket.events import Events  # class
            userID:int = row[Events.COL_VISITOR_ID]
            itemID:int = row[Events.COL_ITEM_ID]

        elif type(self._trainDataset) is DatasetST:
            from datasets.slantour.events import Events  # class
            userID:int = row[Events.COL_USER_ID]
            itemID:int = row[Events.COL_OBJECT_ID]

        userTrainData = self.userProfiles.get(userID, [])
        userTrainData.append(itemID)
        self.userProfiles[userID] = userTrainData
        s = ""

    def resolveUserProfile(self, userProfileStrategy: str, userProfileSize: int, userTrainData: List[int]):
        rec: str = userProfileStrategy
        if self.DEBUG_MODE:
            print(rec)

        if (len(userTrainData) > 0):
            if (userProfileSize > 0):
                val = -1 * userProfileSize
                userTrainData = userTrainData[val:]

            if (rec == "mean") | (rec == "max"):
                weights = [1.0] * len(userTrainData)
            elif rec == "window3":
                userTrainData = userTrainData[-3:]
                weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]
            elif rec == "weightedMean":
                weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]

            if rec == "max":
                agg = np.max
            else:
                agg = np.mean

            if self.DEBUG_MODE:
                print((userTrainData, weights, agg))

            # print((userTrainData, weights, agg))
            return (userTrainData, weights, agg)

        return ([], [], "")

    def recommend(self, userID: int, numberOfItems: int, argumentsDict: Dict[str, object]):
        # print("userID: " + str(userID))
        if type(userID) is not int and type(userID) is not np.int64:
            raise ValueError("Argument userID isn't type int.")
        if type(numberOfItems) is not int and type(numberOfItems) is not np.int64:
            raise ValueError("Argument numberOfItems isn't type int.")
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict isn't type dict.")

        userProfileStrategy: str = argumentsDict[self.ARG_USER_PROFILE_STRATEGY]
        userProfileSize: str = argumentsDict[self.ARG_USER_PROFILE_SIZE]

        userTrainData: List[int] = self.userProfiles.get(userID, [])

        # adding currently viewed item (if any) into the user profile
        itemID = argumentsDict.get("itemID", 0)
        if itemID > 0:
            userTrainData.append(itemID)

        objectIDs: List[int]
        weights: List[float]
        objectIDs, weights, aggregation = self.resolveUserProfile(userProfileStrategy, userProfileSize, userTrainData)

        self._objectIDs = objectIDs

        simList: List = []

        if len(objectIDs) <= 0:
            return pd.Series([], index=[])
        # print(self.cbData)
        # provedu agregaci dle zvolené metody
        # print(objectIDs)
        validObjectIDs = self.cbData.index.intersection(set(objectIDs))  # some OIDs may be missing in CB data
        # print(validObjectIDs)
        objectIDs = [val for (i, val) in enumerate(objectIDs) if val in validObjectIDs]
        weights = [weights[i] for (i, val) in enumerate(objectIDs) if val in validObjectIDs]
        # print(objectIDs)
        if len(objectIDs) <= 0:
            return pd.Series([], index=[])

        results = self.cbData.loc[:, objectIDs]
        results.fillna(0.0)  # due to RetailRocket sparse structure
        # print(results.shape)

        self._results = results

        # print(results.shape)
        weights = np.asarray(weights)
        weights = weights[np.newaxis, :]
        results = results * weights
        # print(results.shape)
        results = aggregation(results, axis=1)
        # print(results.shape)
        if self.DEBUG_MODE:
            print(type(results))
        results.sort_values(ascending=False, inplace=True)


        if self._useMMR:
            print(results.iloc[0:numberOfItems*5])
            resultList = self._toolMMR.mmr_sorted(self._MMR_lambda, results.iloc[0:numberOfItems*5], numberOfItems)
        else:
            resultList = results.iloc[0:numberOfItems]


        if argumentsDict.get(self.ARG_ALLOWED_ITEMIDS) is not None:
            # ARG_ALLOWED_ITEMIDS contains a list of allowed IDs
            # TODO check type of ARG_ALLOWED_ITEMIDS, should be list
            # reducedList = self._sortedTheMostCommon.loc[self._sortedTheMostCommon.index.intersection(argumentsDict[self.ARG_ALLOWED_ITEMIDS])]
            results = results.loc[results.index.intersection(argumentsDict[self.ARG_ALLOWED_ITEMIDS])]
        resultList = results.iloc[0:numberOfItems]

        # normalize scores into the unit vector (for aggregation purposes)
        # !!! tohle je zasadni a je potreba provest normalizaci u vsech recommenderu - teda i pro most popular!
        finalScores = resultList.values
        finalScores = normalize(np.expand_dims(finalScores, axis=0))[0, :]

        return pd.Series(finalScores.tolist(), index=list(resultList.index))
