
from recommender.aRecommender import ARecommender  # class

#!/usr/bin/python3

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from sklearn.neighbors import NearestNeighbors

from datasets.ml.ratings import Ratings  # class

from history.aHistory import AHistory #class


class RecommenderItemBasedKNN(ARecommender):
    def __init__(self, jobID:str, argumentsDict:dict):
        if type(argumentsDict) is not dict:
            raise ValueError("Argument argumentsDict is not type dict.")

        self._jobID = jobID
        self._argumentsDict: dict = argumentsDict
        self._KNNs:DataFrame = None
        self._movieFeaturesDF:DataFrame = None
        self._modelKNN: NearestNeighbors = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20,
                                                            n_jobs=-1)
        self._lastRatedItemPerUser:DataFrame = None
        self._itemsDF:DataFrame = None


    def train(self, history:AHistory, ratingsTrainDF:DataFrame, usersDF:DataFrame, itemsDF:DataFrame):
        # TODO: Check input/object data integrity!

        self._itemsDF = itemsDF
        self._movieFeaturesDF:DataFrame = ratingsTrainDF.pivot(
            index='movieId',
            columns='userId',
            values='rating'
        ).fillna(0)

        self._modelKNN.fit(self._movieFeaturesDF)

        self.KNNs:DataFrame = self._modelKNN.kneighbors(n_neighbors=100, return_distance=False)

        # get last positive feedback from each user
        self._lastRatedItemPerUser = \
            ratingsTrainDF[ratingsTrainDF['rating'] > 3].sort_values('timestamp')\
                .groupby('userId').tail(1).set_index('userId').drop(columns=['rating', 'timestamp'])

    def update(self, ratingsUpdateDF:DataFrame):
        # TODO: Check input/object data integrity!

        row:DataFrame = ratingsUpdateDF.iloc[0]

        userID:int = row[Ratings.COL_USERID]
        objectID:int = row[Ratings.COL_MOVIEID]
        rating:int = row[Ratings.COL_RATING]

        self._movieFeaturesDF[userID][objectID] = rating

        # update last positive feedback
        if rating > 3:
            self._lastRatedItemPerUser['movieId'][userID] = objectID

        self._modelKNN.fit(self._movieFeaturesDF)
        self.KNNs:DataFrame = self._modelKNN.kneighbors(n_neighbors=100, return_distance=False)




    def recommend(self, userID:int, numberOfItems:int=20, argumentsDict:dict={}):
        # TODO: Check input/object data integrity!

        # Check if user is known
        if userID not in self.KNNs:
            # TODO: How to behave if yet no rating from user was recorded? Maybe return TOP-N most popular items?
            return Series()

        # Get recommendations for user
        lastRatedItemFromUser:int = self._lastRatedItemPerUser.loc[userID]['movieId']
        result:Series = Series(self.KNNs[lastRatedItemFromUser][:numberOfItems])

        if self._jobID == 'test':
            print("Last visited film:")
            print(self._itemsDF[self._itemsDF['movieId'] == lastRatedItemFromUser])
            print("List of recommendations:")
            for film in result:
                film_info:DataFrame = self._itemsDF[self._itemsDF['movieId'] == film]
                print('\t', film_info['movieTitle'].to_string(header=False), film_info['Genres'].to_string(header=False))

        return result
