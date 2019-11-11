#!/usr/bin/python3

from recommendation.resultOfRecommendation import ResultOfRecommendation #class

class ResultsOfRecommendations:

   # recommenderIDs:list<str>, resultOfRecommendation:list<ResultOfRecommendation>
   def __init__(self, recommenderIDs=[], resultsOfRecommendations=[]):
      if type(recommenderIDs) is not list:
         raise ValueError("Argument recommenderIDs is not type list.")
      if type(resultsOfRecommendations) is not list:
         raise ValueError("Argument resultsOfRecommendations is not type list.")

      for recommenderIDI in recommenderIDs:
         if type(recommenderIDI) is not str:
            print(type(recommenderIDI))
            raise ValueError("Argument recommenderIDs contains an item of the wrong type (not str).")
      for resultOfRecommendationI in resultsOfRecommendations:
         if type(resultOfRecommendationI) is not ResultOfRecommendation:
            raise ValueError("Argument ratings contains an item of the wrong type (not ResultOfRecommendation).")

      if len(recommenderIDs) != len(list(dict.fromkeys(recommenderIDs))):
         raise ValueError("Argument recommenderIDs contains duplicate recommenderIDs values.")
      if len(recommenderIDs) != len(resultsOfRecommendations):
         raise ValueError("Arguments recommenderIDs (list) and resultsOfRecommendations (list) are not the same length.")

      self._recommenderIDs = recommenderIDs;
      self._resultsOfRecommendations = resultsOfRecommendations;


   # recommenderID:str, resultOfRecommendation:ResultOfRecommendation
   def add(self, recommenderID, resultOfRecommendation):
      if type(recommenderID) is not str:
         raise ValueError("Argument recommenderID is not type str.")
      if type(resultOfRecommendation) is not ResultOfRecommendation:
         raise ValueError("Argument resultsOfRecommendations is not type ResultOfRecommendation.")

      if type(recommenderID) in self._recommenderIDs:
         raise ValueError("Argument recommenderID is already included.")
       
      self._recommenderIDs.append(recommenderID)
      self._resultsOfRecommendations.append(resultOfRecommendation)


   def exportAsDictionaryOfSeries(self):
      # dictionary:Dictionary<str, ResultOfRecommendation>
      dictionary = {}
      for indexI in range(len(self._recommenderIDs)):
         recommenderIDI = self._recommenderIDs[indexI]
         resultsOfRecommI = self._resultsOfRecommendations[indexI]
         dictionary[recommenderIDI] = resultsOfRecommI.exportAsSeries()
      return dictionary;

