#!/usr/bin/python3


class Rating:
  def __init__(self, userID:int, itemID:int, rating:float, timestamp:int):
     if type(userID) is not int:
        raise ValueError("Argument userID isn't type int.")
     if type(itemID) is not int:
        raise ValueError("Argument itemID isn't type int.")
     if type(rating) is not float:
        raise ValueError("Argument rating isn't type float.")
     if type(timestamp) is not int:
        raise ValueError("Argument timestamp isn't type int.")
     self.userID:int = userID
     self.itemID:int = itemID
     self.rating:float = rating
     self.timestamp:int = timestamp

  def exportAsList(self):
     return [self.userID, self.itemID, self.rating, self.timestamp]