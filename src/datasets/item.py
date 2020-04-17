#!/usr/bin/python3


class Item:
  def __init__(self, itemID:int, name:str):
     if type(itemID) is not int:
        raise ValueError("Argument itemID isn't type int.")
     if type(name) is not str:
        raise ValueError("Argument name isn't type str.")

     self.itemID:int = itemID
     self.name:str = name


  def exportAsList(self):
     return [self.itemID, self.name]