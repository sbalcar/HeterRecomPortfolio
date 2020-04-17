#!/usr/bin/python3

class User:
  def __init__(self, userID:int, age:int, sex:str, occupation:str, zipCode:int):
     if type(userID) is not int:
        raise ValueError("Argument userID isn't type int.")
     if type(age) is not int:
        raise ValueError("Argument age isn't type int.")
     if type(sex) is not str:
        raise ValueError("Argument sex isn't type str.")
     if type(occupation) is not str:
        raise ValueError("Argument occupation isn't type str.")
     if type(zipCode) is not str:
        raise ValueError("Argument zipCode isn't type str.")

     self.userID:int = userID
     self.age:int = age
     self.sex:str = sex
     self.occupation:str = occupation
     self.zipCode:str = zipCode


  def exportAsList(self):
     return [self.itemID, self.name]