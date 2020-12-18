#!/usr/bin/python3

from typing import List

from pandas.core.frame import DataFrame #class

import pandas as pd
import os

from evaluationTool.evalToolDHondt import EvalToolDHondt #class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS #class

from datasets.retailrocket.events import Events #class



if __name__ == "__main__":

  #np.random.seed(42)
  #random.seed(42)

  os.chdir("..")
  #os.chdir("..")
  print(os.getcwd())

  userID:int = 565892
  currentItemID:int = 168952
  repetition:int = 0
  # 5919585    b11111100011100000000
  # 5947295    b11010110001101000010
  # 5980935    b11111000000010101000

  userID:int = 1043630
  currentItemID:int = 349104
  repetition:int = 0
  #5952135     b10111101000000000000
  #5962570     b11111100000100100000
  #5980920     b11111011101100101000
  #5980945     b11011100100110101100
  #5995975     b11111011000010001010

  eventsDF:DataFrame = Events.readFromFile()
  #evensDF = evensDF.loc[evens[Events.COL_EVENT] == "transaction"]

  eventsDF:DataFrame = eventsDF.sort_values(by=Events.COL_TIME_STAMP)

  users = eventsDF[Events.COL_VISITOR_ID].tolist()
  items = eventsDF[Events.COL_ITEM_ID].tolist()

  print(users[0])

  eventsDF = eventsDF.loc[eventsDF[Events.COL_VISITOR_ID] == userID]
  eventsDF = eventsDF.loc[eventsDF[Events.COL_ITEM_ID] == currentItemID]

#  for userIDI in users:
#      eventsDF = eventsDF.loc[eventsDF[Events.COL_VISITOR_ID] == userIDI]
#      eventsDF = eventsDF.loc[eventsDF[Events.COL_ITEM_ID] == currentItemID]
#      if len(eventsDF) > 0:
#        print(len(eventsDF))

  print(eventsDF.head(100))
