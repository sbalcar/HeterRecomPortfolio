#!/usr/bin/python3

from typing import List

from pandas.core.series import Series #class

from datasets.ratings import Ratings #class

import pandas as pd
import numpy as np

from pandas.core.frame import DataFrame #class

from simulation.recommendToUser.simulatorOfPortfoliosRecommToUser import ModelOfIndexes #class


def test01():

    df:DataFrame = DataFrame({'$a':[2783,2783,2783,3970], '$b':[1909,1396,2901,3408], '$c':[2,4,5,4]})
    df.columns = [Ratings.COL_USERID, Ratings.COL_MOVIEID, Ratings.COL_RATING]
    print(df)

    m:ModelOfIndexes = ModelOfIndexes(df)

    a1:int = m.getNextIndex(2783, 1909)
    print(a1)

    a2:int = m.getNextIndex(2783, 1396)
    print(a2)

    a3:int = m.getNextIndex(2783, 2901)
    print(a3)


test01()