# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 12:00:49 2022

@author: pc
"""
import pandas as pd

df = pd.read_csv('heart.csv', index_col=None)

df.info()
df.head()
