#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2021/4/16 16:01 
# @Author : hebin 
# @File : test_scores.py
import sys

import numpy as np
import pandas as pd

sys.path.append("../")

from src.irt import IRTModel


def test_scores():
    datas = pd.DataFrame({"item_0": [0, 1, 1, 0, 1, 0, 1],
                          "item_1": [0, 0, 1, 1, 1, 0, 1],
                          "item_2": [0, 0, 1, 1, 1, 0, 1],
                          "item_3": [0, 1, 1, 0, 1, 0, 1],
                          "item_4": [0, 1, 1, 0, 1, 0, 1],
                          })
    print("data shape={}".format(datas.shape))
    irt = IRTModel(5, model=None, model_save_path="./irt-model.jbl",
                   item_para_save_path="./irt-item-coef.jbl")
    irt.fit(datas, dims=1, itemtype="Rasch", method="EM")
    irt.calc_scores("MAP", datas.values, True)

    response_pattern = np.array([np.nan, np.nan, np.nan, np.nan, 1])
    irt.calc_scores("MAP", response_pattern, True)

    response_pattern = np.array([np.nan, np.nan, np.nan, np.nan, 1])
    irt.calc_scores("MAP", response_pattern, True)

    response_pattern = np.array([np.nan, np.nan, np.nan, 0, 1])
    irt.calc_scores("MAP", response_pattern, True)

    response_pattern = np.array([np.nan, np.nan, 0, 0, 1])
    irt.calc_scores("MAP", response_pattern, True)

    response_pattern = np.array([np.nan, 0, 0, 0, 1])
    irt.calc_scores("MAP", response_pattern, True)

    response_pattern = np.array([[np.nan, 0, 0, 0, 1], [0, 0, 0, 0, 1]])
    irt.calc_scores("MAP", response_pattern, True)


if __name__ == "__main__":
    test_scores()
