#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2021/4/16 15:45 
# @Author : hebin 
# @File : test_lsat6_2pl.py
import sys

import numpy as np
import pandas as pd

sys.path.append("../")

from src.irt import IRTModel


def test_lsat6():
    # A data frame with the responses of 1000 individuals to 5 questions
    data = open("../examples/lsat6.dat", "r", encoding="utf-8").readlines()
    datas = pd.DataFrame()
    for it, i in enumerate(data):
        cols = i.strip().split("\t")
        cols = [int(i) for i in cols]
        datas = datas.append(pd.DataFrame([pd.Series(cols).add_prefix("item_")]), ignore_index=True)
    print("data shape={}".format(datas.shape))
    irt = IRTModel(5, model=None, model_save_path="./irt-model.jbl", item_para_save_path="./irt-item-coef.jbl")
    # 试题、能力参数联合估计
    irt.fit(datas, dims=1, itemtype="2PL", method="EM")
    irt.calc_coef(IRTpars="F")
    # 能力参数估计
    scores = irt.calc_scores()
    print("scores={}".format(scores))
    for i, p in enumerate(irt.item_paras):
        print("item {} paras: {}".format(i, p))


def test_lsat6_na():
    # A data frame with the responses of 1000 individuals to 5 questions
    data = open("../examples/lsat6.dat", "r", encoding="utf-8").readlines()
    datas = pd.DataFrame()
    for it, i in enumerate(data):
        cols = i.strip().split("\t")
        cols = [int(i) for i in cols]
        if it == 0:
            # 答题数据缺失时，用np.nan填充
            cols = [0, 1, np.nan, 1, 1]
        datas = datas.append(pd.DataFrame([pd.Series(cols).add_prefix("item_")]), ignore_index=True)
    print("data shape={}".format(datas.shape))

    # 答题数据缺失时
    datas.iloc[0]["item_1"] = np.nan
    datas.iloc[3]["item_2"] = np.nan

    irt = IRTModel(5, model=None, model_save_path="./irt-model.jbl", item_para_save_path="./irt-item-coef.jbl")
    # 试题、能力参数联合估计
    irt.fit(datas, dims=1, itemtype="2PL", method="EM")
    irt.calc_coef(IRTpars="F")
    for i, p in enumerate(irt.item_paras):
        print("item {} paras: {}".format(i, p))


if __name__ == "__main__":
    test_lsat6()
    test_lsat6_na()
