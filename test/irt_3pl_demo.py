#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2021/4/16 15:42 
# @Author : hebin 
# @File : irt_3pl_demo.py
import sys

import json
import joblib
import numpy as np
import pandas as pd

sys.path.append("../")

from src.irt import IRTModel


def irt_3pl_demo():
    # 模拟答题矩阵
    datas = pd.DataFrame(np.random.randint(0, 2, (1000, 10)))
    print("data shape={}".format(datas.shape))
    irt = IRTModel(10, model=None, model_save_path="./resource/demo-irt-model.jbl",
                   item_para_save_path="./resource/demo-irt-item-coef.jbl")
    irt.fit(datas, dims=1, itemtype="3PL", method="EM")
    irt.calc_coef(IRTpars="F")
    irt.extract_items_match_thetas()


if __name__ == "__main__":
    irt_3pl_demo()
