#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/2/24 15:11
# @Author : hebin
# @File : irt.py


import joblib
import time
import numpy as np

# python call R
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri

# activate
pandas2ri.activate()
numpy2ri.activate()

# import package
mirt = importr('mirt')
mirtCAT = importr('mirtCAT')


class IRTModel(object):

    def __init__(self, item_num, model=None, model_save_path="./resource/irt-model.jbl",
                 item_para_save_path="./resource/irt-item-coef.jbl"):
        """
        IRT类初始化.

        Args:
            item_num: int, 题库中试题数量
            model: int, 已估计完成的irt模型文件，使用joblib存储
            model_save_path: string, 估计模型后模型保存路径
            item_para_save_path: string, 估计模型后试题参数保存路径

        """

        self.student_num = 0
        self.item_num = item_num
        # list of numpy.array
        self.item_paras = []
        self.item_obj = [object] * item_num
        # 当前试题匹配的能力theta
        self.item_thetas = []
        self.model_save_path = model_save_path
        self.item_para_save_path = item_para_save_path

        # 加载已完成参数估计的模型对象文件
        self.model = model
        if model is not None:
            # 已估计的模型参数重计算
            self.calc_coef()
            # 计算每道试题对应的最佳匹配能力值
            self.extract_items_match_thetas()

    def define_irt_model(self, data, itemtype="3PL", save=True):
        """
        根据人工经验设置irt参数(如3PL: a, d, g, u)来定义模型.

        Args:
            data: np.array, 已设置好的试题参数，
                item_1: a1, d, g, u
                item_2: a1, d, g, u
                ...
            itemtype: string, 模型类型， "3PL", "2PL"

        Returns:

        """
        data = np.array(data)
        # 更新试题参数
        self.item_num = data.shape[0]
        # 需要对R变量赋值
        ro.r.assign("pars", numpy2ri.py2rpy(data))
        self.model = ro.r("generate.mirt_object(pars, '%s')" % itemtype)
        if save:
            with open(self.model_save_path, "wb") as f:
                joblib.dump(self.model, f)
            print("{} model saved to {}".format(itemtype, self.model_save_path))
        return self.model

    def simdata(self, num):
        """
        模拟答题数据response patterns.

        Args:
            num: int, 返回数据个数

        Returns:

        """
        # 需要对R变量赋值
        ro.r.assign("model", self.model)
        # 运行R函数
        data = ro.r("simdata(model=model, N=%d)" % num)
        return data

    def fit(self, data, dims=1, itemtype="3PL", method="EM", save=True):
        """
        基于答题/响应数据进行IRT模型参数估计.

        Args:
            data: pandas.DataFrame, 行是学生，列是题目(item0, item1, item2, ...)
            dims: mirt.model模型对象或数字指示维度大小, 单维项目反映理论设置为1
            itemtype: 项目参数类型，包括'Rasch', '2PL', '3PL', '3PLu', '4PL'.
                    Form must be: (discrimination, difficulty, lower-bound, upper-bound)
            method: "EM", "MCEM"
            save: boolean, 是否保存模型对象文件

        Returns:

        """
        t_start = time.time()
        self.student_num, self.item_num = data.shape
        # pandas.DataFrame to R objects
        data = pandas2ri.py2rpy(data)
        # 需要对R变量赋值
        ro.r.assign("data", data)
        # 运行R函数
        self.model = ro.r(
            "mirt(data,model=%d,itemtype='%s',method='%s',removeEmptyRows=TRUE)" % (dims, itemtype, method))
        print("{} model fit completed, elapsed time={}".format(itemtype, time.time() - t_start))
        if save:
            with open(self.model_save_path, "wb") as f:
                joblib.dump(self.model, f)
            print("IRT model saved to {}".format(self.model_save_path))

    def calc_coef(self, IRTpars="F", save=True):
        """
        计算已拟合模型的项目item参数.

        Args:
            IRTpars: string, 是否转换为原始IRT参数
            save: boolean, 是否保存参数文件

        Returns:

        """
        t_start = time.time()
        ro.r.assign("model", self.model)
        coef = ro.r("paras <- coef(model, as.data.frame=F, IRTpars=%s)" % IRTpars)
        coef = coef[:self.item_num]
        self.item_paras = [i[0] for i in coef]
        self.item_obj = [object] * self.item_num
        if save:
            with open(self.item_para_save_path, "wb") as f:
                joblib.dump(self.item_paras, f)
        # print("coef calculated completed. time={}".format(time.time() - t_start))
        return self.item_paras

    def normalization(self, x):
        """
        难度值b归一化.
        y = exp(x)/(1+exp(x))
        x = log(y/(1-y))

        Args:
            x:

        Returns:

        """
        return np.exp(x) / (np.exp(x) + 1)

    def calc_scores(self, method="EAP", response_pattern=None, verbose=False):
        """
        已知试题参数，估计被试者的能力分数.

        Args:
            method: string, "EAP", "MAP", "ML"
            response_pattern: numpy, 答题结果，注意shape == item_nun, 如果为None, 则返回原模型的估计能力值

        Returns:
            scores: list of [f1 score, standard error]

        """
        t_start = time.time()
        ro.r.assign("model", self.model)
        if response_pattern is not None:
            # print(response_pattern)
            rp = numpy2ri.py2rpy(response_pattern)
            ro.r.assign("response_pattern", rp)
            scores = ro.r(
                "fscores(model, method='%s', full.scores=TRUE, full.scores.SE=TRUE, response.pattern=response_pattern, append_response.pattern=FALSE)" % method)
        else:
            scores = ro.r(
                "fscores(model, method='%s', full.scores=TRUE, full.scores.SE=TRUE)" % method)
        if verbose:
            print("{} ability estimate {} completed, time={}, theta={}".format(method, len(scores.tolist()),
                                                                               time.time() - t_start, scores))
        return scores

    def extract_items_match_thetas(self):
        """
        提取试题匹配的能力值.
        """
        item_thetas = []
        for i, p in enumerate(self.item_paras):
            self.item_obj[i] = self.extract_item(i)
            theta = self.extract_item_theta(self.item_obj[i])
            item_thetas.append(theta)
        self.item_thetas = np.array(item_thetas)
        return self.item_thetas

    def extract_item(self, item_id):
        """
        根据试题id提取item对象.

        Args:
            item_id:

        Returns:

        """
        ro.r.assign("model", self.model)
        ro.r.assign("item", item_id + 1)
        # R语言中数组是以1开始的
        extra = ro.r("extract.item(model, item, group=NULL, drop.zeros = FALSE)")
        return extra

    def get_item_info(self, item_id, theta):
        ro.r.assign("x", self.item_obj[item_id])
        ro.r.assign("Theta", theta)
        value = ro.r("iteminfo(x, Theta, degrees = NULL, total.info = TRUE, multidim_matrix = FALSE)")
        return value

    def get_items_info(self, cur_theta):
        values = [[i, abs(self.item_thetas[i] - cur_theta)] for i, p in enumerate(self.item_paras)]
        return values

    def get_item_expected_value(self, item_id, theta):
        ro.r.assign("x", self.item_obj[item_id])
        ro.r.assign("Theta", theta)
        value = ro.r("expected.item(x, Theta)")
        return value

    def extract_item_theta(self, x):
        ro.r.assign("x", x)
        theta = np.arange(-6, 6.01, 0.01)
        ro.r.assign("Theta", theta)
        value = ro.r("iteminfo(x, Theta, degrees = NULL, total.info = TRUE, multidim_matrix = FALSE)")
        value = [(i, j) for i, j in zip(theta, value)]
        # 计算最大信息对应的theta
        theta = sorted(value, key=lambda i: i[1], reverse=True)[0][0]
        return theta
