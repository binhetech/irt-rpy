#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2021/4/12 19:26 
# @Author : hebin 
# @File : irt_simulator.py
import joblib
import json
import numpy as np

from irt import IRTModel


class IRTSimulator(object):
    """
    利用IRT理论进行自适应学习题目推荐效果评估.
    """

    def __init__(self, item_num, model_save_path, item_index_path, item_para_save_path):
        model = joblib.load(open(model_save_path, "rb"))
        self.model = IRTModel(item_num, model, model_save_path=model_save_path, item_para_save_path=item_para_save_path)
        self.item_index = joblib.load(open(item_index_path, "rb"))
        self.item_responses = [np.nan for _ in range(item_num)]
        self.problem_id2item_id = {str(v): k for k, v in enumerate(self.item_index)}
        print("IRT simulator loaded (item num={})".format(len(self.problem_id2item_id)))

    def get_response_patterns(self, problem_seq, result_seq):
        value = self.item_responses[:]
        for p, a in zip(problem_seq, result_seq):
            if str(p) in self.problem_id2item_id:
                value[self.problem_id2item_id[str(p)]] = a
        return value

    def evaluate_theta(self, responses, method="EAP"):
        if len([i for i in responses if not np.isnan(i)]) == 0:
            return -1, -1, 0
        try:
            status = 0
            assert len(responses) == len(self.item_responses)
            theta = self.model.calc_scores(method, np.array(responses))[0][0]
            # 能力值归一化
            score = self.model.normalization(theta)
        except Exception:
            status = -1
            theta = -1
            score = 0
        return status, theta, score

    def get_item_prob_answers(self, theta, cur_problem_seq):
        answers = []
        for i in cur_problem_seq:
            if str(i) in self.problem_id2item_id:
                # 基于当前能力估计答题概率
                item_id = self.problem_id2item_id[str(i)]
                ans = self.model.get_item_expected_value(item_id, theta)
                ans = 0 if ans < 0.5 else 1
                answers.append(ans)
            else:
                answers.append(0)
        return answers

    def evaluate_ep(self, history_problem_seq, history_result_seq, recommend_topic_seq, method="predict"):
        try:
            history_response_patterns = self.get_response_patterns(history_problem_seq, history_result_seq)
            # 估计历史答题记录的能力
            status, theta, Es = self.evaluate_theta(history_response_patterns)
            if status < 0:
                return status, 0
            cur_problem_seq = history_problem_seq + recommend_topic_seq
            if method == "predict":
                # 基于能力估计预测答题正确的概率，生成答题结果
                answers = self.get_item_prob_answers(theta, recommend_topic_seq)
            else:
                # 构造老师答题结果
                answers = [1 for i in recommend_topic_seq]
            cur_result_seq = history_result_seq + answers
            cur_response_patterns = self.get_response_patterns(cur_problem_seq, cur_result_seq)
            status, cur_theta, Ee = self.evaluate_theta(cur_response_patterns)
            if status < 0:
                return status, 0
            Ep = (Ee - Es) / (1 - Es)
        except Exception:
            status = -1
            Ep = -1
        return status, Ep
