import torch
import torch.nn.functional as F
import sys
import os
import pandas as pd
import string
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from itertools import islice
import numpy as np
from transformers import BertJapaneseTokenizer
from sklearn.metrics import recall_score, precision_score

sys.path.append("..")
from library.util import Util
from library.quadratic_weighted_kappa import quadratic_weighted_kappa
from eval.attribution import FeatureAttribution


class EvalBase:
    def __init__(self, eval_config):
        self.config = eval_config
        self.prep_type = self.config.preprocessing_type
        self.model_config = Util.load_model_config(eval_config.model_config_path)
        self.prompt_config = Util.load_prompt(self.config)
        self.model = None
        self.unique_id = eval_config.unique_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # set output size for prompt
        self.model_config.output_size = self.select_model_output()
        self.tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char-v2")

    def select_model_output(self):
        if self.config.target_type == "analytic":
            return self.prompt_config.scoring_item_num
        else:
            return 1

    def get_max_score(self, i=None):
        if self.config.target_type == "analytic":
            return self.prompt_config.max_scores[i]
        else:
            return sum(self.prompt_config.max_scores)

    def predict(self, script, sent_vector=False):
        input_ids = torch.tensor(script["input_ids"].to_list(), device=self.device)
        if self.prep_type == "bert":
            token_type_ids = torch.tensor(script["token_type_ids"].to_list(), device=self.device)
            attention_mask = torch.tensor(script["attention_mask"].to_list(), device=self.device)
            return self.model(input_ids, token_type_ids, attention_mask, sent_vector=sent_vector)
        else:
            return self.model(input_ids, sent_vector=sent_vector)

    def script_extractor(self, script):
        # input
        input_ids = torch.tensor(script["input_ids"].to_list(), device=self.device)
        if self.prep_type == "bert":
            token_type_ids = torch.tensor(script["token_type_ids"].to_list(), device=self.device)
            attention_mask = torch.tensor(script["attention_mask"].to_list(), device=self.device)
            additional_args = (token_type_ids, attention_mask)
        else:
            additional_args = tuple()

        # output
        if self.config.target_type == "analytic":
            output = script["score_vector"].to_list()[0]
        else:
            output = [script["score"].to_list()[0]]

        # annotation
        annotation = script["annotation_matrix"].to_list()[0]

        return input_ids, additional_args, output, annotation

    def id_to_string_list(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def calc_rmse(self, pred_scores, gold_scores, max_score):
        pred_scores = pred_scores / max_score
        gold_scores = gold_scores / max_score
        if self.config.target_type == "analytic":
            return float(torch.sqrt(F.mse_loss(pred_scores, gold_scores)))
        elif self.config.target_type == "individual":
            return float(torch.sqrt(F.mse_loss(pred_scores, gold_scores)))
        else:
            raise RuntimeError("Invalid target type")

    def calc_qwk(self, pred_scores, gold_scores, max_score):
        if self.config.target_type == "analytic":
            qwk = quadratic_weighted_kappa(pred_scores, gold_scores, min_rating=0, max_rating=max_score)
            return float(qwk)
        elif self.config.target_type == "individual":
            qwk = quadratic_weighted_kappa(pred_scores, gold_scores, min_rating=0, max_rating=max_score)
            return float(qwk)
        else:
            raise RuntimeError("Invalid target type")

    def eval_scripts(self, dataset):
        results = defaultdict(list)
        for script in tqdm(dataset.iterrows()):
            script = pd.DataFrame(script[1]).T
            with torch.no_grad():
                output = self.predict(script).squeeze(0).to("cpu")
            results["prediction"].append(output.tolist())
            results["prediction_round"].append(torch.round(output).long().tolist())
        results["text"] = dataset["text"].to_list()
        results["score"] = dataset["score"].to_list()
        results["score_vector"] = dataset["score_vector"].to_list()
        return results

    def eval_performance(self, results):
        performances = defaultdict(list)
        pred_score_tensor = torch.FloatTensor(results["prediction"]).clone()
        if self.config.target_type == "analytic":
            gold_score_tensor = torch.FloatTensor(results["score_vector"]).clone()
        else:
            gold_score_tensor = torch.FloatTensor([results["score"]]).clone().T
        prediction_round_tensor = torch.zeros_like(pred_score_tensor)

        for i in range(self.select_model_output()):
            scoring_item_name = chr(ord('A') + i)
            max_score = self.get_max_score(i)
            gold_scores = gold_score_tensor[:, i]
            pred_scores = torch.round(torch.clamp(pred_score_tensor[:, i], 0, max_score))
            prediction_round_tensor[:, i] = pred_scores
            qwk = self.calc_qwk(pred_scores, gold_scores, max_score)
            rmse = self.calc_rmse(pred_scores, gold_scores, max_score)

            performances["item"].append(scoring_item_name)
            performances["RMSE"].append(rmse)
            performances["QWK"].append(qwk)

        return performances

    def get_sentence_vector(self, script):
        output, sent_vector = self.predict(script, sent_vector=True)
        return sent_vector.cpu().flatten().tolist()

    def int_grad_metric(self, attribution, annotation):
        anot = np.array(annotation)
        annotated_idx = np.reshape(np.argwhere(anot == 1), (-1))
        real_output = np.sum(np.array(attribution)[annotated_idx])
        all_output = np.sum(np.array(attribution))
        # grad_score = real_output / ideal_output
        grad_score = real_output / all_output
        return grad_score

    def overlap_metric(self, attribution, annotation):
        anot = np.array(annotation)
        justification_size = np.sum(anot)
        attribution_idx = np.argsort(attribution)[::-1][:justification_size]
        binary_attribution = np.zeros(len(attribution))
        binary_attribution[attribution_idx] = 1
        recall = recall_score(y_true=anot, y_pred=binary_attribution, zero_division=0)
        precision = precision_score(y_true=anot, y_pred=binary_attribution, zero_division=0)
        return recall, precision

    def eval_attributions(self, dataset):
        results = defaultdict(list)
        fitness = defaultdict(list)
        attr = FeatureAttribution(self.config, self.model)
        for idx, script in enumerate(tqdm(dataset.iterrows())):
            script = pd.DataFrame(script[1]).T
            input_ids, args, gold_label, annotation_matrix = self.script_extractor(script)
            # predict
            with torch.no_grad():
                output = self.predict(script)
                pred_label = torch.round(output).squeeze(0).long().to("cpu").tolist()
                pred_value_list = output.squeeze(0).cpu().tolist()


            sentence_vector = self.get_sentence_vector(script)
            for target in range(len(gold_label)):
                alpha = string.ascii_uppercase[target]
                if annotation_matrix:
                    results["Annotation"].append(annotation_matrix[target])
                    results["Annotation_All"].append(annotation_matrix)

                # score
                results["Gold"].append(gold_label[target])
                results["Pred"].append(pred_label[target])

                # deprecated
                results["Idx"].append(idx)
                results["Sample_ID"].append(idx)
                results["Term"].append(alpha)
                results["Max_Score"].append(self.get_max_score(target))

                # for finetuning
                results["Sentence_Vector"].append(sentence_vector)
                input_np = input_ids.squeeze(0).to("cpu").numpy()
                results["Token"].append(self.id_to_string_list(input_np))
                results["Input_IDs"].append(input_np)
                results["Token_Type_IDs"].append(args[0].squeeze(0).to("cpu").numpy())
                results["Attention_Mask"].append(args[1].squeeze(0).to("cpu").numpy())
                results["Score_Vector"].append(gold_label)

                # for xai
                attribution, int_grad, emb, baseline_value = attr.calc_gradient(input_ids, target, args, multiply=True)
                prediction_value = pred_value_list[target]
                results["Attribution"].append(attribution)
                results["Integrated_Gradients"].append(int_grad)
                results["Embedding"].append(emb)
                results["Baseline_Value"].append(baseline_value)
                results["Prediction_Value"].append(prediction_value)
                results["Attribution_Value"].append(float(np.sum(int_grad)))

                # xai eval
                int_score = self.int_grad_metric(attribution, annotation_matrix[target])
                recall, precision = self.overlap_metric(attribution, annotation_matrix[target])
                fitness["Int_Score"].append(int_score)
                fitness["Recall_Score"].append(recall)
                fitness["Precision_Score"].append(precision)
                print('\rR:{:.5f}, P:{:.5f}'.format(recall, precision), end='')

        return results, fitness

    def calc_ranker(self, input_ids, attribution):
        attr_ranking = np.argsort(attribution)[::-1]
        ids = input_ids.squeeze(0).to("cpu").numpy()
        ranker_token_ids = ids[attr_ranking[:20]].tolist()
        ranker_string = self.id_to_string_list(ranker_token_ids)
        return ranker_string, attr_ranking

    def eval(self, dataset, data_type):
        self.model.eval()

        results = self.eval_scripts(dataset)
        performances = self.eval_performance(results)

        dataframe = pd.DataFrame(results)
        dataframe_performance = pd.DataFrame(performances)
        self.dump_results(dataframe, suffix="results", data_type=data_type)
        self.dump_results(dataframe_performance, suffix="performances", data_type=data_type)

        if self.config.attribution:
            attr_results = self.eval_attributions(dataset)
            dataframe_attribution, fitness_df = pd.DataFrame(attr_results)
            print("Outputting...")
            self.dump_results(dataframe_attribution, suffix="attributions", data_type=data_type, csv=False)
            self.dump_results(fitness_df, suffix="fitness", data_type=data_type, csv=True)

    def dump_results(self, dataframe, data_type, suffix, csv=True):
        Util.save_eval_df(dataframe, self.config, data_type, suffix, csv, finetuning=self.config.finetuning)

    def __call__(self):
        self.model = Util.load_model(self.config, self.model_config, heuristics=self.config.finetuning)
        # test set
        print("Test")
        test_dataset = Util.load_dataset(self.config, "test",)
        self.eval(test_dataset, "test")
        # train set
        print("Train")
        train_dataset = Util.load_dataset(self.config, "train",)
        self.eval(train_dataset, "train")


class EvalStatic:
    @staticmethod
    def cross_validation(eval_config, eval=EvalBase, k=5):
        for idx in tqdm(range(k)):
            eval_config.validation = True
            eval_config.validation_idx = idx
            eval(eval_config)()
