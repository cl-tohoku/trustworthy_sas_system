import fire
from itertools import product

# my libraries
from library.util import Util
from preprocess.base import PreprocessBase
from preprocess.supervising import PreprocessSupervising
from preprocess.superficial import PreprocessSuperficial
from train.base import TrainBase, TrainSupervising
from eval.base import EvalBase, EvalSupervising
from eval.lazy import Integration, CheckMasking
from eval.clustering import Clustering
from library.structure import *


class Main:
    def __init__(self, ):
        pass

    def preprocess(self, config_path, **kwargs):
        config = Util.load_preprocess_config(config_path)
        config.update(kwargs)
        PreprocessBase(config)()

    def sf_preprocess(self, config_path, **kwargs):
        config = Util.load_preprocess_config(config_path)
        config.update(kwargs)
        PreprocessSuperficial(config)()

    def train(self, train_config_path, **kwargs):
        # load configuration files
        train_config = Util.load_train_config(train_config_path)
        train_config.update(kwargs)
        trainer = TrainBase

        trainer(train_config)()

    def evaluate(self, eval_config_path, **kwargs):
        # load configuration files
        eval_config = Util.load_eval_config(eval_config_path)
        eval_config.update(kwargs)

        eval_class = EvalBase
        eval_class(eval_config)()

    def clustering(self, eval_config_path, **kwargs):
        eval_config = Util.load_eval_config(eval_config_path)
        eval_config.update(kwargs)
        Clustering(eval_config).make_clustering_datasets()

    def make_dataset(self, prompt_name="Y14_1213", limitation=0, contamination=False):
        print("Preprocess...")
        preprocess_config_path = "config/ys/preprocess/{}.yml".format(prompt_name)
        cont_path = "config/contamination/{}.yml".format(prompt_name) if contamination else ""
        # config に載せて実行
        self.preprocess(config_path=preprocess_config_path, preprocess_name=prompt_name, limitation=limitation,
                        contamination_path=cont_path)

    # 1st round
    def execute(self, prompt_name="Y14_XXXX", script_name="Y14_XXXX_XX", config_file_name="template.yml",
                mode="standard", sf_term=None, sf_idx=None, training=True, evaluation=True, clustering=True):

        # 訓練
        if training:
            print("Training...")
            train_config_path = "config/ys/train/{}".format(config_file_name)
            # override arguments
            self.train(train_config_path=train_config_path, preprocess_name=prompt_name, mode=mode,
                       script_name=script_name, sf_term=sf_term, sf_idx=sf_idx)

        # 評価
        if evaluation:
            print("Evaluating...")
            eval_config_path = "config/ys/eval/{}".format(config_file_name)
            self.evaluate(eval_config_path=eval_config_path, preprocess_name=prompt_name, mode=mode,
                          script_name=script_name)

        # クラスタリング
        if clustering:
            print("Clustering...")
            eval_config_path = "config/ys/eval/{}".format(config_file_name)
            self.clustering(eval_config_path=eval_config_path, preprocess_name=prompt_name, mode=mode,
                            script_name=script_name)

    # 2nd round
    def sv_preprocess(self, config_file_path=None, **kwargs):
        prep_config_path = "config/ys/supervising/preprocess.yml" if config_file_path is None else config_file_path
        config = Util.load_config(prep_config_path, Config=SVPreprocessConfig)
        config.update(kwargs)
        PreprocessSupervising(config).execute()

    def sv_train(self, config_file_name="supervising.yml", model_path="", term="A", **kwargs):
        train_config_path = "config/ys/train/{}".format(config_file_name)
        train_config = Util.load_train_config(train_config_path)
        train_config.update(kwargs)
        train_config.mode = "sv"
        trainer = TrainSupervising
        target_idx = ord("A") - ord(term)
        trainer(train_config, model_path, target_idx)()

    def sv_eval(self, config_file_name="supervising.yml", **kwargs):
        # load configuration files
        eval_config_path = "config/ys/eval/{}".format(config_file_name)
        eval_config = Util.load_eval_config(eval_config_path)
        eval_config.update(kwargs)
        eval_config.mode = "sv"

        eval_class = EvalSupervising
        eval_class(eval_config)()

    def sv_clustering(self, config_file_name="supervising.yml", **kwargs):
        # load configuration files
        eval_config_path = "config/ys/eval/{}".format(config_file_name)
        eval_config = Util.load_eval_config(eval_config_path)
        eval_config.update(kwargs)
        eval_config.mode = "sv"

        Clustering(eval_config).make_clustering_datasets()

    def fitness(self, eval_dir, script_name, **kwargs):
        Integration().quantitative_fitness(eval_dir,  script_name)

    def check_masking(self, eval_dir, script_name, masking_span, term):
        CheckMasking().check_masking_efficiency(eval_dir, script_name, masking_span, term)


if __name__ == "__main__":
    fire.Fire(Main)
