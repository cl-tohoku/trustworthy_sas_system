import fire
from itertools import product

# my libraries
from library.util import Util
from preprocess.base import PreprocessBase
from preprocess.finetuning import PreprocessFinetuning
from preprocess.masking import PreprocessMasking
from preprocess.supervising import PreprocessSupervising
from preprocess.contamination import PreprocessContamination
from train.base import TrainBase, TrainSupervising
from train.base import TrainStatic
from train.finetuning import TrainFinetuning
from train.masking import TrainMasking
from eval.base import EvalBase
from eval.lazy import Integration, CheckMasking
from eval.base import EvalStatic
from eval.clustering import Clustering
from eval.finetuning import EvalFinetuning
from eval.masking import EvalMasking, ClusteringMasking


class Main:
    def __init__(self, ):
        pass

    def preprocess(self, config_path, **kwargs):
        config = Util.load_preprocess_config(config_path)
        config.update(kwargs)
        PreprocessBase(config)()
        if config.contamination_path:
            PreprocessContamination(config)()
        if config.masking_path:
            PreprocessMasking(config)()

    def train(self, train_config_path, **kwargs):
        # load configuration files
        train_config = Util.load_train_config(train_config_path)
        train_config.update(kwargs)
        trainer = TrainBase

        # sweep
        if train_config.sweep:
            TrainStatic.sweep(train_config, trainer)

        # 5 cross-validation
        elif train_config.validation:
            TrainStatic.cross_validation(train_config, trainer)

        # simple experiment
        else:
            trainer(train_config)()

    def evaluate(self, eval_config_path, **kwargs):
        # load configuration files
        eval_config = Util.load_eval_config(eval_config_path)
        eval_config.update(kwargs)

        eval_class = EvalBase
        if eval_config.validation:
            EvalStatic.cross_validation(eval_config, eval=eval_class)
        else:
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

    # 1st rounf
    def execute(self, prompt_name="Y14_1213", script_name="Y14_1213_XX", limitation=0, config_file_name="template.yml",
                mode="standard", training=True, evaluation=True, clustering=True):

        # 訓練
        if training:
            print("Training...")
            train_config_path = "config/ys/train/{}".format(config_file_name)
            self.train(train_config_path=train_config_path, preprocess_name=prompt_name, mode=mode,
                       script_name=script_name, limitation=limitation, wandb_group=script_name)

        # 評価
        if evaluation:
            print("Evaluating...")
            eval_config_path = "config/ys/eval/{}".format(config_file_name)
            self.evaluate(eval_config_path=eval_config_path, preprocess_name=prompt_name, mode=mode,
                          script_name=script_name, limitation=limitation)

        # クラスタリング
        if clustering:
            print("Clustering...")
            eval_config_path = "config/ys/eval/{}".format(config_file_name)
            self.clustering(eval_config_path=eval_config_path, preprocess_name=prompt_name, mode=mode,
                            script_name=script_name, limitation=limitation)

    # 2nd round
    def supervising_preprocess(self, config_path, script_name, prev_mode, cluster_df_path, elimination_id, **kwargs):
        config = Util.load_preprocess_config(config_path)
        config.update(kwargs)
        elimination_list = [int(_id) for _id in elimination_id.split(" ")]
        PreprocessSupervising(config).execute(script_name, prev_mode, cluster_df_path, elimination_list)

    # wrapper
    def sv_preprocess_wrapper(self, prompt_name, script_name, limitation, elimination_id="",
                              cluster_df_path="", prev_mode="contamination"):
        """
        script_name: 2週目の実験名を設定する
        limitation: 実験データの数
        elimination_id: 排除対象のクラスタID
        cluster_df_path: 参照対象のクラスタデータのパス
        prev_mode: "contamination" or "standard"
        """

        # preprocess for supervising
        print("Preprocess...")
        config_path = "config/ys/preprocess/{}.yml".format(prompt_name)

        # execute
        self.supervising_preprocess(config_path=config_path, prev_mode=prev_mode, script_name=script_name,
                                    cluster_df_path=cluster_df_path, elimination_id=elimination_id,
                                    preprocess_name=prompt_name, limitation=limitation,)

    def sv_train(self, config_file_name="supervising.yml", model_path="", **kwargs):
        train_config_path = "config/ys/train/{}".format(config_file_name)
        train_config = Util.load_train_config(train_config_path)
        train_config.update(kwargs)
        train_config.mode = "sv"
        trainer = TrainSupervising
        trainer(train_config, model_path)()

    def fitness(self, eval_dir, script_name, **kwargs):
        Integration().quantitative_fitness(eval_dir,  script_name)

    def check_masking(self, eval_dir, script_name, masking_span, term):
        CheckMasking().check_masking_efficiency(eval_dir, script_name, masking_span, term)


if __name__ == "__main__":
    fire.Fire(Main)
