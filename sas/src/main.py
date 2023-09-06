import fire
from itertools import product

# my libraries
from library.util import Util
from preprocess.base import PreprocessBase
from preprocess.finetuning import PreprocessFinetuning
from preprocess.masking import PreprocessMasking
from preprocess.supervising import PreprocessSupervising
from preprocess.contamination import PreprocessContamination
from train.base import TrainBase
from train.base import TrainStatic
from train.finetuning import TrainFinetuning
from train.masking import TrainMasking
from eval.base import EvalBase
from eval.lazy import Integration, CheckMasking
from eval.base import EvalStatic
from eval.clustering import Clustering2, ForClustering
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
        if config.supervising_path:
            PreprocessSupervising(config)()

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
        #Clustering2(eval_config).make_clustering_results()
        ForClustering(eval_config).make_clustering_datasets()

    def integrate_performance(self, prompt_name, eval_dir_path):
        Integration()(prompt_name, eval_dir_path)

    def make_dataset(self, prompt_name="Y14_1213", limitation=0, contamination=False, masking=False, supervising=False):
        print("Preprocess...")
        preprocess_config_path = "config/ys/preprocess/{}.yml".format(prompt_name)
        cont_path = "config/contamination/{}.yml".format(prompt_name) if contamination else ""
        mask_path = "config/masking/{}.yml".format(prompt_name) if masking else ""
        supervising_path = "config/supervising/{}.yml".format(prompt_name) if supervising else ""
        # config に載せて実行
        self.preprocess(config_path=preprocess_config_path, preprocess_name=prompt_name, limitation=limitation,
                        contamination_path=cont_path, masking_path=mask_path, supervising_path=supervising_path)

    def execute(self, prompt_name="Y14_1213", script_name="Y14_1213_XX", limitation=0, config_file_name="template.yml",
                mode="standard", pretrained_script_name="", training=True, evaluation=True, clustering=True,):
        """
        script_name は自由に指定できる引数、実験名になる、
        """

        # script_name は実験内容を表すユニークな名称
        # 訓練
        if training:
            print("Training...")
            train_config_path = "config/ys/train/{}".format(config_file_name)
            self.train(train_config_path=train_config_path, preprocess_name=prompt_name, mode=mode,
                       script_name=script_name, limitation=limitation, wandb_group=script_name,
                       pretrained_script_name=pretrained_script_name)

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

    def fitness(self, eval_dir, script_name, **kwargs):
        Integration().quantitative_fitness(eval_dir,  script_name)

    def check_masking(self, eval_dir, script_name, masking_span, term):
        CheckMasking().check_masking_efficiency(eval_dir, script_name, masking_span, term)


if __name__ == "__main__":
    fire.Fire(Main)
