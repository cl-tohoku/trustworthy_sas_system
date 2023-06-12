import fire
from itertools import product

# my libraries
from library.util import Util
from preprocess.base import PreprocessBase
from preprocess.finetuning import PreprocessFinetuning
from train.base import TrainBase
from train.base import TrainStatic
from train.finetuning import TrainFinetuning
from eval.base import EvalBase
from eval.lazy import Integration
from eval.base import EvalStatic
from eval.clustering import Clustering2
from eval.finetuning import EvalFinetuning

class Main:
    def __init__(self, ):
        pass

    def preprocess(self, config_path, **kwargs):
        config = Util.load_preprocess_config(config_path)
        config.update(kwargs)
        PreprocessBase(config)()

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
        Clustering2(eval_config).make_clustering_results()

    def integrate_performance(self, prompt_name, eval_dir_path):
        Integration()(prompt_name, eval_dir_path)

    def execute_combinations(self, prompt_list, limitation_list, config_file_list,
                             preprocessing=True, training=True, evaluation=True, clustering=True ):
        # parse argments
        prompt_list = [str(x) for x in prompt_list.split(" ")]
        limitation_list = [int(x) for x in limitation_list.split(" ")] if type(limitation_list) == str else [limitation_list]
        config_file_list = [str(x) for x in config_file_list.split(" ")]
        # for loop
        for prompt_name, limitation, config_file_name in product(prompt_list, limitation_list, config_file_list):
            print(prompt_name, limitation, config_file_name)
            try:
                self.execute(prompt_name, limitation, config_file_name, preprocessing, training, evaluation, clustering)
            except Exception as e:
                print("Error:", e)


    def execute(self, prompt_name="Y14_1213", limitation=0, config_file_name="template.yml",
                preprocessing=True, training=True, evaluation=True, clustering=True,):
        script_name = prompt_name if limitation == 0 else "{}_{}".format(prompt_name, limitation)
        print(script_name)

        if preprocessing:
            print("Preprocess...")
            preprocess_config_path = "config/ys/preprocess/{}.yml".format(prompt_name)
            self.preprocess(config_path=preprocess_config_path, script_name=script_name, train_size_limitation=limitation, download_ft=False)

        if training:
            print("Training...")
            train_config_path = "config/ys/train/{}".format(config_file_name)
            self.train(train_config_path=train_config_path, script_name=script_name, wandb_group=script_name)

        if evaluation:
            # evaluate
            print("Evaluating...")
            eval_config_path = "config/ys/eval/{}".format(config_file_name)
            self.evaluate(eval_config_path=eval_config_path, script_name=script_name)

        if clustering:
            print("Clustering...")
            eval_config_path = "config/ys/eval/{}".format(config_file_name)
            self.clustering(eval_config_path=eval_config_path, script_name=script_name)

    def finetuning_preprocess(self, prep_config_path, eval_config_path, **kwargs):
        config = Util.load_preprocess_config(prep_config_path)
        config.update(kwargs)
        eval_config = Util.load_eval_config(eval_config_path)
        eval_config.update(kwargs)
        PreprocessFinetuning(config, eval_config)()

    def finetuning_train(self, train_config_path, **kwargs):
        config = Util.load_train_config(train_config_path)
        config.update(kwargs)
        TrainFinetuning(config).execute()

    def finetuning_eval(self, eval_config_path, given_term, **kwargs):
        config = Util.load_eval_config(eval_config_path)
        config.update(kwargs)
        EvalFinetuning(config).execute(given_term=given_term)

    def fitness(self, eval_dir, script_name, **kwargs):
        Integration().fitness(eval_dir, script_name)

if __name__ == "__main__":
    fire.Fire(Main)
