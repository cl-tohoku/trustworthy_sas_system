from fire import Fire
import subprocess
from main import Main
import os


def run(x):
    print(x.split())
    subprocess.run(x.split(),)


class Console:
    def __init__(self):
        pass

    def first_process(self, prompt_name, term, idx):
        worker = Main()
        worker.make_dataset(prompt_name=prompt_name)
        worker.sf_preprocess(config_path="./config/ys/preprocess/{}.yml".format(prompt_name))
        worker.execute(prompt_name=prompt_name, script_name="{}_superficial_{}-{}".format(prompt_name, term, idx),
                       sf_term=term, sf_idx=idx, config_file_name="template.yml", mode="superficial",
                       training=True, evaluation=True)

    def second_process(self, prompt_name, term, idx, score):
        worker = Main()
        code = worker.sv_preprocess(preprocess_name=prompt_name, prompt_path="config/prompt/{}.yml".format(prompt_name),
                                    sf_term=term, sf_idx=idx, target_score=score, prev_mode="superficial",
                                    script_name="{}_supervising_{}-{}".format(prompt_name, term, idx),
                                    prev_script_name="{}_superficial_{}-{}".format(prompt_name, term, idx))
        if code == 0:
            worker.sv_train()

    def first_round(self, prompt_name, term_list, cuda="0,1,2,3"):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
        for term in term_list.split():
            for idx in range(5):
                print(prompt_name, term, idx)
                self.first_process(prompt_name, term, idx)

    def second_round(self, prompt_name, term_list, score_list,  cuda="0,1,2,3"):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
        for term_idx, term in enumerate(term_list.split()):
            for idx in range(5):
                print(prompt_name, term, idx, score_list[term_idx])
                self.second_process(prompt_name, term, idx, score_list[term_idx])


if __name__ == "__main__":
    Fire(Console)
