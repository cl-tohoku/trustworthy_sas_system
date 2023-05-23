from pathlib import Path
import sys
import string
import json

sys.path.append("..")
from library.structure import Script, PromptConfig
from library.util import Util

class YS:
    def __init__(self):
        pass

    def load_dataset(self, script_path):
        with open(script_path, "r") as f:
            return json.load(f)

    def dict_to_script(self, dict_data):
        script = Script()
        script.text = dict_data["mecab"]
        script.score = dict_data["score"]
        alphabet_list = string.ascii_uppercase
        for alphabet in alphabet_list:
            if "{}_Score".format(alphabet) in dict_data:
                script.score_vector.append(dict_data["{}_Score".format(alphabet)])
            if alphabet in dict_data:
                annotation_list = [int(x) for x in dict_data[alphabet].split(" ")]
                script.annotation_word.append(annotation_list)
            if "C_{}".format(alphabet) in dict_data:
                annotation_list = [int(x) for x in dict_data["C_{}".format(alphabet)].split(" ")]
                script.annotation_char.append(annotation_list)
        return script

    def load_prompt(self, prompt_path):
        if prompt_path is None:
            raise RuntimeError("Prompt path is not set")
        else:
            return Util.load_prompt_config(prompt_path)

    def __call__(self, config):
        script_path = config.script_path
        json_dataset = self.load_dataset(script_path)
        # list(dict) -> list(script)
        scripts = [self.dict_to_script(d) for d in json_dataset]
        prompt = self.load_prompt(config.prompt_path)
        return scripts, prompt
