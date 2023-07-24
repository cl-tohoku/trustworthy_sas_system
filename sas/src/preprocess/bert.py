import sys

sys.path.append("..")
from library.util import Util
from library.structure import Script, ScriptBert
from transformers import BertTokenizer, BertJapaneseTokenizer
from dataclasses import asdict


class PreprocessBert:
    def __init__(self, config):
        self.config = config
        # self.tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        self.tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char-v2")

        # special token
        self.special_token_converter = dict()
        for k, v in zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids):
            self.special_token_converter[k] = v

    def remove_u3000(self, text, annotation):
        removed_annotation = []
        char_list = "".join(text.split(" "))
        for idx, char in enumerate(char_list):
            if char == "\u3000":
                continue
            else:
                removed_annotation.append(annotation[idx])
        return removed_annotation

    def to_bert_script(self, script):
        # text = "".join(script.text.split())
        text = script.text
        tokenized = self.tokenizer.tokenize(text)
        encoded = self.tokenizer.encode_plus(text, padding="do_not_pad", truncation=True,
                                             max_length=self.config.max_length, return_tensors="pt")
        bert_data = dict()
        bert_data.update(asdict(script))
        bert_data["tokenized"] = tokenized
        for key in encoded.keys():
            bert_data[key] = encoded[key].squeeze(0).tolist()


        # annotation
        annotation_matrix = []
        for annotation in script.annotation_char:
            # remove \u3000
            if len(annotation) + 2 != len(bert_data["input_ids"]):
                annotation = self.remove_u3000(text, annotation)

            annotation = [0] + annotation + [0]
            if len(annotation) != len(bert_data["input_ids"]):
                # 3点リーダの場合ずれる、とりあえず雑に埋めておく
                padding_length = len(bert_data["input_ids"]) - len(annotation)
                annotation.extend([0] * padding_length)
            # padded = Util.padding(annotation, self.config.max_length, 0)
            padded = annotation
            annotation_matrix.append(padded)
        bert_data["annotation_matrix"] = annotation_matrix

        script_bert = ScriptBert(**bert_data)
        script_bert.unknown_id = self.special_token_converter["[UNK]"]
        return script_bert

    def __call__(self, scripts):
        bert_scripts = []
        for script in scripts:
            bert_scripts.append(self.to_bert_script(script))
        return bert_scripts
