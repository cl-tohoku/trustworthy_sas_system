import MeCab
import unidic
import sys

sys.path.append("..")
from library.structure import Script


class JapaneseParser:
    def __init__(self):
        self.mecab_wakati = MeCab.Tagger('-Owakati')

    def mecab(self, text):
        result = self.mecab_wakati.parse(text)
        return result
