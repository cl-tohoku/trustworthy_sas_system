from fastapi import FastAPI
from pprint import pprint
from typing import Union
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import pandas as pd
from glob import glob
import os
from fastapi.responses import FileResponse
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import numpy as np
import time


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,   # 追記により追加
    allow_methods=["*"],      # 追記により追加
    allow_headers=["*"]       # 追記により追加
)

default_script = "Y14_1213_superficial_A-0"

class State:
    def __init__(self):
        self.base_df = None
        self.df = None
        self.load_data()

    def load_data(self, script_name=default_script, data_type="train"):
        script_path = Path("data") / data_type / script_name / "data_df.gzip.pkl"
        self.base_df = pd.read_pickle(script_path, compression="gzip")

    def join_cluster_data(self, script_name, data_type, term, score, cluster_k):
        # load cluster results
        file_name = "{}_{}_{}.gzip.pkl".format(term, score, cluster_k)
        script_path = Path("data") / data_type / script_name / "cluster" / file_name
        cluster_df = pd.read_pickle(script_path, compression="gzip")
        # join
        self.df = self.base_df[self.base_df["Term"] == term].merge(cluster_df, on='Sample_ID', how='inner')
    
    def generate_response(self):
        cluster_label = self.df["Cluster"].to_list()
        cluster_size = max(cluster_label) + 1
        color = self.df["Color"].to_list()
        masked = self.df["Masked_Color"].to_list()
        token = self.df["Token"].to_list()

        # make response
        responses = {"cluster": cluster_label, "token": token, "color": color, "just": masked, "max": cluster_size}
        return responses

    def get_scatter_path(self, script_name, data_type, term, score, cluster_k):
        return "./data/{}/{}/scatter/{}_{}_{}.png".format(data_type, script_name, term, score, cluster_k)
    
    def get_dendrogram_path(self, script_name, data_type, term, score, cluster_k):
        return "./data/{}/{}/dendrogram/{}_{}_{}.png".format(data_type, script_name, term, score, cluster_k)
    
    def get_inertia_path(self, script_name, data_type, term, score):
        return "./data/{}/{}/inertia/{}_{}.png".format(data_type, script_name, term, score)
        

state = State()

@app.get("/")
async def root():
    return {"message": "Hello World"}
    
@app.get("/file")
def file():
    file_list = glob("data/train/*")
    project_list = []
    for file_path in file_list:
        file_path = os.path.splitext(os.path.basename(file_path))[0]
        project_list.append(str(file_path))
    project_list = sorted(project_list)
    response = {"file": project_list}
    return response

@app.get("/clustering/{script_name}/{data_type}/{term}/{score}/{cluster_k}")
def clustering(script_name: str, data_type: str, term: str, score: int, cluster_k: int):
    state.load_data(script_name=script_name, data_type=data_type)
    state.join_cluster_data(script_name, data_type, term, score, cluster_k)
    return state.generate_response()

@app.get("/scatter/{script_name}/{data_type}/{term}/{score}/{cluster_k}")
def scatter(script_name: str, data_type: str, term: str, score: int, cluster_k: int):
    file_path = state.get_scatter_path(script_name, data_type, term, score, cluster_k)
    return FileResponse(file_path, media_type="image/png")

@app.get("/inertia/{script_name}/{data_type}/{term}/{score}")
def inertia(script_name: str, data_type: str, term: str, score: int):
    file_path = state.get_inertia_path(script_name, data_type, term, score)
    return FileResponse(file_path, media_type="image/png")

@app.get("/dendrogram/{script_name}/{data_type}/{term}/{score}/{cluster_k}")
def dendrogram(script_name: str, data_type: str, term: str, score: int, cluster_k: int):
    file_path = state.get_dendrogram_path(script_name, data_type, term, score, cluster_k)
    print(file_path)
    return FileResponse(file_path, media_type="image/png")

@app.get("/rubric/{script_name}")
def rubric(script_name: str):
    response = {
        "term": ["A", "B", "C"],
        "description": [
            "緑の庭は",
            "自然と人間の論理のせめぎあいから生まれる本来の共生ではなく",
            "自然の論理が排除され、人間の論理だけで作られたものだから",
        ],
    }
    return response