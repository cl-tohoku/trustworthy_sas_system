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

default_script = "Y14_2115_standard"

class State:
    def __init__(self):
        self.df = None
        self.cluster_label = None
        self.load_data()
        self.get_hdbscan()
        #self.do_dbscan()
        #self.plot_scatter()
        self.timestamp = ""

    def load_data(self, script_name=default_script, data_type="train", load=True):
        script_path = Path("data") / data_type / script_name / "cluster_df.gzip.pkl"
        if load:
            self.df = pd.read_pickle(script_path, compression="gzip")

    def do_dbscan(self, eps=0.01, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cosine_distances = np.array(self.df["Distance"].to_list())
        self.cluster_label = dbscan.fit_predict(cosine_distances).tolist()
        self.plot_scatter()

    def plot_scatter(self, script_name=default_script):
        # do tsne
        cosine_distances = np.array(self.df["Distance"].to_list())
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(cosine_distances)        

        # plot figure        
        labels = self.cluster_label
        plt.figure(figsize=(7, 4))
        for cluster in np.unique(labels):
            plt.scatter(tsne_results[labels == cluster, 0], tsne_results[labels == cluster, 1], label=f"Cluster {cluster}")
        plt.legend()
        plt.title("DBSCAN Clustering with T-SNE visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        os.makedirs("./data/tmp/", exist_ok=True)
        timestamp = str(time.time())
        self.timestamp = timestamp
        plt.savefig("./data/tmp/scatter_{}_{}.png".format(script_name, timestamp))
    
    def generate_response(self, timestamp):
        cluster_size = max(self.cluster_label)
        print(cluster_size)
        color = self.df["Color"].to_list()
        masked = self.df["Masked_Color"].to_list()
        token = self.df["Token"].to_list()

        # make response
        responses = {"cluster": self.cluster_label, "token": token, "color": color, "just": masked, "max": cluster_size}
        return responses

    def get_hdbscan(self):
        self.cluster_label = self.df["HDBSCAN"].to_list()

    def get_scatter_path(self, script_name, data_type):
        return "./data/{}/{}/scatter.png".format(data_type, script_name)
        

state = State()

class DbscanItem(BaseModel):
    eps: float
    min_samples: int
    script_name: str
    data_type: str


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
    responses = {"file": project_list}
    return responses

@app.post("/dbscan")
def dbscan(item: DbscanItem):
    print("Loading...")
    state.load_data(script_name=item.script_name, data_type=item.data_type)
    print("Clustering...")
    state.do_dbscan(eps=item.eps, min_samples=item.min_samples)
    print("Plotting...")
    timestamp = state.plot_scatter(script_name=item.script_name)
    return state.generate_response(timestamp)

@app.get("/scatter/{script_name}")
def scatter(script_name: str):
    # create response
    timestamp = state.timestamp
    file_path = "./data/tmp/scatter_{}_{}.png".format(script_name, timestamp)
    return FileResponse(file_path, media_type="image/png")

@app.get("/clustering/{script_name}/{data_type}/{term}/{score}")
def hdbscan(script_name: str, data_type: str, term: int, score: int):
    state.load_data(script_name=script_name, data_type=data_type, load=False)
    state.get_hdbscan()
    return state.generate_response(timestamp="")

@app.get("/tnse/{script_name}/{data_type}/{term}/{score}")
def hdbscan_scatter(script_name: str, data_type: str, term: int, score: int):
    file_path = state.get_scatter_path(script_name, data_type)
    return FileResponse(file_path, media_type="image/png")