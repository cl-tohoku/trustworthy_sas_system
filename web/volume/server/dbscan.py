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


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,   # 追記により追加
    allow_methods=["*"],      # 追記により追加
    allow_headers=["*"]       # 追記により追加
)

default_script = "Y15_2315_standard"
state = State()

class State:
    def __init__(self):
        self.df = None
        self.cluster_label = None
        self.load_data()
        self.dbscan()

    def load_data(self, script_name=default_script, data_type="train"):
        script_path = Path("data") / data_type / script_name / "cluster_data.gzip.pkl"
        self.df = pd.load_data(script_path, compression="gzip")

    def do_dbscan(self, eps=0.1, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        self.cluster_label = dbscan.fit_predict(cosine_distances).tolist()
        self.plot_scatter()
    
    def generate_response():
        cluster_size = max(self.cluster_label)
        color = self.df["Color"].to_list()
        masked = self.df["Masked_Color"].to_list()
        token = self.df["Token"].to_list()

        # make response
        responses = {"cluster": self.cluster_label, "token": token, "color": color, "just": masked, "max": cluster_size}
        return response

    def plot_scatter():
        plt.figure(figsize=(10, 6))
        for cluster in np.unique(labels):
            plt.scatter(tsne_results[labels == cluster, 0], tsne_results[labels == cluster, 1], label=f"Cluster {cluster}")
        plt.legend()
        plt.title("DBSCAN Clustering with T-SNE visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        os.makedirs("./data/tmp/", exist_ok=True)
        plt.savefig("./data/tmp/scatter.png")


class DbscanItem(BaseModel):
    eps: float
    min_samples: int
    script_name: str
    data_type: str


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/dbscan")
async def dbscan(item: DbscanItem):
    state.load_data(script_name=item.script_name, data_type=item.data_type)
    state.do_dbscan(eps=item.eps, min_samples=item.min_samples)
    return state.generate_response()
    
@app.get("/file")
async def file():
    file_list = glob("data/train/*")
    project_list = []
    for file_path in file_list:
        file_path = os.path.splitext(os.path.basename(file_path))[0]
        project_list.append(str(file_path))
    project_list = sorted(project_list)
    responses = {"file": project_list}
    return responses

