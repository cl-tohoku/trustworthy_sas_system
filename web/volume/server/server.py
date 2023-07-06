from fastapi import FastAPI
from pprint import pprint
from typing import Union
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import pandas as pd
from result import ClusteringBeta
from glob import glob
import os
from fastapi.responses import FileResponse


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,   # 追記により追加
    allow_methods=["*"],      # 追記により追加
    allow_headers=["*"]       # 追記により追加
)
beta = ClusteringBeta()


class Item(BaseModel):
    size: Union[str, int]
    setting: str
    data_type: str


class UpdateItem(BaseModel):
    size: Union[str, int]
    setting: str
    problematic_list: list
    data_type: str


class SearchItem(BaseModel):
    setting: str
    data_type: str
    keyword: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/distance")
def distance(item: Item):
    # setting
    size = 10 if not item.size else int(item.size)
    beta.load_sentence_data(setting=item.setting, data_type=item.data_type)
    beta.load_cluster_data(setting=item.setting, size=size, data_type=item.data_type)

    # process query
    result_df = beta.do_query(distance=size)
    color = result_df["Color"].to_list()
    just = result_df["Mask_Color"].to_list()
    cluster_size = max(result_df["Number"].to_list())

    # make response
    responses = {"cluster": result_df["Number"].to_list(), "token": result_df["Token"].to_list(),
                 "color": color, "max": cluster_size, "just": just, "cluster_mode": True}
    return responses


@app.get("/file")
def file():
    file_list = glob("data/train/*")
    project_list = []
    for file_path in file_list:
        file_path = os.path.splitext(os.path.basename(file_path))[0]
        # file_path = file_path.replace("_counter_hierarchy", "")
        project_list.append(str(file_path))
    project_list = sorted(project_list)
    responses = {"file": project_list}
    return responses


@app.get("/dendrogram/{data_type}/{setting}/{size}")
def get_dendrogram(data_type: str, setting: str, size: Union[int, str] = 3):
    # Open the image file in binary mode
    # Return the image as a FileResponse
    # file_path = "data/figure/{}/{}_counter_dendrogram.png".format(size, setting)
    file_path = "data/{}/{}/{}/dendrogram.png".format(data_type, setting, size)
    return FileResponse(file_path, media_type="image/png")

@app.post("/update")
def update_dataset(item: UpdateItem):
    # setting
    beta.load_sentence_data(setting=item.setting, data_type=item.data_type)
    beta.load_cluster_data(setting=item.setting, size=item.size, data_type=item.data_type)
    result_df = beta.do_query(distance=item.size)

    # update & write
    update_df = beta.update_df(result_df, item.problematic_list)
    beta.write_df(update_df, data_type=item.data_type, setting=item.setting)

@app.post("/search")
def search(item: SearchItem):
    beta.load_sentence_data(setting=item.setting, data_type=item.data_type)
    df = beta.search_data(item.keyword)

    color = result_df["Color"].to_list()
    just = result_df["Mask_Color"].to_list()

    # make response
    responses = {"token": result_df["Token"].to_list(), "color": color, "just": just, "cluster_mode": False}
    return responses
    
