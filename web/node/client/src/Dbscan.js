import React, { useState, useEffect } from 'react';
import './App.css';
import Heatmap from "./Heatmap";
import Image from "./Image";
import SearchBox from './Search';
import ClusterRange from './ClusterRange';
import FileList from './FileList';
import HeatmapWrapper from './HeatmapWrapper';
import Scatter from './Scatter';
import Cluster from './Cluster';
import InputForm from './InputForm';

export function Dbscan(props){
  const [result, setResult] = useState();
  const [clusterSize, setClusterSize] = useState(10);
  const [mask, setMask] = useState(false);
  const [mode, setMode] = useState(true);
  const [setting, setSetting] = useState('Y14_2115_standard');
  const [dataType, setDataType] = useState('train');
  const [keyword, setKeyword] = useState("");
  const [eps, setEps] = useState(0.1);
  const [minSamples, setMinSamples] = useState(5);
  const [imagePath, setImagePath] = useState("");
  const [loading, setLoading] = useState(false)
  const [term, setTerm] = useState("A")
  const [score, setScore] = useState(2)
  const [inertiaPath, setInertiaPath] = useState("");

  const SetClusterSize = (e) => {
    setClusterSize(e.target.value)
  };

  const SetMask = (e) => {
    setMask(!mask)
  };
  
  const SetType = (e) => {
    if (dataType === "train") {
      setDataType("test")
    } else {
      setDataType("train")
    }
  };

  const RenderScatter = () => {
    return <Scatter imagePath={imagePath} />;
  }
  
  const RenderInertia = () => {
    return <Scatter imagePath={inertiaPath} />;
  }

  const GetDbscanResults = (e) => {
    const serverUrl = `/clustering/${setting}/${dataType}/${term}/${score}/${clusterSize}`;
    fetch(serverUrl)
      .then(response => response.json())
      .then(data => {setResult(data);});
  };

  const GetImagePath = () => {
    const imagePoint = `/scatter/${setting}/${dataType}/${term}/${score}/${clusterSize}`
    fetch(imagePoint)
      .then(response => response.blob())
      .then(parsed => {
        setImagePath(URL.createObjectURL(parsed));
      });
  };
  
  const GetInertiaPath = () => {
    const inertiaPoint = `/inertia/${setting}/${dataType}/${term}/${score}`
    fetch(inertiaPoint)
      .then(response => response.blob())
      .then(parsed => {
        setInertiaPath(URL.createObjectURL(parsed));
      });
  };

  useEffect(() => {
    return () => {
      GetDbscanResults();
    };
  }, [setting, dataType, clusterSize, mask]);

  useEffect(() => {
    return () => {
      GetImagePath();
      GetInertiaPath();
    };
  }, [result]);

  return (
    <div className="flex w-screen h-screen text-gray-700">
      <div className="flex flex-col w-full">
        <div className="flex flex-row items-center flex-shrink-0 h-16 px-8 border-b border-gray-300">
          <div className="flex-none w-1/2">
            <h1 className="font-medium">{setting}</h1>
          </div>
          <div className="flex-none">
            {loading ? <div>Loading...</div> : <div>Loaded</div>}
          </div>
          <div className="flex flex-row flex-none justify-end">
            <div className="flex-none h-10 w-64 px-4 ml-2 items-center justify-center">
              <SearchBox setting={setting} dataType={dataType} setKeyword={setKeyword} setMode={setMode} />
            </div>
            <div className="flex-none items-center justify-center h-10 w-48 px-4 ml-2 text-sm font-medium bg-white">
              <ClusterRange SetClusterSize={SetClusterSize} clusterSize={clusterSize}/>
            </div>
            <button className="flex-none h-10 w-32 px-4 ml-2 text-sm font-medium bg-gray-200 rounded hover:bg-gray-300"
              onClick={SetType}>
              {dataType}
            </button>
            <button className="flex-none h-10 w-48 px-4 ml-2 text-sm font-medium bg-gray-200 rounded hover:bg-gray-300"
              onClick={SetMask}>
              Justification Cue
            </button>
          </div>
        </div>
        <div className="flex flex-row h-full">
          <div className="flex flex-col w-64 border-r border-gray-300">
            <div className="flex flex-col flex-grow p-4 overflow-auto">
              <FileList setSetting={setSetting}/>
            </div>
          </div>
          <div className="flex flex-col w-screen overflow-auto bg-gray-200">
            <div className="flex flex-row h-96 gap-4">
              <div className="flex-none m-4 p-2 w-1/3 rounded-lg border border-gray-200 bg-white">
                {result && RenderScatter()}
              </div>
              <div className="flex-none m-4 p-2 w-1/3 rounded-lg border border-gray-200 bg-white">
                {result && RenderInertia()}
              </div>
            </div>
            <div className="flex flex-col flex-1 bg-gray-0">
              <div className="flex-1 p-4 pt-6">
                <Cluster result={result} mask={mask} clusterSize={10}/>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
