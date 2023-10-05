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
import Dendrogram from './Dendrogram';

export function Dbscan(props){
  const [result, setResult] = useState();
  const [clusterSize, setClusterSize] = useState(10);
  const [mask, setMask] = useState(false);
  const [setting, setSetting] = useState('Y14_2115_standard');
  const [dataType, setDataType] = useState('train');
  const [imagePath, setImagePath] = useState("");
  const [term, setTerm] = useState("A")
  const [score, setScore] = useState(2)
  const [inertiaPath, setInertiaPath] = useState("");
  const [dendrogramPath, setDendrogramPath] = useState("");
  const [rubric, setRubric] = useState();

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
    return (
      <div className="flex-none m-4 p-2 w-1/3 rounded-lg border border-gray-200 bg-white">
        <Scatter imagePath={imagePath} />
      </div>
    );
  }
  
  const RenderInertia = () => {
    return (
      <div className="flex-none m-4 p-2 w-1/3 rounded-lg border border-gray-200 bg-white">
        <Scatter imagePath={inertiaPath} />
      </div>
    );
  }

  const RenderDendrogram = () => {
    return (
      <div className="flex-none w-auto bg-white">
        <Dendrogram imagePath={dendrogramPath} />
      </div>
    );
  }
  
  const RenderRubric = () => {
    console.log(rubric)
    return (
      <div className="flex-auto w-auto m-4 p-2 rounded-lg border border-gray-200 bg-white">
        {
          rubric.term.map((value, index) => {
            return (
              <p>
                {value}:{rubric.description[index]}
              </p>
            );
          })
        }
      </div>
    );
  }

  const GetClusteringResults = (e) => {
    const serverUrl = `/clustering/${setting}/${dataType}/${term}/${score}/${clusterSize}`;
    fetch(serverUrl)
      .then(response => response.json())
      .then(data => {setResult(data);});
    GetImagePath();
    GetInertiaPath();
    GetDendrogramPath();
    GetRubric();
    console.log(rubric)
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
  
  const GetDendrogramPath = () => {
    const dendrogramPoint = `/dendrogram/${setting}/${dataType}/${term}/${score}/${clusterSize}`
    fetch(dendrogramPoint)
      .then(response => response.blob())
      .then(parsed => {
        setDendrogramPath(URL.createObjectURL(parsed));
      });
  };

  const GetRubric = () => {
    const rubricPoint = `/rubric/${setting}`
    fetch(rubricPoint)
      .then(response => response.json())
      .then(data => {setRubric(data);});
  };
  

  const SizeBox = () => {
    const handleSizeChange = (e) => {
      setClusterSize(e.target.value);
    };
    return (
      <div className="flex items-center justify-center">
        <input
          type="number"
          step="1"
          placeholder="Size"
          className="border-2 border-gray-300 p-4 rounded-md w-24 h-10"
          value={clusterSize}
          onChange={handleSizeChange}
        />
      </div>
    );
  };

  const ScoreBox = () => {
    const handleScoreChange = (e) => {
      setScore(e.target.value);
    };
    return (
      <div className="flex items-center justify-center">
        <input
          type="number"
          step="1"
          placeholder="Score"
          className="border-2 border-gray-300 p-4 rounded-md w-24 h-10"
          value={score}
          onChange={handleScoreChange}
        />
      </div>
    );
  };

  const TermBox = () => {
    const handleTermChange = (e) => {
      setTerm(e.target.value);
    };
    return (
      <div className="flex items-center justify-center">
        <input
          type="text"
          maxlength="1"
          pattern="[A-Z]"
          placeholder="Term"
          className="border-2 border-gray-300 p-4 rounded-md w-24 h-10"
          value={term}
          onChange={handleTermChange}
        />
      </div>
    );
  };

  return (
    <div className="flex w-screen h-screen text-gray-700">
      <div className="flex flex-col w-full">
        <div className="flex flex-row items-center flex-shrink-0 h-16 px-8 border-b border-gray-300">
          <div className="flex-none w-1/3">
            <h1 className="font-medium">{setting}</h1>
          </div>
          <div className="flex flex-row flex-auto w-auto justify-end">
            <div className="flex-none items-center justify-center h-10 w-28 px-4 ml-2 text-sm font-medium bg-white">
              {TermBox()}
            </div>
            <div className="flex-none items-center justify-center h-10 w-28 px-4 ml-2 text-sm font-medium bg-white">
              {ScoreBox()}
            </div>
            <div className="flex-none items-center justify-center h-10 w-28 px-4 ml-2 text-sm font-medium bg-white">
              {SizeBox()}
            </div>
            <button className="flex-none h-10 w-32 px-4 ml-2 text-sm font-medium bg-gray-200 rounded hover:bg-gray-300"
              onClick={SetType}>
              {dataType}
            </button>
            <button className="flex-none h-10 w-48 px-4 ml-2 text-sm font-medium bg-gray-200 rounded hover:bg-gray-300"
              onClick={SetMask}>
              Justification Cue
            </button>
            <button className="flex-none h-10 w-48 px-4 ml-2 text-sm rounded bg-blue-500 hover:bg-blue-700 text-white font-bold"
              onClick={GetClusteringResults}>
              clustering
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
              {result && RenderInertia()}
              {rubric && RenderRubric()}
            </div>
            <div className="flex flex-row flex-1 bg-gray-0 m-4">
              <div className="flex-none">
                {result && RenderDendrogram()}
              </div>
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
