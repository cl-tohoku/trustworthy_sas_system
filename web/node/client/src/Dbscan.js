import React, { useState, useEffect } from 'react';
import './App.css';
import Heatmap from "./Heatmap";
import Image from "./Image";
import SearchBox from './Search';
import ClusterRange from './ClusterRange';
import FileList from './FileList';
import HeatmapWrapper from './HeatmapWrapper';

export function Dbscan(props){
  const [result, setResult] = useState();
  const [clusterSize, setClusterSize] = useState(10);
  const [mask, setMask] = useState(false);
  const [mode, setMode] = useState(true);
  const [setting, setSetting] = useState('Y15_1213');
  const [dataType, setDataType] = useState('train');
  const [keyword, setKeyword] = useState("");


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


  const renderScatter = () => {
    return <Image />;
  }


  return (
    <div className="flex w-screen h-screen text-gray-700">
      {/* Component Start */}
      <div className="flex flex-col w-full">
        <div className="flex flex-row items-center flex-shrink-0 h-16 px-8 border-b border-gray-300">
          <div className="flex-none w-1/2">
            <h1 className="text-lg font-medium">{setting}</h1>
          </div>
          <div className="flex flex-row flex-none w-1/2 justify-end">
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
          <div className="flex flex-col w-screen overflow-auto">
            <div className="flex flex-row h-52 bg-gray-0">
              <div className="flex-1 w-auto">
                {result && renderScatter()}
              </div>
            </div>
            <div className="flex flex-col flex-1 bg-gray-0">
              <div className="flex-1 p-4 bg-gray-200 pt-6">
                <HeatmapWrapper result={result} setResult={setResult} setting={setting} mask={mask}
                  dataType={dataType} clusterSize={clusterSize} keyword={keyword} setKeyword={setKeyword}/>
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* Component End  */}
    </div>
  );
}
