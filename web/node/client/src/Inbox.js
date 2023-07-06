import React, { useState, useEffect } from 'react';
import './App.css';
import Heatmap from "./Heatmap";
import Image from "./Image";
import SearchBox from './Search';
import ClusterRange from './ClusterRange';
import FileList from './FileList';


export function Inbox(props){
  const [result, setResult] = useState();
  const [clusterSize, setClusterSize] = useState(10);
  const [mask, setMask] = useState(false);
  const [mode, setMode] = useState(true);
  const [setting, setSetting] = useState('Y14_1213_100_A_R');
  const [dataType, setDataType] = useState('train');
  const [keyword, setKeyword] = useState("");

  const requestOptions = {
    method: 'POST',
    headers: {
      'Accept': 'application/json, */*',
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    },
    body: JSON.stringify({
      'size': clusterSize,
      'setting': setting,
      'mask': mask,
      'data_type': dataType,
    }),
  };

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      console.log(`Value changed to ${clusterSize}`);
      GetClusteringResults();
    }, 200);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [clusterSize]);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      console.log(`Value changed to ${mask}`);
      GetClusteringResults();
    }, 200);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [mask]);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      console.log(`Value changed to ${setting}`);
      GetClusteringResults();
    }, 200);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [setting]);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      console.log(`Value changed to ${dataType}`);
      GetClusteringResults();
    }, 200);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [dataType]);

  const url = "/distance";
  const GetClusteringResults = (e) => {
    fetch(url, requestOptions)
      .then(response => response.json())
      .then(data => setResult(data))
  };

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

  const renderHeatmapList = () => {
    const list = [];
    for (let idx = 0; idx < result.max; idx++) {
      list.push(
        <div className="border-2 border-gray-200">
          <Heatmap result={result} number={idx + 1} size={clusterSize} mask={mask}/>
        </div>
      );
    }
    return list;
  };

  const renderImage = () => {
    return <Image endpoint="dendrogram" dataType={dataType} setting={setting} size={clusterSize} />;
  }


  return (
    <div className="flex w-screen h-screen text-gray-700">
      {/* Component Start */}
      <div className="flex flex-col items-center w-16 pb-4 overflow-auto border-r border-gray-300">
        <a
          className="flex items-center justify-center flex-shrink-0 w-full h-16 bg-gray-300"
          href="#"
        >
          <svg
            className="w-8 h-8"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
            />
          </svg>
        </a>
      </div>
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
          <div className="flex flex-row w-screen overflow-auto">
            <div className="flex flex-row h-auto bg-gray-0">
              <div className="flex-1 w-52 justify-center items-center">
                {result && renderImage()}
              </div>
            </div>
            <div className="flex flex-col flex-1 bg-gray-0">
              <div className="flex-1 p-4 bg-gray-200 pt-6">
                <div className="grid grid-cols-1 gap-2">
                  {result && renderHeatmapList()}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* Component End  */}
    </div>
  );
}