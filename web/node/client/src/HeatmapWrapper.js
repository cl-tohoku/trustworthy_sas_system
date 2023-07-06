import React, { useState, useEffect } from 'react';
import Heatmap from './Heatmap'

const HeatmapWrapper = (props) => {
  const renderClusterHeatmap = () => {
    const list = [];
    for (let idx = 0; idx < props.result.max; idx++) {
      list.push(
        <div className="border-2 border-gray-200">
          <Heatmap result={props.result} number={idx + 1} size={props.clusterSize} mask={props.mask}/>
        </div>
      );
    }
    return list;
  };

  const clusterUrl = "/distance";
  const GetClusterResults = (e) => {
    fetch(clusterUrl, requestOptions)
      .then(response => response.json())
      .then(data => props.setResult(data))
  };

  const requestOptions = {
    method: 'POST',
    headers: {
      'Accept': 'application/json, */*',
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    },
    body: JSON.stringify({
      'size': props.clusterSize,
      'setting': props.setting,
      'data_type': props.dataType,
    }),
  };
  
  const searchUrl = "/search";
  const GetSearchResults = (e) => {
    fetch(searchUrl, searchOptions)
      .then(response => response.json())
      .then(data => props.setResult(data))
  };

  const searchOptions = {
    method: 'POST',
    headers: {
      'Accept': 'application/json, */*',
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    },
    body: JSON.stringify({
      'keyword': props.keyword,
      'setting': props.setting,
      'data_type': props.dataType,
    }),
  };

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      GetClusterResults();
    }, 200);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [, props.clusterSize, props.mask, props.setting, props.dataType]);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      GetSearchResults();
    }, 200);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [props.keyword]);

  return (
    <div>
      <div className="grid grid-cols-1 gap-2">
        {props.result && renderClusterHeatmap()}
      </div>
    </div>
  );
};

export default HeatmapWrapper;