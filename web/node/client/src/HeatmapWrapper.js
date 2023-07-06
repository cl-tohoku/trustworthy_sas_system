import React, { useState, useEffect } from 'react';
import Heatmap from './Heatmap'

const HeatmapWrapper = (props) => {
  const renderHeatmapList = () => {
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

  const url = "/distance";
  const GetResults = (e) => {
    fetch(url, requestOptions)
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
      'mask': props.mask,
      'data_type': props.dataType,
    }),
  };

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      GetResults();
    }, 200);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [props.clusterSize, props.mask, props.setting, props.dataType, props.keyword]);

  return (
    <div>
      <div className="grid grid-cols-1 gap-2">
        {props.result && renderHeatmapList()}
      </div>
    </div>
  );
};

export default HeatmapWrapper;