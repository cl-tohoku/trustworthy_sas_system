import React, { useState, useEffect } from 'react';
import HeatmapRender from './HeatmapRender';

const Cluster = (props) => {
  const createArray = (i) => Array.from({ length: i }, (_, index) => index);

  const renderCluster = () => {
    const numberArray = createArray(props.result.max);
    const clusterArray = numberArray.map(idx =>
      <div className="border-2 border-gray-200">
        <HeatmapRender result={props.result} clusterIdx={idx} mask={props.mask} expansion={props.expansion}/>
      </div>
    );
    return clusterArray
  };

  return (
    <div>
      <div className="grid grid-cols-1 gap-2">
        {props.result && renderCluster()}
      </div>
    </div>
  );
};

export default Cluster;