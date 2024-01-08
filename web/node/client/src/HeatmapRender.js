import React, { useState, useEffect } from 'react';
import HeatmapLine from "./HeatmapLine";

export function HeatmapRender(props) {
  const [slice, setSlice] = useState(true);
  const toggleSlice = () => {
    setSlice(!slice)
  }

  function argWhere(array, conditionFn) {
    const indices = [];
    array.forEach((value, index) => {
        if (conditionFn(value)) {
            indices.push(index);
        }
    });
    return indices;
  }

  const Render = () => {
    const clusterArg = argWhere(props.result.cluster, x => x == props.clusterIdx);
    return clusterArg.map(idx =>
      <HeatmapLine token={props.result.token[idx]} color={props.result.color[idx]} just={props.result.just[idx]} mask={props.mask} expansion={props.expansion}/>
    );
  };

  return (
    <div className="flex flex-row col-span-1 bg-white h-auto p-1 px-2" onClick={toggleSlice}>
      <div className="flex-none w-8">
        {props.clusterIdx}
      </div>
      <div className="flex-auto w-auto">
        {props && (slice ? Render().slice(0, 3) : Render())}
      </div>
    </div>
  );
}

export default HeatmapRender;