import React, { useState, useEffect } from 'react';
import HeatmapLine from "./HeatmapLine";

export function Heatmap(props) {
  const [slice, setSlice] = useState(true);

  const changeValue = () => {
    setSlice(!slice)
  }


  const Render = () => {
    const heatmap = () => {
      return props.result.cluster.map((c, idx) => {
        if (c !== props.number) {
          return 0
        } else {
          return (
            <div onClick={changeValue}>
              <p><HeatmapLine token={props.result.token[idx]} color={props.result.color[idx]} just={props.result.just[idx]} /></p>
            </div>
          );
        }
      }).filter(e => e)
    }

    const slicer = () => {
      if (!slice) {
        return (
          <div className="col-span-1 bg-white h-auto p-1 px-2">
            {heatmap()}
          </div>
        );
      } else if (parseInt(props.size, 10) < 11) {
        return (
          <div className="col-span-1 bg-white h-[86px] p-1 px-2">
            {heatmap().slice(0, 3)}
          </div>
        );
      } else if (parseInt(props.size, 10) < 21) {
        return (
          <div className="col-span-1 bg-white h-[88px] p-1 px-2">
            {heatmap().slice(0, 3)}
          </div>
        );
      } else {
        return (
          <div className="col-span-1 bg-white h-[89px] p-1 px-2">
            {heatmap().slice(0, 3)}
          </div>
        );
      };
    }
    return slicer();
  }


  return (
    <div>
      {props && <div>{Render()}</div>}
    </div>
  );
}

export default Heatmap;