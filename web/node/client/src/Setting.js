import React, { useState, useEffect } from 'react';

export function Setting(props) {
  const Render = () => {
    return (
      props.result.cluster.map((c, idx) => {
        if (c !== props.number) {
          return 0
        } else {
          return (
            <div>
              <p><HeatmapLine token={props.result.token[idx]} color={props.result.color[idx]}/></p>
            </div>
          );
        }
      }).filter(e => e).slice(0, 10)
    );
  }

  return (
      <div>
        {props && <div>{Render()}</div>}
      </div>
  );
}

export default Heatmap;