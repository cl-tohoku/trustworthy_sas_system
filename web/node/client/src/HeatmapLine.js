import React from 'react';

export function HeatmapLine(props) {
  const Render = (token, color, just) => {
    if (props.mask) {
      return token.map((t, jdx) => {
        return <span className="text-sm" style={{backgroundColor: color[jdx], textDecorationColor: just[jdx],
           textDecorationLine: "underline", textDecorationThickness: "4px"}}>{t}</span>
      });
    } else {
      return token.map((t, jdx) => {
        return <span className="text-sm" style={{backgroundColor: color[jdx]}}>{t}</span>
      });
    }
  }

  // {data ? <div>{data.cluster}</div> : <button onClick={GetData}>データを取得</button>}
  // {data ? <div>Hello?, {data.cluster} </div> : <div> none </div>}
  // {data && <div>{data.cluster}</div>}
  const convert = (just) => {
    return just.map((j) => {
      return j === "#f0e68c" ? "#4169e1" : j
    })
  }

  return (
      <div>
        {props && <div>{Render(props.token, props.color, convert(props.just))}</div>}
      </div>
  );
}

export default HeatmapLine;