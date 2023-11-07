import React from 'react';

export function HeatmapLine(props) {
  const NormalLine = (t, c, j) => {
    return (
      <span className="text-sm" style={{backgroundColor: c, textDecorationColor: j,
        textDecorationLine: "underline", textDecorationThickness: "4px"}}>{t}</span>
    );
  };
  const ExplansionLine = (t, c, j) => {
    return (
      <span className="text-sm mb-4" style={{backgroundColor: c, textDecorationColor: j,
        textDecorationLine: "underline", textDecorationThickness: "4px"}}>{t}</span>
    );
  };

  const Render = (token, color, just) => {
    if (props.mask) {
      if (props.exansion) {
        return token.map((t, jdx) => {
          return ExplansionLine(t, color[jdx], just[jdx]);
        });
      } else {
        return token.map((t, jdx) => {
          return NormalLine(t, color[jdx], just[jdx]);
        });
      };
    } else {
      return token.map((t, jdx) => {
        return <span className="text-sm" style={{backgroundColor: color[jdx]}}>{t}</span>
      });
    }
  }

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