import React from 'react';

export function ClusterNumber(props) {
  const Render = (data) => {
    const NumberList = [];
    data.cluster.forEach((value, idx) => {
      NumberList.push(<p>cluster={value}</p>);
    })
    console.log(NumberList);
    return NumberList;
  }

  return (
      <div>
        {props && <div>{Render(props.result)}</div>}
      </div>
  );
}

export default ClusterNumber;