import React from 'react';

export function ClusterRange(props) {
  return (
    <div>
      <label for="customRange3" class="inline-block text-neutral-700 dark:text-neutral-200">Size={props.clusterSize}
      </label>
      <input type="range" className="h-4 w-32 px-2 ml-auto text-sm font-medium rounded hover:bg-gray-300"
        min="2" max="30" onChange={props.SetClusterSize} id="customRange3" />
    </div>
  );
}

export default ClusterRange;