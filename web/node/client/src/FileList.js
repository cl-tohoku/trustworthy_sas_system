import React, { useState, useEffect } from 'react';

const FileList = (props) => {
  const [files, setFiles] = useState(['']);

  const requestFileOptions = {
    method: 'GET',
    headers: {
      'Accept': 'application/json, */*',
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    },
  };

  // ファイルリストの読み込み
  useEffect(() => {
    fetch("/file", requestFileOptions)
      .then(response => response.json())
      .then(data => setFiles(data.file))
  }, []);

  const renderFileList = () => {
    return files.map(file => (
      <button className="flex items-center flex-shrink-0 h-10 px-2 text-sm font-medium rounded hover:bg-gray-300" onClick={() => props.setSetting(file)} key={file}>
        {file}
      </button>
    ))
  }
  return (
    <div>
      {files && renderFileList()}
    </div>
  );
};

export default FileList;