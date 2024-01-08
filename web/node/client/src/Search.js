import React, { useState } from 'react';

const SearchBox = (props) => {
  const [inputValue, setInputValue] = useState("");

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    props.setMode(true)
    props.setKeyword(inputValue)
  };

  const endpoint = "/search"
  const requestOptions = {
    method: 'POST',
    headers: {
      'Accept': 'application/json, */*',
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    },
    body: JSON.stringify({
      'setting': props.setting,
      'data_type': props.dataType,
      'keyword': inputValue,
    }),
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-center justify-between border border-gray-300 p-2 rounded-md shadow-sm">
      <input
        className="ml-1 outline-none bg-transparent text-gray-700"
        type="search"
        placeholder="Please enter a keyword"
        value={inputValue}
        onChange={handleInputChange}
      />
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" className="h-5 w-5 text-gray-400">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    </form>
  );
};

export default SearchBox;