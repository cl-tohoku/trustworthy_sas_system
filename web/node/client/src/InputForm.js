import React, { useState } from 'react';

const InputForm = (props) => {
  const [inputValue, setInputValue] = useState("");

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    props.setValue(inputValue)
  };

  return (
    <form onSubmit={handleSubmit} className="items-center justify-between">
      <input
        className="ml-1 outline-none bg-transparent text-gray-700"
        type="search"
        placeholder="Please enter a keyword"
        value={props.value}
        onChange={handleInputChange}
      />
    </form>
  );
};

export default InputForm;