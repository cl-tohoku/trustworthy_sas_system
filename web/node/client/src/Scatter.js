import React, { useState, useEffect } from 'react';

function Scatter(props) {
    return (
        <div>
            {props.imagePath && (
                <img
                    src={props.imagePath}
                    alt="Example Image"
                    className="h-80 w-auto"
                />
            )}
        </div>
    );
}

export default Scatter;