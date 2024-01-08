import React, { useState, useEffect } from 'react';

function Scatter(props) {
    return (
        <div>
            {props.imagePath && (
                <img
                    src={props.imagePath}
                    alt="Example Image"
                    className="h-60 w-96"
                />
            )}
        </div>
    );
}

export default Scatter;