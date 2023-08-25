import React, { useState, useEffect } from 'react';

function Scatter(props) {
    return (
        <div>
            {imageData && (
                <img
                    src="./data/tmp/scatter.png"
                    alt="Example Image"
                    className="w-auto h-auto"
                />
            )}
        </div>
    );
}

export default Scatter;