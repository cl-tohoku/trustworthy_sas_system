import React, { useState, useEffect } from 'react';

function Dendrogram(props) {
    return (
        <div>
            {props.imagePath && (
                <img
                    src={props.imagePath}
                    alt="Example Image"
                    className="h-auto w-auto"
                />
            )}
        </div>
    );
}

export default Dendrogram;