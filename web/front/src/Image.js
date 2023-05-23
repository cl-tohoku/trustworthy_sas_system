import React, { useState, useEffect } from 'react';

function Image(props) {
    const [imageData, setImageData] = useState(null);
    const url = `/${props.endpoint}/${props.dataType}/${props.setting}/${props.size}`

    useEffect(() => {
        fetch(url)
            .then(response => response.blob())
            .then(data => {
                setImageData(URL.createObjectURL(data));
            });
    }, [url]);

    return (
        <div>
            {imageData && (
                <img
                    src={imageData}
                    alt="Example Image"
                    className="w-auto h-auto"
                />
            )}
        </div>
    );
}

export default Image;