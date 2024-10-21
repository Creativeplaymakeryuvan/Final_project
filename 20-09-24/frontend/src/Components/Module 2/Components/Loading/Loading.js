import React from 'react';
import './loading.css';

const Loading = () => {
    return (
        <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <p>Scanning, please wait...</p>
        </div>
    );
};

export default Loading;
