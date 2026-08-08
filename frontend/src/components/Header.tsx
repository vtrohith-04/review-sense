
import React from 'react';

const Header: React.FC = () => {
    return (
        <header className="w-full p-4 border-b border-gray-700">
            <h1 className="text-3xl font-bold text-center text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-blue-500">
                Review Sense
            </h1>
            <p className="text-center text-gray-400 mt-1">AI-Powered Emotion Analysis for Customer Reviews</p>
        </header>
    );
};

export default Header;
