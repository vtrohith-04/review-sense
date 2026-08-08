
import React, { useState } from 'react';

interface ReviewInputFormProps {
    onSingleSubmit: (review: string) => void;
    onBatchSubmit: (reviews: string[]) => void;
    isLoading: boolean;
}

const ReviewInputForm: React.FC<ReviewInputFormProps> = ({ onSingleSubmit, onBatchSubmit, isLoading }) => {
    const [reviewText, setReviewText] = useState('');

    const handleSingleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (reviewText.trim()) {
            onSingleSubmit(reviewText.trim());
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const content = event.target?.result as string;
                const reviews = content.split('\n').map(line => line.trim()).filter(line => line.length > 0);
                if(reviews.length > 0) {
                    onBatchSubmit(reviews);
                }
            };
            reader.readAsText(file);
        }
        e.target.value = ''; // Reset file input
    };

    return (
        <div className="p-6 bg-gray-800 rounded-lg shadow-xl border border-gray-700 space-y-8">
            <div>
                <h2 className="text-xl font-semibold mb-3 text-gray-100">Analyze a Single Review</h2>
                <form onSubmit={handleSingleSubmit}>
                    <textarea
                        value={reviewText}
                        onChange={(e) => setReviewText(e.target.value)}
                        placeholder="Enter a customer review here..."
                        className="w-full h-32 p-3 bg-gray-900 border border-gray-600 rounded-md focus:ring-2 focus:ring-green-500 focus:outline-none transition duration-200 resize-none"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !reviewText.trim()}
                        className="mt-3 w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-500 disabled:cursor-not-allowed text-white font-bold py-2 px-4 rounded-md transition duration-200"
                    >
                        {isLoading ? 'Analyzing...' : 'Analyze Emotion'}
                    </button>
                </form>
            </div>
            <div className="border-t border-gray-700"></div>
            <div>
                <h2 className="text-xl font-semibold mb-3 text-gray-100">Analyze a Batch of Reviews</h2>
                <p className="text-sm text-gray-400 mb-3">Upload a .txt or .csv file with one review per line.</p>
                <input
                    type="file"
                    accept=".txt,.csv"
                    onChange={handleFileChange}
                    disabled={isLoading}
                    className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 disabled:opacity-50"
                />
            </div>
        </div>
    );
};

export default ReviewInputForm;
