import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { AnalysisResult, DashboardData } from '../types';
import { Emotion } from '../types';
import { EMOTION_COLORS } from '../constants';

interface EmotionBadgeProps {
    emotion: Emotion;
    score: number;
}

const EmotionBadge: React.FC<EmotionBadgeProps> = ({ emotion, score }) => (
    <div
        className="flex items-center justify-between p-3 rounded-lg text-white"
        style={{ backgroundColor: `${EMOTION_COLORS[emotion]}33` }} // Add alpha for background
    >
        <span className="font-semibold">{emotion}</span>
        <span
            className="font-bold px-2 py-1 rounded"
            style={{ backgroundColor: EMOTION_COLORS[emotion] }}
        >
            {(score * 100).toFixed(1)}%
        </span>
    </div>
);

const renderSingleResult = (result: AnalysisResult) => {
    const sortedEmotions = Object.entries(result.scores)
        .sort(([, a], [, b]) => b - a) as [Emotion, number][];
    
    const primaryEmotion = sortedEmotions[0][0];

    return (
        <div className="space-y-6">
            <div>
                <h3 className="text-lg font-semibold text-gray-300 mb-2">Original Review:</h3>
                <p className="p-4 bg-gray-900 border border-gray-700 rounded-md italic text-gray-300">"{result.review}"</p>
            </div>
            <div>
                <h3 className="text-lg font-semibold text-gray-300 mb-2">Primary Emotion:</h3>
                 <div
                    className="inline-block text-xl font-bold px-4 py-2 rounded-lg text-white"
                    style={{ backgroundColor: EMOTION_COLORS[primaryEmotion] }}
                >
                    {primaryEmotion}
                </div>
            </div>
            <div>
                <h3 className="text-lg font-semibold text-gray-300 mb-2">Emotion Scores:</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {sortedEmotions.map(([emotion, score]) => (
                        <EmotionBadge key={emotion} emotion={emotion} score={score} />
                    ))}
                </div>
            </div>
        </div>
    );
};

const renderDashboard = (data: DashboardData) => {
    return (
        <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-center">
                <div className="p-4 bg-gray-900 rounded-lg border border-gray-700">
                    <p className="text-sm text-gray-400">Total Reviews Analyzed</p>
                    <p className="text-3xl font-bold">{data.totalReviews}</p>
                </div>
                <div className="p-4 bg-gray-900 rounded-lg border border-gray-700">
                    <p className="text-sm text-gray-400">Overall Sentiment</p>
                    <p className="text-3xl font-bold" style={{color: data.overallSentiment !== 'Mixed' ? EMOTION_COLORS[data.overallSentiment] : '#ffffff'}}>
                        {data.overallSentiment}
                    </p>
                </div>
            </div>
            <div>
                 <h3 className="text-lg font-semibold text-gray-300 mb-4">Emotion Distribution</h3>
                 <div className="w-full h-80 bg-gray-900 p-4 rounded-lg border border-gray-700">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data.emotionCounts} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
                            <XAxis dataKey="name" stroke="#a0aec0" />
                            <YAxis stroke="#a0aec0" />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1a202c', border: '1px solid #4a5568' }}
                                labelStyle={{ color: '#a0aec0' }}
                                itemStyle={{ fontWeight: 'bold' }}
                            />
                            <Bar dataKey="value" name="Count">
                                {data.emotionCounts.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={EMOTION_COLORS[entry.name]} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                 </div>
            </div>
        </div>
    );
};


interface AnalysisResultsProps {
    singleResult: AnalysisResult | null;
    dashboardData: DashboardData | null;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ singleResult, dashboardData }) => {
    let content;
    if (dashboardData) {
        content = renderDashboard(dashboardData);
    } else if (singleResult) {
        content = renderSingleResult(singleResult);
    } else {
        content = (
            <div className="text-center text-gray-400">
                <h2 className="text-2xl font-bold mb-2">Welcome to Review Sense</h2>
                <p>Enter a review or upload a file to begin analyzing customer emotions.</p>
            </div>
        );
    }

    return (
        <div className="p-6 bg-gray-800 rounded-lg shadow-xl border border-gray-700 h-full">
            <h2 className="text-xl font-semibold mb-6 text-gray-100 border-b border-gray-700 pb-3">Analysis Results</h2>
            {content}
        </div>
    );
};

export default AnalysisResults;