
import React, { useState, useCallback } from 'react';
import Header from './components/Header';
import ReviewInputForm from './components/ReviewInputForm';
import AnalysisResults from './components/AnalysisResults';
import Loader from './components/Loader';
import { analyzeEmotion } from './services/geminiService';
import { Emotion } from './types';
import type { AnalysisResult, DashboardData } from './types';

const App: React.FC = () => {
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [singleResult, setSingleResult] = useState<AnalysisResult | null>(null);
    const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);

    const resetState = () => {
        setError(null);
        setSingleResult(null);
        setDashboardData(null);
    };

    const handleSingleSubmit = useCallback(async (review: string) => {
        resetState();
        setIsLoading(true);
        try {
            const scores = await analyzeEmotion(review);
            setSingleResult({ review, scores });
        } catch (e) {
            setError(e instanceof Error ? e.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    const handleBatchSubmit = useCallback(async (reviews: string[]) => {
        resetState();
        setIsLoading(true);
        try {
            const results: AnalysisResult[] = await Promise.all(
                reviews.map(async (review) => {
                    const scores = await analyzeEmotion(review);
                    return { review, scores };
                })
            );
            
            const emotionCounts: { [key in Emotion]?: number } = {};
            results.forEach(result => {
                const primaryEmotion = Object.entries(result.scores).reduce((a, b) => a[1] > b[1] ? a : b)[0] as Emotion;
                emotionCounts[primaryEmotion] = (emotionCounts[primaryEmotion] || 0) + 1;
            });

            const formattedCounts = (Object.entries(emotionCounts) as [Emotion, number][])
                .map(([name, value]) => ({ name, value }))
                .sort((a, b) => b.value - a.value);
            
            let overallSentiment: Emotion | 'Mixed' = 'Mixed';
            if (formattedCounts.length > 0) {
                 overallSentiment = formattedCounts[0].name;
            }

            setDashboardData({
                totalReviews: reviews.length,
                emotionCounts: formattedCounts,
                overallSentiment
            });

        } catch (e) {
            setError(e instanceof Error ? e.message : 'An unknown error occurred during batch processing.');
        } finally {
            setIsLoading(false);
        }
    }, []);


    return (
        <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center">
            <Header />
            <main className="w-full max-w-7xl mx-auto p-4 md:p-8 flex-grow">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div>
                        <ReviewInputForm
                            onSingleSubmit={handleSingleSubmit}
                            onBatchSubmit={handleBatchSubmit}
                            isLoading={isLoading}
                        />
                    </div>
                    <div className="flex items-center justify-center min-h-[400px]">
                        {isLoading ? (
                            <Loader />
                        ) : error ? (
                            <div className="text-center text-red-400 p-6 bg-red-900/50 border border-red-500 rounded-lg">
                                <h3 className="font-bold text-lg mb-2">Analysis Failed</h3>
                                <p>{error}</p>
                            </div>
                        ) : (
                           <AnalysisResults singleResult={singleResult} dashboardData={dashboardData} />
                        )}
                    </div>
                </div>
            </main>
             <footer className="w-full text-center p-4 text-gray-500 text-sm">
                <p>Powered by Google Gemini</p>
            </footer>
        </div>
    );
};

export default App;
