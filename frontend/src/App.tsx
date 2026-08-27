import React, { useCallback, useState } from 'react';
import { BarChart, Bar, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { EMOTION_COLORS, EMOTION_LABELS, EMOTIONS } from './constants';
import { analyzeEmotion } from './services/emotionApi';
import { Emotion } from './types';
import type { AnalysisResult, DashboardData } from './types';

const sampleReview = "I was honestly surprised by how quickly support fixed my issue. The reply felt personal and saved me a lot of time.";

type ActiveView = 'dashboard' | 'history' | 'batchJobs' | 'settings';

const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: 'GR' },
    { id: 'history', label: 'History', icon: 'CL' },
    { id: 'batchJobs', label: 'Batch Jobs', icon: 'FI' },
    { id: 'settings', label: 'Settings', icon: 'GE' },
] as const;

const formatPercent = (score: number) => `${Math.round(score * 100)}%`;

const getSnippet = (review: string) =>
    review.length > 86 ? `${review.slice(0, 86)}...` : review;

const getModelLabel = (modelName?: string) => {
    if (!modelName) {
        return 'TF-IDF Logistic Regression Baseline';
    }

    return modelName
        .split('-')
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
};

const EmotionChip: React.FC<{ emotion: Emotion }> = ({ emotion }) => (
    <span
        className="inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-bold"
        style={{
            color: EMOTION_COLORS[emotion],
            backgroundColor: `${EMOTION_COLORS[emotion]}1a`,
        }}
    >
        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: EMOTION_COLORS[emotion] }} />
        {EMOTION_LABELS[emotion]}
    </span>
);

const Sidebar: React.FC<{
    activeView: ActiveView;
    onViewChange: (view: ActiveView) => void;
}> = ({ activeView, onViewChange }) => (
    <aside className="hidden h-screen w-64 shrink-0 border-r border-slate-200 bg-white px-5 py-6 lg:sticky lg:top-0 lg:flex lg:flex-col">
        <div className="mb-10 flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-xl bg-slate-950 text-white">
                <span className="text-lg font-black">RS</span>
            </div>
            <div>
                <p className="text-lg font-extrabold tracking-tight text-slate-950">Review Sense</p>
                <p className="text-xs font-semibold text-slate-500">Emotion Analytics</p>
            </div>
        </div>

        <nav className="space-y-2">
            {navItems.map((item) => (
                <button
                    key={item.label}
                    type="button"
                    onClick={() => onViewChange(item.id)}
                    className={`flex w-full items-center gap-3 rounded-lg px-4 py-3 text-left text-sm font-bold transition ${
                        activeView === item.id
                            ? 'bg-blue-600 text-white shadow-sm'
                            : 'text-slate-600 hover:bg-slate-100 hover:text-slate-950'
                    }`}
                >
                    <span className="grid h-5 w-5 place-items-center text-xs uppercase">{item.icon}</span>
                    {item.label}
                </button>
            ))}
        </nav>

        <div className="mt-auto rounded-xl border border-slate-200 bg-slate-50 p-3">
            <p className="text-xs font-bold uppercase tracking-wide text-slate-500">Active Model</p>
            <p className="mt-1 text-sm font-bold text-slate-900">TF-IDF Logistic Regression</p>
            <p className="mt-1 text-xs text-slate-500">Baseline model connected</p>
        </div>
    </aside>
);

const TopBar: React.FC<{ onNewAnalysis: () => void }> = ({ onNewAnalysis }) => (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 px-4 py-4 backdrop-blur md:px-8">
        <div className="flex items-center justify-between gap-4">
            <div className="flex min-w-0 flex-1 items-center gap-3 rounded-full bg-slate-100 px-4 py-2 text-slate-500 md:max-w-md">
                <span className="text-sm">Search</span>
                <input
                    className="w-full bg-transparent text-sm outline-none placeholder:text-slate-400"
                    placeholder="Search reviews, reports..."
                />
            </div>
            <button
                type="button"
                onClick={onNewAnalysis}
                className="rounded-full bg-slate-950 px-5 py-2 text-sm font-bold text-white shadow-sm transition hover:bg-blue-700"
            >
                + New Analysis
            </button>
        </div>
    </header>
);

const InputPanel: React.FC<{
    reviewText: string;
    isLoading: boolean;
    onReviewTextChange: (value: string) => void;
    onSingleSubmit: () => void;
    onBatchSubmit: (reviews: string[]) => void;
}> = ({ reviewText, isLoading, onReviewTextChange, onSingleSubmit, onBatchSubmit }) => {
    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) {
            return;
        }

        const reader = new FileReader();
        reader.onload = (readerEvent) => {
            const content = readerEvent.target?.result as string;
            const reviews = content
                .split(/\r?\n/)
                .map((line) => line.trim())
                .filter(Boolean);

            if (reviews.length > 0) {
                onBatchSubmit(reviews);
            }
        };
        reader.readAsText(file);
        event.target.value = '';
    };

    return (
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-5 flex items-center justify-between">
                <div>
                    <p className="text-xs font-extrabold uppercase tracking-wide text-slate-400">Data Input</p>
                    <h2 className="mt-1 text-xl font-extrabold text-slate-950">Analyze Review</h2>
                </div>
                <div className="rounded-lg bg-slate-100 p-1 text-xs font-bold">
                    <span className="rounded-md bg-white px-3 py-1 text-blue-700 shadow-sm">Text</span>
                    <span className="px-3 py-1 text-slate-500">Batch</span>
                </div>
            </div>

            <label className="text-xs font-bold uppercase tracking-wide text-slate-500">Paste Review Text</label>
            <textarea
                value={reviewText}
                onChange={(event) => onReviewTextChange(event.target.value)}
                className="mt-2 h-64 w-full resize-none rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm leading-6 text-slate-800 outline-none transition focus:border-blue-500 focus:ring-4 focus:ring-blue-100"
                placeholder={sampleReview}
                disabled={isLoading}
            />

            <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <span className="text-xs font-semibold text-slate-500">{reviewText.length} / 5000 chars</span>
                <button
                    type="button"
                    onClick={onSingleSubmit}
                    disabled={isLoading || !reviewText.trim()}
                    className="rounded-xl bg-blue-600 px-6 py-3 text-sm font-extrabold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
                >
                    {isLoading ? 'Analyzing...' : 'Analyze Emotion ->'}
                </button>
            </div>

            <label className="mt-5 flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-center transition hover:border-blue-300 hover:bg-blue-50">
                <span className="text-sm font-bold text-slate-800">Upload Batch Data</span>
                <span className="mt-1 text-xs font-semibold text-slate-500">CSV or TXT, one review per line</span>
                <input className="hidden" type="file" accept=".txt,.csv" onChange={handleFileChange} disabled={isLoading} />
            </label>
        </section>
    );
};

const PrimaryEmotionCard: React.FC<{ result: AnalysisResult | null; isLoading: boolean }> = ({ result, isLoading }) => {
    const emotion = result?.primaryEmotion ?? Emotion.Neutral;
    const topScore = result?.topEmotions[0]?.score ?? 0;
    const color = EMOTION_COLORS[emotion];

    return (
        <section className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="absolute right-0 top-0 h-40 w-40 rounded-full blur-3xl" style={{ backgroundColor: `${color}26` }} />
            <div className="relative">
                <p className="text-xs font-extrabold uppercase tracking-wide text-slate-400">Primary Emotion</p>
                <div className="mt-5 flex flex-col gap-6 sm:flex-row sm:items-center">
                    <div className="grid h-36 w-36 shrink-0 place-items-center rounded-full border-8 border-white shadow-inner" style={{ backgroundColor: `${color}20` }}>
                        <div className="text-center">
                            <p className="text-4xl font-extrabold" style={{ color }}>
                                {result ? formatPercent(topScore) : '--'}
                            </p>
                            <p className="mt-1 text-xs font-bold uppercase tracking-wide text-slate-500">Confidence</p>
                        </div>
                    </div>
                    <div>
                        <EmotionChip emotion={emotion} />
                        <h3 className="mt-3 text-5xl font-extrabold tracking-tight text-slate-950">
                            {isLoading ? 'Reading...' : result ? EMOTION_LABELS[emotion] : 'Ready'}
                        </h3>
                        <p className="mt-3 max-w-xl text-sm leading-6 text-slate-600">
                            {result
                                ? `The model found ${EMOTION_LABELS[emotion].toLowerCase()} as the strongest emotional signal in this review.`
                                : 'Paste a customer review to see the detected emotion, confidence score, and top supporting signals.'}
                        </p>
                        <p className="mt-4 text-xs font-bold uppercase tracking-wide text-slate-400">
                            Model: {getModelLabel(result?.modelName)}
                        </p>
                    </div>
                </div>
            </div>
        </section>
    );
};

const EmotionBars: React.FC<{ result: AnalysisResult | null }> = ({ result }) => {
    const rows = result?.topEmotions.length
        ? result.topEmotions
        : EMOTIONS.slice(0, 4).map((label) => ({ label, score: 0 }));

    return (
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="text-lg font-extrabold text-slate-950">Top Emotion Scores</h3>
            <div className="mt-6 space-y-5">
                {rows.map(({ label, score }) => (
                    <div key={label}>
                        <div className="mb-2 flex items-center justify-between text-sm font-bold">
                            <span className="flex items-center gap-2 text-slate-700">
                                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: EMOTION_COLORS[label] }} />
                                {EMOTION_LABELS[label]}
                            </span>
                            <span className="text-slate-500">{formatPercent(score)}</span>
                        </div>
                        <div className="h-2 overflow-hidden rounded-full" style={{ backgroundColor: `${EMOTION_COLORS[label]}1f` }}>
                            <div
                                className="h-full rounded-full transition-all duration-700"
                                style={{ width: `${Math.round(score * 100)}%`, backgroundColor: EMOTION_COLORS[label] }}
                            />
                        </div>
                    </div>
                ))}
            </div>
        </section>
    );
};

const DistributionChart: React.FC<{ dashboardData: DashboardData | null; result: AnalysisResult | null }> = ({ dashboardData, result }) => {
    const chartData = dashboardData?.emotionCounts.length
        ? dashboardData.emotionCounts
        : result
            ? result.topEmotions.map((emotion) => ({ name: emotion.label, value: Math.round(emotion.score * 100) }))
            : EMOTIONS.map((emotion) => ({ name: emotion, value: 0 }));

    return (
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="text-lg font-extrabold text-slate-950">Emotion Distribution</h3>
            <div className="mt-4 h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
                        <XAxis dataKey="name" tickFormatter={(value: Emotion) => EMOTION_LABELS[value]} stroke="#64748b" fontSize={11} />
                        <YAxis stroke="#64748b" fontSize={11} allowDecimals={false} />
                        <Tooltip />
                        <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                            {chartData.map((entry) => (
                                <Cell key={entry.name} fill={EMOTION_COLORS[entry.name]} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </section>
    );
};

const RecentAnalyses: React.FC<{ analyses: AnalysisResult[] }> = ({ analyses }) => (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm">
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4">
            <h3 className="text-lg font-extrabold text-slate-950">Recent Analyses</h3>
            <span className="text-xs font-bold text-blue-700">{analyses.length} saved locally</span>
        </div>
        <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-sm">
                <thead className="bg-slate-50 text-xs font-extrabold uppercase tracking-wide text-slate-500">
                    <tr>
                        <th className="px-5 py-3">Timestamp</th>
                        <th className="px-5 py-3">Review Snippet</th>
                        <th className="px-5 py-3">Primary Emotion</th>
                        <th className="px-5 py-3 text-right">Confidence</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                    {analyses.length === 0 ? (
                        <tr>
                            <td className="px-5 py-8 text-center text-slate-500" colSpan={4}>
                                New results will appear here after you analyze a review.
                            </td>
                        </tr>
                    ) : (
                        analyses.map((analysis, index) => (
                            <tr key={`${analysis.review}-${index}`} className="hover:bg-slate-50">
                                <td className="whitespace-nowrap px-5 py-4 text-slate-500">Just now</td>
                                <td className="px-5 py-4 font-medium text-slate-800">"{getSnippet(analysis.review)}"</td>
                                <td className="px-5 py-4">
                                    <EmotionChip emotion={analysis.primaryEmotion} />
                                </td>
                                <td className="px-5 py-4 text-right font-bold text-slate-800">
                                    {formatPercent(analysis.topEmotions[0]?.score ?? 0)}
                                </td>
                            </tr>
                        ))
                    )}
                </tbody>
            </table>
        </div>
    </section>
);

const BatchSummary: React.FC<{ dashboardData: DashboardData | null }> = ({ dashboardData }) => (
    <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <p className="text-xs font-extrabold uppercase tracking-wide text-slate-400">Reviews Processed</p>
            <p className="mt-2 text-3xl font-extrabold text-slate-950">{dashboardData?.totalReviews ?? 0}</p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <p className="text-xs font-extrabold uppercase tracking-wide text-slate-400">Average Confidence</p>
            <p className="mt-2 text-3xl font-extrabold text-slate-950">{formatPercent(dashboardData?.averageConfidence ?? 0)}</p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <p className="text-xs font-extrabold uppercase tracking-wide text-slate-400">Dominant Emotion</p>
            <p className="mt-3">
                {dashboardData && dashboardData.overallSentiment !== 'Mixed'
                    ? <EmotionChip emotion={dashboardData.overallSentiment} />
                    : <span className="text-sm font-semibold text-slate-500">No batch yet</span>}
            </p>
        </div>
    </div>
);

const PageHeader: React.FC<{
    eyebrow: string;
    title: string;
    description: string;
    children?: React.ReactNode;
}> = ({ eyebrow, title, description, children }) => (
    <div className="mb-6 flex flex-col justify-between gap-4 xl:flex-row xl:items-end">
        <div>
            <p className="text-sm font-extrabold uppercase tracking-wide text-blue-700">{eyebrow}</p>
            <h1 className="mt-2 text-4xl font-extrabold tracking-tight text-slate-950 md:text-5xl">{title}</h1>
            <p className="mt-3 max-w-3xl text-base leading-7 text-slate-600">{description}</p>
        </div>
        {children}
    </div>
);

const StatusBadges: React.FC = () => (
    <div className="flex gap-3">
        <span className="rounded-full bg-emerald-50 px-4 py-2 text-xs font-extrabold text-emerald-700">
            API Connected
        </span>
        <span className="rounded-full bg-amber-50 px-4 py-2 text-xs font-extrabold text-amber-700">
            Baseline Model
        </span>
    </div>
);

const DashboardView: React.FC<{
    reviewText: string;
    isLoading: boolean;
    error: string | null;
    singleResult: AnalysisResult | null;
    dashboardData: DashboardData | null;
    recentAnalyses: AnalysisResult[];
    onReviewTextChange: (value: string) => void;
    onSingleSubmit: () => void;
    onBatchSubmit: (reviews: string[]) => void;
}> = ({
    reviewText,
    isLoading,
    error,
    singleResult,
    dashboardData,
    recentAnalyses,
    onReviewTextChange,
    onSingleSubmit,
    onBatchSubmit,
}) => (
    <>
        <PageHeader
            eyebrow="Review Sense"
            title="Emotion Analysis Dashboard"
            description="Analyze customer review emotions in real time and turn messy feedback into clear signals."
        >
            <StatusBadges />
        </PageHeader>

        {error && (
            <div className="mb-5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-700">
                {error}
            </div>
        )}

        <div className="grid gap-6 xl:grid-cols-12">
            <div className="xl:col-span-5">
                <InputPanel
                    reviewText={reviewText}
                    isLoading={isLoading}
                    onReviewTextChange={onReviewTextChange}
                    onSingleSubmit={onSingleSubmit}
                    onBatchSubmit={onBatchSubmit}
                />
            </div>
            <div className="space-y-6 xl:col-span-7">
                <PrimaryEmotionCard result={singleResult} isLoading={isLoading} />
                <div className="grid gap-6 lg:grid-cols-2">
                    <EmotionBars result={singleResult} />
                    <DistributionChart dashboardData={dashboardData} result={singleResult} />
                </div>
            </div>
        </div>

        <div className="mt-6">
            <BatchSummary dashboardData={dashboardData} />
        </div>

        <div className="mt-6">
            <RecentAnalyses analyses={recentAnalyses} />
        </div>
    </>
);

const HistoryView: React.FC<{ analyses: AnalysisResult[] }> = ({ analyses }) => (
    <>
        <PageHeader
            eyebrow="Review Archive"
            title="Analysis History"
            description="Review the latest emotion predictions created during this local session."
        >
            <span className="rounded-full bg-blue-50 px-4 py-2 text-xs font-extrabold text-blue-700">
                {analyses.length} local results
            </span>
        </PageHeader>
        <RecentAnalyses analyses={analyses} />
    </>
);

const BatchJobsView: React.FC<{
    dashboardData: DashboardData | null;
    analyses: AnalysisResult[];
}> = ({ dashboardData, analyses }) => (
    <>
        <PageHeader
            eyebrow="Batch Processing"
            title="Batch Jobs"
            description="Track uploaded TXT or CSV review batches and see their strongest emotion patterns."
        >
            <span className="rounded-full bg-slate-900 px-4 py-2 text-xs font-extrabold text-white">
                Upload from Dashboard
            </span>
        </PageHeader>
        <BatchSummary dashboardData={dashboardData} />
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <DistributionChart dashboardData={dashboardData} result={analyses[0] ?? null} />
            <EmotionBars result={analyses[0] ?? null} />
        </div>
        <div className="mt-6">
            <RecentAnalyses analyses={analyses} />
        </div>
    </>
);

const SettingsView: React.FC = () => (
    <>
        <PageHeader
            eyebrow="System"
            title="Settings"
            description="Keep model configuration visible while we decide what to train or add next."
        >
            <StatusBadges />
        </PageHeader>
        <div className="grid gap-6 lg:grid-cols-2">
            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <p className="text-xs font-extrabold uppercase tracking-wide text-slate-400">Model Configuration</p>
                <h2 className="mt-2 text-xl font-extrabold text-slate-950">Baseline Emotion Model</h2>
                <div className="mt-5 space-y-3 text-sm text-slate-600">
                    <p><span className="font-bold text-slate-900">Model:</span> TF-IDF + Logistic Regression</p>
                    <p><span className="font-bold text-slate-900">Mode:</span> multi-label emotion classification</p>
                    <p><span className="font-bold text-slate-900">API route:</span> /api/v1/predict/emotion</p>
                </div>
            </section>
            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <p className="text-xs font-extrabold uppercase tracking-wide text-slate-400">Next Capability</p>
                <h2 className="mt-2 text-xl font-extrabold text-slate-950">Fake or Spam Review Detection</h2>
                <p className="mt-4 text-sm leading-6 text-slate-600">
                    This is a strong next feature because it turns Review Sense from emotion analytics into review quality intelligence.
                </p>
            </section>
        </div>
    </>
);

const App: React.FC = () => {
    const [reviewText, setReviewText] = useState(sampleReview);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [singleResult, setSingleResult] = useState<AnalysisResult | null>(null);
    const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
    const [recentAnalyses, setRecentAnalyses] = useState<AnalysisResult[]>([]);
    const [activeView, setActiveView] = useState<ActiveView>('dashboard');

    const saveRecent = (results: AnalysisResult[]) => {
        setRecentAnalyses((current) => [...results, ...current].slice(0, 6));
    };

    const handleSingleSubmit = useCallback(async () => {
        if (!reviewText.trim()) {
            return;
        }

        setError(null);
        setDashboardData(null);
        setIsLoading(true);

        try {
            const result = await analyzeEmotion(reviewText.trim());
            setSingleResult(result);
            saveRecent([result]);
            setActiveView('dashboard');
        } catch (caughtError) {
            setError(caughtError instanceof Error ? caughtError.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [reviewText]);

    const handleBatchSubmit = useCallback(async (reviews: string[]) => {
        setError(null);
        setIsLoading(true);

        try {
            const results = await Promise.all(reviews.map((review) => analyzeEmotion(review)));
            const emotionCounts = results.reduce((counts, result) => {
                counts[result.primaryEmotion] = (counts[result.primaryEmotion] ?? 0) + 1;
                return counts;
            }, {} as Partial<Record<Emotion, number>>);

            const formattedCounts = Object.entries(emotionCounts)
                .map(([name, value]) => ({ name: name as Emotion, value }))
                .sort((left, right) => right.value - left.value);

            const averageConfidence = results.reduce((sum, result) => sum + (result.topEmotions[0]?.score ?? 0), 0) / results.length;

            setSingleResult(results[0] ?? null);
            setDashboardData({
                totalReviews: results.length,
                emotionCounts: formattedCounts,
                overallSentiment: formattedCounts[0]?.name ?? 'Mixed',
                averageConfidence,
            });
            saveRecent(results);
            setActiveView('batchJobs');
        } catch (caughtError) {
            setError(caughtError instanceof Error ? caughtError.message : 'An unknown error occurred during batch processing.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    const handleNewAnalysis = () => {
        setError(null);
        setReviewText('');
        setActiveView('dashboard');
    };

    const renderActiveView = () => {
        switch (activeView) {
            case 'history':
                return <HistoryView analyses={recentAnalyses} />;
            case 'batchJobs':
                return <BatchJobsView dashboardData={dashboardData} analyses={recentAnalyses} />;
            case 'settings':
                return <SettingsView />;
            default:
                return (
                    <DashboardView
                        reviewText={reviewText}
                        isLoading={isLoading}
                        error={error}
                        singleResult={singleResult}
                        dashboardData={dashboardData}
                        recentAnalyses={recentAnalyses}
                        onReviewTextChange={setReviewText}
                        onSingleSubmit={handleSingleSubmit}
                        onBatchSubmit={handleBatchSubmit}
                    />
                );
        }
    };

    return (
        <div className="min-h-screen bg-slate-50" style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }}>
            <div className="flex">
                <Sidebar activeView={activeView} onViewChange={setActiveView} />
                <div className="min-w-0 flex-1">
                    <TopBar onNewAnalysis={handleNewAnalysis} />
                    <main className="px-4 py-6 md:px-8">
                        {renderActiveView()}
                    </main>
                </div>
            </div>
        </div>
    );
};

export default App;
