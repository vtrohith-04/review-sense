import React, { useCallback, useEffect, useId, useMemo, useState } from 'react';
import {
    DEFAULT_THRESHOLDS,
    EMOTION_BG_COLORS,
    EMOTION_COLORS,
    EMOTION_DESCRIPTIONS,
    EMOTION_ICONS,
    EMOTION_LABELS,
    EMOTIONS,
    PRESET_REVIEWS,
} from './constants';
import { analyzeBatch, analyzeEmotion, getBackendHealth } from './services/emotionApi';
import { Emotion } from './types';
import type { AnalysisResult, BackendHealth, BatchJobRecord } from './types';

type ActiveView = 'dashboard' | 'history' | 'batchJobs' | 'settings';

const DEFAULT_REVIEW = PRESET_REVIEWS[1].text;

export const App: React.FC = () => {
    const [activeView, setActiveView] = useState<ActiveView>('dashboard');
    const [reviewText, setReviewText] = useState<string>(DEFAULT_REVIEW);
    const [inputMode, setInputMode] = useState<'text' | 'batch'>('text');
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [currentResult, setCurrentResult] = useState<AnalysisResult | null>(null);
    const [history, setHistory] = useState<AnalysisResult[]>([]);
    const [batchJobs, setBatchJobs] = useState<BatchJobRecord[]>([]);
    const [backendHealth, setBackendHealth] = useState<BackendHealth>({
        status: 'offline',
        activeModel: 'checking...',
        device: 'unknown',
        version: '1.0.0',
    });
    const [thresholds, setThresholds] = useState<Record<Emotion, number>>(DEFAULT_THRESHOLDS);
    const [searchQuery, setSearchQuery] = useState<string>('');

    // Fetch backend health on startup
    const checkHealth = useCallback(async () => {
        const health = await getBackendHealth();
        setBackendHealth(health);
    }, []);

    useEffect(() => {
        checkHealth();
        const timer = setInterval(checkHealth, 15000);
        return () => clearInterval(timer);
    }, [checkHealth]);

    // Handle single review analysis
    const handleAnalyze = async (textToAnalyze?: string) => {
        const text = (textToAnalyze ?? reviewText).trim();
        if (!text) {
            setError('Please enter a review text to analyze.');
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            const result = await analyzeEmotion(text, thresholds);
            setCurrentResult(result);
            setHistory((prev) => [result, ...prev.slice(0, 49)]);
        } catch (err: any) {
            setError(err.message || 'An error occurred while analyzing the emotion.');
        } finally {
            setIsLoading(false);
        }
    };

    // Run initial demo analysis on load
    useEffect(() => {
        handleAnalyze(DEFAULT_REVIEW);
    }, []);

    // Handle batch CSV upload
    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setIsLoading(true);
        setError(null);

        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const content = e.target?.result as string;
                const rows = content
                    .split(/\r?\n/)
                    .map((r) => r.trim().replace(/^["']|["']$/g, ''))
                    .filter((r) => r.length > 5);

                const header = rows[0]?.toLowerCase();
                const startIndex = header?.includes('review') || header?.includes('text') ? 1 : 0;
                const sampleTexts = rows.slice(startIndex, startIndex + 100);

                if (sampleTexts.length === 0) {
                    throw new Error('No valid review text found in the uploaded file.');
                }

                const batchRes = await analyzeBatch(sampleTexts);

                // Calculate dominant emotion
                const counts: Partial<Record<Emotion, number>> = {};
                batchRes.results.forEach((res) => {
                    counts[res.primaryEmotion] = (counts[res.primaryEmotion] || 0) + 1;
                });
                const sortedEmotions = Object.entries(counts).sort((a, b) => (b[1] || 0) - (a[1] || 0));
                const dominant = (sortedEmotions[0]?.[0] as Emotion) || Emotion.Neutral;

                const newJob: BatchJobRecord = {
                    id: Math.random().toString(36).substring(2, 9),
                    filename: file.name,
                    totalReviews: batchRes.results.length,
                    dominantEmotion: dominant,
                    avgConfidence: Math.round(
                        (batchRes.results.reduce((acc, curr) => acc + curr.primaryScore, 0) / batchRes.results.length) * 100
                    ),
                    latencyMs: Math.round(batchRes.totalLatencyMs),
                    timestamp: new Date().toLocaleDateString() + ' ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                    status: 'completed',
                    results: batchRes.results,
                };

                setBatchJobs((prev) => [newJob, ...prev]);
                setActiveView('batchJobs');
            } catch (err: any) {
                setError(err.message || 'Failed to process CSV file.');
            } finally {
                setIsLoading(false);
            }
        };
        reader.readAsText(file);
    };

    const formatPercent = (score: number) => `${Math.round(score * 100)}%`;

    const getDisplayModelName = (modelName?: string) => {
        if (!modelName) return 'DeBERTa-v3 Emotion Transformer';
        if (modelName.toLowerCase().includes('deberta')) return 'DeBERTa-v3 Multi-Label Transformer';
        if (modelName.toLowerCase().includes('tfidf') || modelName.toLowerCase().includes('baseline')) {
            return 'TF-IDF Logistic Regression Baseline';
        }
        return modelName;
    };

    // Filtered history
    const filteredHistory = useMemo(() => {
        if (!searchQuery.trim()) return history;
        const q = searchQuery.toLowerCase();
        return history.filter(
            (item) => item.review.toLowerCase().includes(q) || item.primaryEmotion.toLowerCase().includes(q)
        );
    }, [history, searchQuery]);

    const isHealthy = backendHealth.status === 'ok';

    return (
        <div className="flex min-h-screen bg-[#F8FAFC] text-[#0F172A] font-sans antialiased">
            {/* FIXED LEFT SIDEBAR */}
            <aside className="fixed left-0 top-0 h-full w-64 bg-white border-r border-[#E2E8F0] z-50 flex flex-col justify-between shadow-sm">
                <div>
                    {/* Logo & App Name */}
                    <div className="h-16 flex items-center px-6 gap-3 border-b border-[#F1F5F9]">
                        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-[#0F172A] to-[#2563EB] flex items-center justify-center text-white shadow-md">
                            <span className="material-symbols-outlined text-[20px]">psychology</span>
                        </div>
                        <div className="flex flex-col">
                            <span className="font-bold text-[17px] tracking-tight text-[#0F172A]">Review Sense</span>
                            <span className="text-[11px] text-[#64748B] font-medium -mt-1">Emotion Intelligence</span>
                        </div>
                    </div>

                    {/* Navigation Items */}
                    <nav className="p-4 space-y-1.5">
                        <button
                            onClick={() => setActiveView('dashboard')}
                            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-semibold transition-all ${
                                activeView === 'dashboard'
                                    ? 'bg-[#2563EB] text-white shadow-md shadow-blue-500/20'
                                    : 'text-[#64748B] hover:bg-[#F1F5F9] hover:text-[#0F172A]'
                            }`}
                        >
                            <span className="material-symbols-outlined text-[20px]">dashboard</span>
                            <span>Dashboard</span>
                        </button>

                        <button
                            onClick={() => setActiveView('history')}
                            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-semibold transition-all ${
                                activeView === 'history'
                                    ? 'bg-[#2563EB] text-white shadow-md shadow-blue-500/20'
                                    : 'text-[#64748B] hover:bg-[#F1F5F9] hover:text-[#0F172A]'
                            }`}
                        >
                            <span className="material-symbols-outlined text-[20px]">history</span>
                            <span>History</span>
                            {history.length > 0 && (
                                <span className="ml-auto bg-slate-200 text-slate-700 text-[11px] px-2 py-0.5 rounded-full font-bold">
                                    {history.length}
                                </span>
                            )}
                        </button>

                        <button
                            onClick={() => setActiveView('batchJobs')}
                            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-semibold transition-all ${
                                activeView === 'batchJobs'
                                    ? 'bg-[#2563EB] text-white shadow-md shadow-blue-500/20'
                                    : 'text-[#64748B] hover:bg-[#F1F5F9] hover:text-[#0F172A]'
                            }`}
                        >
                            <span className="material-symbols-outlined text-[20px]">clinical_notes</span>
                            <span>Batch Jobs</span>
                            {batchJobs.length > 0 && (
                                <span className="ml-auto bg-blue-100 text-blue-800 text-[11px] px-2 py-0.5 rounded-full font-bold">
                                    {batchJobs.length}
                                </span>
                            )}
                        </button>

                        <button
                            onClick={() => setActiveView('settings')}
                            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-semibold transition-all ${
                                activeView === 'settings'
                                    ? 'bg-[#2563EB] text-white shadow-md shadow-blue-500/20'
                                    : 'text-[#64748B] hover:bg-[#F1F5F9] hover:text-[#0F172A]'
                            }`}
                        >
                            <span className="material-symbols-outlined text-[20px]">settings</span>
                            <span>Settings</span>
                        </button>
                    </nav>
                </div>

                {/* Profile Card */}
                <div className="p-4 border-t border-[#F1F5F9]">
                    <div className="flex items-center gap-3 bg-[#F8FAFC] p-2.5 rounded-2xl border border-[#E2E8F0]">
                        <div className="w-10 h-10 rounded-xl bg-[#0F172A] text-white flex items-center justify-center font-bold text-sm">
                            VR
                        </div>
                        <div className="flex flex-col min-w-0">
                            <p className="text-sm font-bold text-[#0F172A] truncate">Rohith V</p>
                            <span className="text-[11px] text-[#2563EB] font-semibold">Phase 6 Active</span>
                        </div>
                    </div>
                </div>
            </aside>

            {/* MAIN CONTENT AREA */}
            <div className="pl-64 flex-1 flex flex-col min-h-screen">
                {/* TOP HEADER */}
                <header className="sticky top-0 bg-white/85 backdrop-blur-md border-b border-[#E2E8F0] z-40 h-16 flex items-center justify-between px-8">
                    {/* Search Bar */}
                    <div className="flex items-center flex-1 max-w-md">
                        <div className="relative w-full">
                            <span className="material-symbols-outlined absolute left-3.5 top-1/2 -translate-y-1/2 text-[#94A3B8] text-[20px]">
                                search
                            </span>
                            <input
                                type="text"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                placeholder="Search analyzed reviews..."
                                className="w-full bg-[#F1F5F9] border-none rounded-full py-2 pl-10 pr-4 text-sm focus:ring-2 focus:ring-blue-500/30 outline-none text-[#0F172A] transition-all"
                            />
                        </div>
                    </div>

                    {/* Status Chips & Actions */}
                    <div className="flex items-center gap-3">
                        {/* Live Model Status Pill */}
                        <div
                            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold border ${
                                isHealthy
                                    ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                                    : 'bg-amber-50 text-amber-700 border-amber-200'
                            }`}
                        >
                            <span
                                className={`w-2 h-2 rounded-full ${
                                    isHealthy ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'
                                }`}
                            />
                            <span>
                                {isHealthy
                                    ? backendHealth.activeModel.includes('deberta')
                                        ? 'DeBERTa-v3 Active ⚡'
                                        : 'TF-IDF Baseline Active'
                                    : 'API Offline'}
                            </span>
                        </div>

                        {/* Hardware Indicator */}
                        {isHealthy && (
                            <span className="bg-slate-100 text-slate-700 text-xs px-2.5 py-1 rounded-full font-medium uppercase tracking-wide">
                                {backendHealth.device}
                            </span>
                        )}

                        {/* Latency badge */}
                        {currentResult && (
                            <span className="bg-blue-50 text-blue-700 text-xs px-3 py-1.5 rounded-full font-semibold border border-blue-200 flex items-center gap-1">
                                <span className="material-symbols-outlined text-[14px]">bolt</span>
                                {currentResult.latencyMs}ms
                            </span>
                        )}
                    </div>
                </header>

                {/* VIEW CONTAINER */}
                <main className="flex-1 p-8">
                    {/* ERROR BANNER */}
                    {error && (
                        <div className="mb-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-2xl flex items-center justify-between text-sm">
                            <div className="flex items-center gap-2">
                                <span className="material-symbols-outlined text-red-600">error</span>
                                <span>{error}</span>
                            </div>
                            <button onClick={() => setError(null)} className="text-red-500 hover:text-red-800 text-xs font-bold uppercase">
                                Dismiss
                            </button>
                        </div>
                    )}

                    {/* 1. DASHBOARD VIEW */}
                    {activeView === 'dashboard' && (
                        <div className="space-y-8">
                            {/* Title Bar */}
                            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
                                <div>
                                    <h1 className="text-3xl font-extrabold text-[#0F172A] tracking-tight">
                                        Emotion Analysis Dashboard
                                    </h1>
                                    <p className="text-[#64748B] text-base mt-1">
                                        Multi-label review emotion classification with calibrated decision thresholds.
                                    </p>
                                </div>

                                <div className="flex items-center gap-2">
                                    <button
                                        onClick={() => handleAnalyze()}
                                        disabled={isLoading}
                                        className="bg-[#0F172A] hover:bg-[#1E293B] text-white px-5 py-2 rounded-full font-semibold text-xs transition-all shadow-sm flex items-center gap-2"
                                    >
                                        <span className="material-symbols-outlined text-[16px]">refresh</span>
                                        Re-analyze
                                    </button>
                                </div>
                            </div>

                            {/* Split Layout: Left Input Panel, Right Results Panel */}
                            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                                {/* LEFT PANEL: INPUT */}
                                <div className="lg:col-span-5 space-y-6">
                                    <div className="bg-white rounded-3xl border border-[#E2E8F0] shadow-sm p-6 relative overflow-hidden">
                                        <div className="flex items-center justify-between mb-4">
                                            <h2 className="text-lg font-bold text-[#0F172A]">Input Review</h2>
                                            {/* Mode Switcher */}
                                            <div className="flex bg-[#F1F5F9] p-1 rounded-xl">
                                                <button
                                                    onClick={() => setInputMode('text')}
                                                    className={`px-3.5 py-1 text-xs font-semibold rounded-lg transition-all ${
                                                        inputMode === 'text'
                                                            ? 'bg-white text-[#0F172A] shadow-sm'
                                                            : 'text-[#64748B]'
                                                    }`}
                                                >
                                                    Text
                                                </button>
                                                <button
                                                    onClick={() => setInputMode('batch')}
                                                    className={`px-3.5 py-1 text-xs font-semibold rounded-lg transition-all ${
                                                        inputMode === 'batch'
                                                            ? 'bg-white text-[#0F172A] shadow-sm'
                                                            : 'text-[#64748B]'
                                                    }`}
                                                >
                                                    Upload CSV
                                                </button>
                                            </div>
                                        </div>

                                        {inputMode === 'text' ? (
                                            <div className="space-y-4">
                                                <label className="text-[11px] font-bold text-[#64748B] uppercase tracking-wider block">
                                                    Paste Review Text
                                                </label>
                                                <div className="relative">
                                                    <textarea
                                                        rows={6}
                                                        value={reviewText}
                                                        onChange={(e) => setReviewText(e.target.value)}
                                                        placeholder="Type or paste a product or service review..."
                                                        className="w-full bg-[#F8FAFC] border border-[#E2E8F0] text-[#0F172A] p-4 rounded-2xl text-sm leading-relaxed resize-none focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all placeholder:text-[#94A3B8]"
                                                    />
                                                </div>

                                                <div className="flex items-center justify-between pt-1">
                                                    <span className="text-xs text-[#94A3B8] font-medium">
                                                        {reviewText.length} / 2000 chars
                                                    </span>
                                                    <button
                                                        onClick={() => handleAnalyze()}
                                                        disabled={isLoading}
                                                        className="bg-[#2563EB] hover:bg-blue-700 text-white px-6 py-2.5 rounded-xl font-bold text-xs shadow-md shadow-blue-500/20 transition-all flex items-center gap-2 disabled:opacity-50"
                                                    >
                                                        {isLoading ? (
                                                            <>
                                                                <span className="material-symbols-outlined text-[16px] animate-spin">
                                                                    progress_activity
                                                                </span>
                                                                <span>Analyzing...</span>
                                                            </>
                                                        ) : (
                                                            <>
                                                                <span>Analyze Emotion</span>
                                                                <span className="material-symbols-outlined text-[16px]">
                                                                    arrow_forward
                                                                </span>
                                                            </>
                                                        )}
                                                    </button>
                                                </div>
                                            </div>
                                        ) : (
                                            /* BATCH CSV UPLOAD ZONE */
                                            <div className="space-y-4">
                                                <label className="text-[11px] font-bold text-[#64748B] uppercase tracking-wider block">
                                                    Bulk Review Upload
                                                </label>
                                                <label className="border-2 border-dashed border-[#CBD5E1] hover:border-blue-500 bg-[#F8FAFC] rounded-2xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all group">
                                                    <span className="material-symbols-outlined text-[44px] text-[#94A3B8] group-hover:text-blue-600 transition-colors">
                                                        cloud_upload
                                                    </span>
                                                    <p className="mt-2 text-sm font-bold text-[#0F172A]">Drop CSV file here</p>
                                                    <p className="text-xs text-[#64748B] mt-0.5">Supports .csv files up to 500 reviews</p>
                                                    <input
                                                        type="file"
                                                        accept=".csv,.txt"
                                                        onChange={handleFileUpload}
                                                        className="hidden"
                                                    />
                                                </label>
                                            </div>
                                        )}

                                        {/* Quick Test Presets */}
                                        <div className="mt-6 pt-5 border-t border-[#F1F5F9]">
                                            <p className="text-[11px] font-bold text-[#64748B] uppercase tracking-wider mb-2.5">
                                                Quick Test Presets
                                            </p>
                                            <div className="flex flex-wrap gap-1.5">
                                                {PRESET_REVIEWS.map((preset, index) => (
                                                    <button
                                                        key={index}
                                                        onClick={() => {
                                                            setReviewText(preset.text);
                                                            handleAnalyze(preset.text);
                                                        }}
                                                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-[#F1F5F9] text-[#475569] hover:bg-[#E2E8F0] hover:text-[#0F172A] transition-all"
                                                    >
                                                        <span className="material-symbols-outlined text-[15px]">{preset.icon}</span>
                                                        <span>{preset.title}</span>
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    </div>

                                    {/* Active Model Engine Card */}
                                    <div className="bg-white rounded-2xl border border-[#E2E8F0] p-4 flex items-center gap-4 shadow-sm">
                                        <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center font-bold">
                                            <span className="material-symbols-outlined text-[22px]">smart_toy</span>
                                        </div>
                                        <div>
                                            <p className="text-[11px] font-bold text-[#94A3B8] uppercase tracking-wider">
                                                Active Inference Engine
                                            </p>
                                            <p className="text-sm font-bold text-[#0F172A]">
                                                {getDisplayModelName(backendHealth.activeModel)}
                                            </p>
                                        </div>
                                    </div>
                                </div>

                                {/* RIGHT PANEL: RESULTS & PROBABILITIES */}
                                <div className="lg:col-span-7 space-y-6">
                                    {currentResult ? (
                                        <>
                                            {/* Primary Result Hero Card */}
                                            <div className="bg-white rounded-3xl border border-[#E2E8F0] shadow-md p-6 relative overflow-hidden">
                                                <div
                                                    className="absolute inset-0 opacity-15 pointer-events-none"
                                                    style={{ backgroundColor: EMOTION_COLORS[currentResult.primaryEmotion] }}
                                                />

                                                <div className="relative z-10 flex flex-col md:flex-row items-center gap-6">
                                                    {/* Circular Confidence Meter */}
                                                    <div className="relative w-32 h-32 shrink-0 flex items-center justify-center">
                                                        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                                                            <circle
                                                                cx="50"
                                                                cy="50"
                                                                r="42"
                                                                fill="transparent"
                                                                stroke="#E2E8F0"
                                                                strokeWidth="8"
                                                            />
                                                            <circle
                                                                cx="50"
                                                                cy="50"
                                                                r="42"
                                                                fill="transparent"
                                                                stroke={EMOTION_COLORS[currentResult.primaryEmotion]}
                                                                strokeWidth="8"
                                                                strokeDasharray={`${2 * Math.PI * 42}`}
                                                                strokeDashoffset={`${
                                                                    2 * Math.PI * 42 * (1 - currentResult.primaryScore)
                                                                }`}
                                                                strokeLinecap="round"
                                                                className="transition-all duration-1000 ease-out"
                                                            />
                                                        </svg>
                                                        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                                                            <span
                                                                className="text-2xl font-extrabold tracking-tight"
                                                                style={{ color: EMOTION_COLORS[currentResult.primaryEmotion] }}
                                                            >
                                                                {formatPercent(currentResult.primaryScore)}
                                                            </span>
                                                            <span className="text-[10px] text-[#64748B] font-bold uppercase tracking-wider">
                                                                Score
                                                            </span>
                                                        </div>
                                                    </div>

                                                    {/* Primary Emotion Meta */}
                                                    <div className="flex-1 text-center md:text-left">
                                                        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold mb-2 shadow-sm"
                                                            style={{
                                                                backgroundColor: EMOTION_BG_COLORS[currentResult.primaryEmotion],
                                                                color: EMOTION_COLORS[currentResult.primaryEmotion],
                                                            }}
                                                        >
                                                            <span className="material-symbols-outlined text-[15px]">
                                                                {EMOTION_ICONS[currentResult.primaryEmotion]}
                                                            </span>
                                                            <span className="uppercase tracking-wider">Primary Emotion</span>
                                                        </div>

                                                        <h3 className="text-3xl font-extrabold text-[#0F172A] capitalize">
                                                            {currentResult.primaryEmotion}
                                                        </h3>

                                                        <p className="text-sm text-[#64748B] mt-1.5 leading-relaxed">
                                                            {EMOTION_DESCRIPTIONS[currentResult.primaryEmotion]}
                                                        </p>

                                                        {/* Secondary Emotions Chips */}
                                                        {currentResult.secondaryEmotions.length > 0 && (
                                                            <div className="mt-4 pt-3 border-t border-[#F1F5F9]">
                                                                <span className="text-[11px] font-bold text-[#64748B] uppercase tracking-wider block mb-2">
                                                                    Secondary Emotions Detected
                                                                </span>
                                                                <div className="flex flex-wrap gap-2">
                                                                    {currentResult.secondaryEmotions.map((sec, i) => (
                                                                        <span
                                                                            key={i}
                                                                            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl text-xs font-semibold border"
                                                                            style={{
                                                                                backgroundColor: EMOTION_BG_COLORS[sec.label],
                                                                                color: EMOTION_COLORS[sec.label],
                                                                                borderColor: EMOTION_COLORS[sec.label] + '33',
                                                                            }}
                                                                        >
                                                                            <span className="material-symbols-outlined text-[14px]">
                                                                                {EMOTION_ICONS[sec.label]}
                                                                            </span>
                                                                            <span className="capitalize">{sec.label}</span>
                                                                            <span className="font-extrabold ml-1">
                                                                                {formatPercent(sec.score)}
                                                                            </span>
                                                                        </span>
                                                                    ))}
                                                                </div>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Probability Distribution Bars for all 8 emotions */}
                                            <div className="bg-white rounded-3xl border border-[#E2E8F0] shadow-sm p-6">
                                                <div className="flex items-center justify-between mb-4">
                                                    <div>
                                                        <h3 className="text-base font-bold text-[#0F172A]">
                                                            Emotion Probability Distribution
                                                        </h3>
                                                        <p className="text-xs text-[#64748B] mt-0.5">
                                                            Full 8-class scores with calibrated decision threshold markers
                                                        </p>
                                                    </div>
                                                    <span className="text-xs bg-slate-100 text-slate-600 px-2.5 py-1 rounded-lg font-bold">
                                                        8 Classes
                                                    </span>
                                                </div>

                                                <div className="space-y-3.5">
                                                    {EMOTIONS.map((emotion) => {
                                                        const score = currentResult.scores[emotion] ?? 0;
                                                        const thresh = thresholds[emotion] ?? 0.5;
                                                        const isPrimary = emotion === currentResult.primaryEmotion;

                                                        return (
                                                            <div key={emotion} className="space-y-1">
                                                                <div className="flex items-center justify-between text-xs">
                                                                    <div className="flex items-center gap-2">
                                                                        <span
                                                                            className="w-2.5 h-2.5 rounded-full"
                                                                            style={{ backgroundColor: EMOTION_COLORS[emotion] }}
                                                                        />
                                                                        <span className={`font-semibold capitalize ${isPrimary ? 'text-[#0F172A] font-extrabold' : 'text-[#475569]'}`}>
                                                                            {EMOTION_LABELS[emotion]}
                                                                        </span>
                                                                        {isPrimary && (
                                                                            <span className="text-[10px] bg-blue-100 text-blue-800 px-1.5 py-0.2 rounded font-bold uppercase">
                                                                                Primary
                                                                            </span>
                                                                        )}
                                                                    </div>
                                                                    <div className="flex items-center gap-3">
                                                                        <span className="text-[11px] text-[#94A3B8]">
                                                                            cutoff: {Math.round(thresh * 100)}%
                                                                        </span>
                                                                        <span className="font-extrabold text-[#0F172A] w-10 text-right">
                                                                            {formatPercent(score)}
                                                                        </span>
                                                                    </div>
                                                                </div>

                                                                {/* Progress Bar with Threshold Marker */}
                                                                <div className="relative w-full h-3 bg-[#F1F5F9] rounded-full overflow-hidden">
                                                                    <div
                                                                        className="h-full rounded-full transition-all duration-700 ease-out"
                                                                        style={{
                                                                            width: `${Math.min(100, Math.max(0, score * 100))}%`,
                                                                            backgroundColor: EMOTION_COLORS[emotion],
                                                                        }}
                                                                    />
                                                                    {/* Threshold indicator line */}
                                                                    <div
                                                                        className="absolute top-0 bottom-0 w-0.5 bg-[#0F172A]/40 z-10"
                                                                        style={{ left: `${thresh * 100}%` }}
                                                                        title={`Threshold: ${Math.round(thresh * 100)}%`}
                                                                    />
                                                                </div>
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                            </div>
                                        </>
                                    ) : (
                                        /* Empty State */
                                        <div className="bg-white rounded-3xl border border-[#E2E8F0] p-12 text-center flex flex-col items-center justify-center min-h-[400px]">
                                            <div className="w-16 h-16 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center mb-4">
                                                <span className="material-symbols-outlined text-[32px]">psychology</span>
                                            </div>
                                            <h3 className="text-xl font-bold text-[#0F172A]">No Review Analyzed Yet</h3>
                                            <p className="text-sm text-[#64748B] max-w-sm mt-1">
                                                Type a review or select one of the quick test presets to generate real-time emotion predictions.
                                            </p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* 2. HISTORY VIEW */}
                    {activeView === 'history' && (
                        <div className="space-y-6">
                            <div className="flex items-center justify-between">
                                <div>
                                    <h1 className="text-2xl font-extrabold text-[#0F172A]">Analysis History</h1>
                                    <p className="text-sm text-[#64748B]">
                                        Past predictions recorded in this session.
                                    </p>
                                </div>
                                <span className="bg-slate-200 text-slate-800 text-xs px-3 py-1 rounded-full font-bold">
                                    {filteredHistory.length} total entries
                                </span>
                            </div>

                            {filteredHistory.length === 0 ? (
                                <div className="bg-white rounded-3xl border border-[#E2E8F0] p-12 text-center text-[#64748B]">
                                    No history entries match your search.
                                </div>
                            ) : (
                                <div className="bg-white rounded-3xl border border-[#E2E8F0] shadow-sm overflow-hidden">
                                    <div className="divide-y divide-[#F1F5F9]">
                                        {filteredHistory.map((item, idx) => (
                                            <div
                                                key={item.id || idx}
                                                className="p-5 hover:bg-[#F8FAFC] transition-colors flex items-center justify-between gap-4"
                                            >
                                                <div className="flex-1 min-w-0">
                                                    <div className="flex items-center gap-2 mb-1">
                                                        <span
                                                            className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs font-bold"
                                                            style={{
                                                                backgroundColor: EMOTION_BG_COLORS[item.primaryEmotion],
                                                                color: EMOTION_COLORS[item.primaryEmotion],
                                                            }}
                                                        >
                                                            <span className="material-symbols-outlined text-[13px]">
                                                                {EMOTION_ICONS[item.primaryEmotion]}
                                                            </span>
                                                            <span className="capitalize">{item.primaryEmotion}</span>
                                                        </span>
                                                        <span className="text-xs font-extrabold text-[#0F172A]">
                                                            {formatPercent(item.primaryScore)}
                                                        </span>
                                                        <span className="text-xs text-[#94A3B8]">•</span>
                                                        <span className="text-xs text-[#94A3B8]">{item.timestamp}</span>
                                                        <span className="text-xs text-[#94A3B8]">•</span>
                                                        <span className="text-xs text-[#94A3B8]">{item.latencyMs}ms</span>
                                                    </div>
                                                    <p className="text-sm text-[#334155] font-medium truncate">
                                                        "{item.review}"
                                                    </p>
                                                </div>
                                                <button
                                                    onClick={() => {
                                                        setReviewText(item.review);
                                                        setCurrentResult(item);
                                                        setActiveView('dashboard');
                                                    }}
                                                    className="px-3.5 py-1.5 rounded-lg text-xs font-bold bg-[#F1F5F9] text-[#475569] hover:bg-[#E2E8F0] transition-colors shrink-0"
                                                >
                                                    View Details
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* 3. BATCH JOBS VIEW */}
                    {activeView === 'batchJobs' && (
                        <div className="space-y-6">
                            <div>
                                <h1 className="text-2xl font-extrabold text-[#0F172A]">Batch Jobs & Analytics</h1>
                                <p className="text-sm text-[#64748B]">
                                    Bulk dataset classification results and summary metrics.
                                </p>
                            </div>

                            {/* Metric Cards */}
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                <div className="bg-white rounded-2xl border border-[#E2E8F0] p-6 shadow-sm">
                                    <span className="text-xs font-bold text-[#64748B] uppercase tracking-wider">
                                        Total Bulk Reviews
                                    </span>
                                    <p className="text-3xl font-extrabold text-[#0F172A] mt-2">
                                        {batchJobs.reduce((acc, job) => acc + job.totalReviews, 0)}
                                    </p>
                                    <span className="text-xs text-emerald-600 font-semibold mt-1 inline-block">
                                        Across {batchJobs.length} job runs
                                    </span>
                                </div>

                                <div className="bg-white rounded-2xl border border-[#E2E8F0] p-6 shadow-sm">
                                    <span className="text-xs font-bold text-[#64748B] uppercase tracking-wider">
                                        Average Confidence
                                    </span>
                                    <p className="text-3xl font-extrabold text-[#0F172A] mt-2">
                                        {batchJobs.length > 0
                                            ? Math.round(
                                                  batchJobs.reduce((acc, job) => acc + job.avgConfidence, 0) /
                                                      batchJobs.length
                                              )
                                            : 0}
                                        %
                                    </p>
                                    <span className="text-xs text-blue-600 font-semibold mt-1 inline-block">
                                        Validation calibrated
                                    </span>
                                </div>

                                <div className="bg-white rounded-2xl border border-[#E2E8F0] p-6 shadow-sm">
                                    <span className="text-xs font-bold text-[#64748B] uppercase tracking-wider">
                                        Dominant Cluster
                                    </span>
                                    <p className="text-3xl font-extrabold text-[#0F172A] mt-2 capitalize">
                                        {batchJobs[0]?.dominantEmotion || 'None'}
                                    </p>
                                    <span className="text-xs text-purple-600 font-semibold mt-1 inline-block">
                                        Most frequent in latest batch
                                    </span>
                                </div>
                            </div>

                            {/* Job History Table */}
                            <div className="bg-white rounded-3xl border border-[#E2E8F0] shadow-sm overflow-hidden">
                                <div className="p-5 border-b border-[#F1F5F9] flex items-center justify-between">
                                    <h3 className="font-bold text-base text-[#0F172A]">Batch Logs</h3>
                                    <label className="bg-[#0F172A] text-white px-4 py-2 rounded-xl text-xs font-bold cursor-pointer hover:bg-[#1E293B] transition-all">
                                        + Upload New CSV
                                        <input
                                            type="file"
                                            accept=".csv,.txt"
                                            onChange={handleFileUpload}
                                            className="hidden"
                                        />
                                    </label>
                                </div>

                                {batchJobs.length === 0 ? (
                                    <div className="p-12 text-center text-[#64748B]">
                                        No batch jobs have been run yet. Upload a CSV file to see bulk analytics.
                                    </div>
                                ) : (
                                    <table className="w-full text-left text-sm">
                                        <thead className="bg-[#F8FAFC] text-[11px] font-bold text-[#64748B] uppercase tracking-wider border-b border-[#E2E8F0]">
                                            <tr>
                                                <th className="py-3.5 px-6">Filename</th>
                                                <th className="py-3.5 px-6">Reviews</th>
                                                <th className="py-3.5 px-6">Dominant Emotion</th>
                                                <th className="py-3.5 px-6">Avg Confidence</th>
                                                <th className="py-3.5 px-6">Latency</th>
                                                <th className="py-3.5 px-6">Status</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-[#F1F5F9]">
                                            {batchJobs.map((job) => (
                                                <tr key={job.id} className="hover:bg-[#F8FAFC] transition-colors">
                                                    <td className="py-4 px-6 font-bold text-[#0F172A]">{job.filename}</td>
                                                    <td className="py-4 px-6 font-medium text-[#475569]">{job.totalReviews}</td>
                                                    <td className="py-4 px-6">
                                                        <span
                                                            className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-bold capitalize"
                                                            style={{
                                                                backgroundColor: EMOTION_BG_COLORS[job.dominantEmotion],
                                                                color: EMOTION_COLORS[job.dominantEmotion],
                                                            }}
                                                        >
                                                            {job.dominantEmotion}
                                                        </span>
                                                    </td>
                                                    <td className="py-4 px-6 font-bold text-[#0F172A]">
                                                        {job.avgConfidence}%
                                                    </td>
                                                    <td className="py-4 px-6 text-[#64748B]">{job.latencyMs}ms</td>
                                                    <td className="py-4 px-6">
                                                        <span className="bg-emerald-100 text-emerald-800 text-xs px-2.5 py-0.5 rounded-full font-bold">
                                                            {job.status}
                                                        </span>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                )}
                            </div>
                        </div>
                    )}

                    {/* 4. SETTINGS VIEW */}
                    {activeView === 'settings' && (
                        <div className="space-y-8 max-w-3xl">
                            <div>
                                <h1 className="text-2xl font-extrabold text-[#0F172A]">System Settings</h1>
                                <p className="text-sm text-[#64748B]">
                                    Model configurations and calibrated emotion decision thresholds.
                                </p>
                            </div>

                            {/* Backend Health & Engine Details */}
                            <div className="bg-white rounded-3xl border border-[#E2E8F0] p-6 space-y-4 shadow-sm">
                                <h3 className="text-base font-bold text-[#0F172A]">Model Service Details</h3>
                                <div className="grid grid-cols-2 gap-4 text-sm">
                                    <div className="bg-[#F8FAFC] p-3.5 rounded-xl border border-[#E2E8F0]">
                                        <span className="text-xs text-[#64748B] font-semibold block">Active Model</span>
                                        <span className="font-bold text-[#0F172A] mt-1 block">
                                            {getDisplayModelName(backendHealth.activeModel)}
                                        </span>
                                    </div>
                                    <div className="bg-[#F8FAFC] p-3.5 rounded-xl border border-[#E2E8F0]">
                                        <span className="text-xs text-[#64748B] font-semibold block">Compute Device</span>
                                        <span className="font-bold text-[#0F172A] mt-1 uppercase block">
                                            {backendHealth.device}
                                        </span>
                                    </div>
                                    <div className="bg-[#F8FAFC] p-3.5 rounded-xl border border-[#E2E8F0]">
                                        <span className="text-xs text-[#64748B] font-semibold block">API Status</span>
                                        <span className="font-bold text-emerald-600 mt-1 block capitalize">
                                            {backendHealth.status}
                                        </span>
                                    </div>
                                    <div className="bg-[#F8FAFC] p-3.5 rounded-xl border border-[#E2E8F0]">
                                        <span className="text-xs text-[#64748B] font-semibold block">Service Version</span>
                                        <span className="font-bold text-[#0F172A] mt-1 block">
                                            v{backendHealth.version}
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* Calibrated Threshold Controls */}
                            <div className="bg-white rounded-3xl border border-[#E2E8F0] p-6 space-y-5 shadow-sm">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <h3 className="text-base font-bold text-[#0F172A]">Decision Thresholds</h3>
                                        <p className="text-xs text-[#64748B]">
                                            Optimized on GoEmotions validation split to maximize F1 score.
                                        </p>
                                    </div>
                                    <button
                                        onClick={() => setThresholds(DEFAULT_THRESHOLDS)}
                                        className="text-xs text-blue-600 hover:text-blue-800 font-bold"
                                    >
                                        Reset to Defaults
                                    </button>
                                </div>

                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                    {EMOTIONS.map((emotion) => {
                                        const val = thresholds[emotion];
                                        return (
                                            <div key={emotion} className="bg-[#F8FAFC] p-3.5 rounded-2xl border border-[#E2E8F0]">
                                                <div className="flex items-center justify-between text-xs mb-1.5">
                                                    <span className="font-bold capitalize text-[#0F172A]">
                                                        {EMOTION_LABELS[emotion]}
                                                    </span>
                                                    <span className="font-extrabold text-blue-600">
                                                        {Math.round(val * 100)}%
                                                    </span>
                                                </div>
                                                <input
                                                    type="range"
                                                    min="0.10"
                                                    max="0.90"
                                                    step="0.01"
                                                    value={val}
                                                    onChange={(e) =>
                                                        setThresholds((prev) => ({
                                                            ...prev,
                                                            [emotion]: parseFloat(e.target.value),
                                                        }))
                                                    }
                                                    className="w-full accent-blue-600"
                                                />
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
};

export default App;
