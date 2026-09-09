import { EMOTIONS } from '../constants';
import { Emotion } from '../types';
import type { AnalysisResult, BackendHealth, EmotionScore, EmotionScores } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

interface EmotionApiResponse {
    text: string;
    primary_emotion: string;
    primary_score: number;
    secondary_emotions: { label: string; score: number; threshold: number; exceeds_threshold: boolean }[];
    all_scores: Record<string, number>;
    model_used: string;
    latency_ms: number;
}

interface BatchApiResponse {
    total_reviews: number;
    results: EmotionApiResponse[];
    total_latency_ms: number;
}

const emptyScores = (): EmotionScores =>
    EMOTIONS.reduce((scores, emotion) => {
        scores[emotion] = 0;
        return scores;
    }, {} as EmotionScores);

const toEmotion = (label: string): Emotion => {
    const clean = label.toLowerCase().trim();
    if (EMOTIONS.includes(clean as Emotion)) {
        return clean as Emotion;
    }
    return Emotion.Neutral;
};

const handleError = async (response: Response, fallback: string) => {
    if (response.ok) return;
    let detail = '';
    try {
        const body = await response.json() as { detail?: string };
        detail = body.detail ?? '';
    } catch {
        // Keep the friendly fallback when the API does not return JSON.
    }
    throw new Error(response.status === 503
        ? 'The backend model is not available yet. Please ensure the API is running with trained weights.'
        : detail || fallback);
};

const mapPrediction = (payload: EmotionApiResponse, fallbackText: string): AnalysisResult => {
    const scores = emptyScores();
    Object.entries(payload.all_scores).forEach(([label, score]) => {
        scores[toEmotion(label)] = score;
    });
    const secondary = (payload.secondary_emotions || []).map((emotion) => ({
        label: toEmotion(emotion.label),
        score: emotion.score,
        threshold: emotion.threshold,
        exceedsThreshold: emotion.exceeds_threshold,
    }));
    const primary = toEmotion(payload.primary_emotion);
    const topEmotions: EmotionScore[] = Object.entries(scores)
        .map(([label, score]) => ({ label: label as Emotion, score }))
        .sort((a, b) => b.score - a.score);

    return {
        id: Math.random().toString(36).substring(2, 9),
        review: payload.text || fallbackText,
        scores,
        primaryEmotion: primary,
        primaryScore: payload.primary_score,
        topEmotions,
        secondaryEmotions: secondary,
        modelName: payload.model_used,
        latencyMs: payload.latency_ms,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
};

export const getBackendHealth = async (): Promise<BackendHealth> => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            return { status: 'offline', activeModel: 'none', device: 'none', version: '1.0.0' };
        }
        const data = await response.json() as { status: string; active_model: string; device: string; version: string };
        return {
            status: data.status === 'ok' ? 'ok' : 'degraded',
            activeModel: data.active_model,
            device: data.device,
            version: data.version,
        };
    } catch {
        return { status: 'offline', activeModel: 'offline', device: 'none', version: '1.0.0' };
    }
};

export const analyzeEmotion = async (
    reviewText: string,
    thresholdOverride?: Record<string, number>,
): Promise<AnalysisResult> => {
    const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: reviewText,
            top_k: 3,
            threshold_override: thresholdOverride,
        }),
    });

    await handleError(response, 'Failed to analyze review. Please make sure the backend API is running.');
    const payload = await response.json() as EmotionApiResponse;
    return mapPrediction(payload, reviewText);
};

export const analyzeBatch = async (reviews: string[]): Promise<{
    results: AnalysisResult[];
    totalLatencyMs: number;
}> => {
    const response = await fetch(`${API_BASE_URL}/api/v1/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: reviews, top_k: 3 }),
    });
    await handleError(response, 'Failed to process the batch. Please make sure the backend API is running.');
    const payload = await response.json() as BatchApiResponse;
    return {
        results: payload.results.map((result, index) => mapPrediction(result, reviews[index] ?? '')),
        totalLatencyMs: payload.total_latency_ms,
    };
};
