import { EMOTIONS } from '../constants';
import { Emotion } from '../types';
import type { AnalysisResult, BackendHealth, CredibilityResult, EmotionScore, EmotionScores } from '../types';

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

interface CredibilityApiResponse {
    text: string;
    is_fake: boolean;
    credibility_score: number;
    fake_probability: number;
    risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
    flagged_signals: string[];
    latency_ms: number;
}

interface ComprehensiveApiResponse {
    text: string;
    emotion: EmotionApiResponse;
    credibility: CredibilityApiResponse;
    total_latency_ms: number;
}

interface BatchApiResponse {
    total_reviews: number;
    results: EmotionApiResponse[];
    total_latency_ms: number;
}

interface BatchCredibilityApiResponse {
    total_reviews: number;
    results: CredibilityApiResponse[];
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

const mapCredibility = (payload: CredibilityApiResponse): CredibilityResult => ({
    isFake: payload.is_fake,
    credibilityScore: payload.credibility_score,
    fakeProbability: payload.fake_probability,
    riskLevel: payload.risk_level,
    flaggedSignals: payload.flagged_signals,
    latencyMs: payload.latency_ms,
});

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

const mapPrediction = (
    payload: EmotionApiResponse,
    fallbackText: string,
    credibilityPayload?: CredibilityApiResponse,
): AnalysisResult => {
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
        credibility: credibilityPayload ? mapCredibility(credibilityPayload) : undefined,
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
    try {
        // Try unified comprehensive analysis endpoint first (returns both emotion and credibility)
        const response = await fetch(`${API_BASE_URL}/api/v1/analyze/comprehensive`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: reviewText, top_k: 3 }),
        });

        if (response.ok) {
            const data = await response.json() as ComprehensiveApiResponse;
            return mapPrediction(data.emotion, reviewText, data.credibility);
        }
    } catch {
        // Fallback to emotion-only endpoint if comprehensive endpoint fails
    }

    const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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

export const analyzeCredibility = async (reviewText: string): Promise<CredibilityResult> => {
    const response = await fetch(`${API_BASE_URL}/api/v1/analyze/credibility`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: reviewText }),
    });

    await handleError(response, 'Failed to analyze review credibility.');
    const payload = await response.json() as CredibilityApiResponse;
    return mapCredibility(payload);
};

export const analyzeBatch = async (reviews: string[]): Promise<{
    results: AnalysisResult[];
    totalLatencyMs: number;
}> => {
    // 1. Fetch batch emotion predictions
    const response = await fetch(`${API_BASE_URL}/api/v1/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: reviews, top_k: 3 }),
    });
    await handleError(response, 'Failed to process the batch. Please make sure the backend API is running.');
    const payload = await response.json() as BatchApiResponse;

    // 2. Fetch batch credibility predictions if available
    let credibilityResults: CredibilityApiResponse[] = [];
    try {
        const credResponse = await fetch(`${API_BASE_URL}/api/v1/analyze/credibility/batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts: reviews }),
        });
        if (credResponse.ok) {
            const credData = await credResponse.json() as BatchCredibilityApiResponse;
            credibilityResults = credData.results;
        }
    } catch {
        // Credibility batch is optional enhancement
    }

    return {
        results: payload.results.map((result, index) =>
            mapPrediction(result, reviews[index] ?? '', credibilityResults[index]),
        ),
        totalLatencyMs: payload.total_latency_ms,
    };
};

