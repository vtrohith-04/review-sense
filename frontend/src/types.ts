export enum Emotion {
    Happy = "happy",
    Sad = "sad",
    Angry = "angry",
    Frustrated = "frustrated",
    Surprised = "surprised",
    Fearful = "fearful",
    Disgusted = "disgusted",
    Neutral = "neutral"
}

export type EmotionScores = Record<Emotion, number>;

export interface EmotionScore {
    label: Emotion;
    score: number;
    threshold?: number;
    exceedsThreshold?: boolean;
}

export interface AnalysisResult {
    id?: string;
    review: string;
    scores: EmotionScores;
    primaryEmotion: Emotion;
    primaryScore: number;
    topEmotions: EmotionScore[];
    secondaryEmotions: EmotionScore[];
    modelName: string;
    latencyMs: number;
    timestamp?: string;
}

export interface DashboardData {
    totalReviews: number;
    emotionCounts: { name: Emotion; value: number }[];
    overallSentiment: Emotion | "Mixed";
    averageConfidence: number;
    modelName: string;
    totalLatencyMs: number;
    results: AnalysisResult[];
}

export interface BackendHealth {
    status: 'ok' | 'degraded' | 'offline';
    activeModel: string;
    device: string;
    version: string;
}

export interface BatchJobRecord {
    id: string;
    filename: string;
    totalReviews: number;
    dominantEmotion: Emotion;
    avgConfidence: number;
    latencyMs: number;
    timestamp: string;
    status: 'completed' | 'processing' | 'failed';
    results: AnalysisResult[];
}
