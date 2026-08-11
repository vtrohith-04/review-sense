
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
}

export interface AnalysisResult {
    review: string;
    scores: EmotionScores;
    primaryEmotion: Emotion;
    topEmotions: EmotionScore[];
    modelName: string;
}

export interface DashboardData {
    totalReviews: number;
    emotionCounts: { name: Emotion; value: number }[];
    overallSentiment: Emotion | "Mixed";
    averageConfidence: number;
}
