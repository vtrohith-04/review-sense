
export enum Emotion {
    Neutral = "Neutral",
    Happy = "Happy",
    Angry = "Angry",
    Sad = "Sad",
    Frustrated = "Frustrated",
    Surprised = "Surprised",
    Fearful = "Fearful",
    Disgusted = "Disgusted"
}

export type EmotionScores = {
    [key in Emotion]: number;
};

export interface AnalysisResult {
    review: string;
    scores: EmotionScores;
}

export interface DashboardData {
    totalReviews: number;
    emotionCounts: { name: Emotion; value: number }[];
    overallSentiment: Emotion | 'Mixed';
}
