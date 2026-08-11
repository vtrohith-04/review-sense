import { EMOTIONS } from '../constants';
import { Emotion } from '../types';
import type { AnalysisResult, EmotionScore, EmotionScores } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

interface EmotionApiResponse {
    primary_emotion: string;
    top_emotions: { label: string; score: number }[];
    model_name: string;
}

const emptyScores = (): EmotionScores =>
    EMOTIONS.reduce((scores, emotion) => {
        scores[emotion] = 0;
        return scores;
    }, {} as EmotionScores);

const toEmotion = (label: string): Emotion => {
    if (EMOTIONS.includes(label as Emotion)) {
        return label as Emotion;
    }

    return Emotion.Neutral;
};

export const analyzeEmotion = async (reviewText: string): Promise<AnalysisResult> => {
    const response = await fetch(`${API_BASE_URL}/api/v1/predict/emotion`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: reviewText,
            top_k: 3,
            threshold: 0.35,
        }),
    });

    if (!response.ok) {
        const message = response.status === 503
            ? 'The backend model is not available yet. Train the model, then restart the API.'
            : 'Failed to analyze review. Please make sure the backend API is running.';
        throw new Error(message);
    }

    const payload = await response.json() as EmotionApiResponse;
    const scores = emptyScores();
    const topEmotions: EmotionScore[] = payload.top_emotions.map((emotion) => {
        const label = toEmotion(emotion.label);
        scores[label] = emotion.score;
        return { label, score: emotion.score };
    });

    const primaryEmotion = toEmotion(payload.primary_emotion);

    return {
        review: reviewText,
        scores,
        primaryEmotion,
        topEmotions,
        modelName: payload.model_name,
    };
};
