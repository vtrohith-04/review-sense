
import { Emotion } from './types';

export const EMOTIONS: Emotion[] = [
    Emotion.Happy,
    Emotion.Sad,
    Emotion.Angry,
    Emotion.Frustrated,
    Emotion.Surprised,
    Emotion.Fearful,
    Emotion.Disgusted,
    Emotion.Neutral
];

export const EMOTION_COLORS: Record<Emotion, string> = {
    [Emotion.Happy]: '#f59e0b',
    [Emotion.Sad]: '#3b82f6',
    [Emotion.Angry]: '#ef4444',
    [Emotion.Frustrated]: '#f97316',
    [Emotion.Surprised]: '#8b5cf6',
    [Emotion.Fearful]: '#6366f1',
    [Emotion.Disgusted]: '#10b981',
    [Emotion.Neutral]: '#64748b'
};

export const EMOTION_LABELS: Record<Emotion, string> = {
    [Emotion.Happy]: 'Happy',
    [Emotion.Sad]: 'Sad',
    [Emotion.Angry]: 'Angry',
    [Emotion.Frustrated]: 'Frustrated',
    [Emotion.Surprised]: 'Surprised',
    [Emotion.Fearful]: 'Fearful',
    [Emotion.Disgusted]: 'Disgusted',
    [Emotion.Neutral]: 'Neutral'
};
