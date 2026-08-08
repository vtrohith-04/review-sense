
import { Emotion } from './types';

export const EMOTIONS: Emotion[] = [
    Emotion.Neutral,
    Emotion.Happy,
    Emotion.Angry,
    Emotion.Sad,
    Emotion.Frustrated,
    Emotion.Surprised,
    Emotion.Fearful,
    Emotion.Disgusted
];

export const EMOTION_COLORS: Record<Emotion, string> = {
    [Emotion.Neutral]: '#6b7280',  // gray-500
    [Emotion.Happy]: '#22c55e',    // green-500
    [Emotion.Angry]: '#dc2626',     // red-600
    [Emotion.Sad]: '#3b82f6',      // blue-500
    [Emotion.Frustrated]: '#d97706', // amber-600
    [Emotion.Surprised]: '#f97316', // orange-500
    [Emotion.Fearful]: '#7c3aed',   // violet-600
    [Emotion.Disgusted]: '#64748b'  // slate-500
};
