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
    [Emotion.Happy]: '#D97706',       // Amber / Gold
    [Emotion.Sad]: '#0284C7',         // Sky Blue
    [Emotion.Angry]: '#EF4444',       // Crimson Red
    [Emotion.Frustrated]: '#F97316',  // Burnt Orange
    [Emotion.Surprised]: '#8B5CF6',   // Purple / Violet
    [Emotion.Fearful]: '#6366F1',     // Deep Indigo
    [Emotion.Disgusted]: '#10B981',   // Emerald / Lime
    [Emotion.Neutral]: '#64748B'      // Slate Gray
};

export const EMOTION_BG_COLORS: Record<Emotion, string> = {
    [Emotion.Happy]: '#FEF3C7',
    [Emotion.Sad]: '#E0F2FE',
    [Emotion.Angry]: '#FEE2E2',
    [Emotion.Frustrated]: '#FFEDD5',
    [Emotion.Surprised]: '#EDE9FE',
    [Emotion.Fearful]: '#E0E7FF',
    [Emotion.Disgusted]: '#D1FAE5',
    [Emotion.Neutral]: '#F1F5F9'
};

export const EMOTION_ICONS: Record<Emotion, string> = {
    [Emotion.Happy]: 'sentiment_very_satisfied',
    [Emotion.Sad]: 'sentiment_dissatisfied',
    [Emotion.Angry]: 'sentiment_very_dissatisfied',
    [Emotion.Frustrated]: 'mood_bad',
    [Emotion.Surprised]: 'emergency',
    [Emotion.Fearful]: 'warning',
    [Emotion.Disgusted]: 'thumb_down',
    [Emotion.Neutral]: 'sentiment_neutral'
};

export const EMOTION_DESCRIPTIONS: Record<Emotion, string> = {
    [Emotion.Happy]: 'The review expresses satisfaction, delight, praise, or gratitude.',
    [Emotion.Sad]: 'The review conveys disappointment, regret, or unmet expectations.',
    [Emotion.Angry]: 'The review expresses strong indignation, outrage, or customer dissatisfaction.',
    [Emotion.Frustrated]: 'The review reflects annoyance, delivery confusion, or friction.',
    [Emotion.Surprised]: 'The review notes unexpected outcomes, astonishment, or sudden realization.',
    [Emotion.Fearful]: 'The review expresses anxiety, security concerns, or hesitation.',
    [Emotion.Disgusted]: 'The review describes physical or aesthetic revulsion and severe defect.',
    [Emotion.Neutral]: 'The review states factual, objective, or transactional feedback without strong emotion.'
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

export const DEFAULT_THRESHOLDS: Record<Emotion, number> = {
    [Emotion.Happy]: 0.47,
    [Emotion.Sad]: 0.62,
    [Emotion.Angry]: 0.29,
    [Emotion.Frustrated]: 0.33,
    [Emotion.Surprised]: 0.36,
    [Emotion.Fearful]: 0.22,
    [Emotion.Disgusted]: 0.39,
    [Emotion.Neutral]: 0.22
};

export interface PresetReview {
    title: string;
    text: string;
    icon: string;
}

export const PRESET_REVIEWS: PresetReview[] = [
    {
        title: 'Damaged & Delayed',
        icon: 'local_shipping',
        text: 'The package arrived two weeks late, completely crushed and broken! Support refused a refund and was extremely rude.'
    },
    {
        title: 'Delighted Customer',
        icon: 'thumb_up',
        text: 'Honestly blown away by the quality! Fast shipping, excellent packaging, and the product works flawlessly.'
    },
    {
        title: 'Confused & Annoyed',
        icon: 'help',
        text: 'The instructions in the box make no sense at all and contradicted the website. Wasted two hours trying to set this up.'
    },
    {
        title: 'Disappointed Expectation',
        icon: 'sentiment_dissatisfied',
        text: 'I really wanted to love this, but the materials feel flimsy and cheap compared to what was advertised in the photos.'
    },
    {
        title: 'Factual Delivery',
        icon: 'inventory_2',
        text: 'Order received on Wednesday. Package contained the standard blue model with charger cord as ordered.'
    }
];
