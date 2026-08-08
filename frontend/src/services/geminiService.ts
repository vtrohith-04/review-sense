
import { GoogleGenAI, Type } from "@google/genai";
import { EMOTIONS } from '../constants';
import type { EmotionScores } from '../types';

if (!process.env.API_KEY) {
    throw new Error("API_KEY environment variable not set");
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const properties = EMOTIONS.reduce((acc, emotion) => {
    acc[emotion] = {
        type: Type.NUMBER,
        description: `A score from 0 to 1 representing the confidence of the ${emotion} emotion.`,
    };
    return acc;
}, {} as Record<string, { type: Type; description: string }>);


const emotionSchema = {
    type: Type.OBJECT,
    properties,
};

export const analyzeEmotion = async (reviewText: string): Promise<EmotionScores> => {
    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: `Analyze the emotion of the following customer review. Return scores from 0 to 1 for these labels only: Happy, Sad, Angry, Frustrated, Surprised, Fearful, Disgusted, and Neutral. The review is: "${reviewText}"`,
            config: {
                responseMimeType: "application/json",
                responseSchema: emotionSchema,
            },
        });

        const jsonText = response.text.trim();
        const parsedJson = JSON.parse(jsonText);
        return parsedJson as EmotionScores;

    } catch (error) {
        console.error("Error analyzing emotion:", error);
        throw new Error("Failed to analyze review. Please check the console for details.");
    }
};
