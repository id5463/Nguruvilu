
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  console.error("API_KEY environment variable not set. Gemini API calls will fail.");
}

const ai = new GoogleGenAI({ apiKey: API_KEY! }); 

const translationModel = 'gemini-2.5-flash-preview-04-17';
const summarizationModel = 'gemini-2.5-flash-preview-04-17';
const titleGenerationModel = 'gemini-2.5-flash-preview-04-17'; // Can use the same flash model

// translateInterimSnippet function has been removed as Stage 1 translation is deprecated.

export async function translateText(text: string, sourceLangName: string, targetLangName: string): Promise<string> {
  if (!API_KEY) return Promise.reject(new Error("API Key not configured."));
  if (!text.trim()) return Promise.resolve("");

  const prompt = `Translate the following text accurately from ${sourceLangName} to ${targetLangName}. The text might be a single sentence or a larger segment of a conversation. Respond ONLY with the translated text, without any additional explanations, conversational phrases, or markdown formatting.

Text to translate:
"${text}"`;

  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model: translationModel,
      contents: prompt,
    });
    return response.text.trim();
  } catch (error) {
    console.error("Gemini translation error:", error);
    throw new Error(`Translation API request failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}

export async function summarizeSegmentForNote(segment: string, previousNoteSummary: string | null): Promise<string> {
  if (!API_KEY) return Promise.reject(new Error("API Key not configured."));

  const systemInstruction = `You are an expert note-taking AI. Your task is to create a concise, well-structured summary or set of notes for the "Current Text Segment". If "Previous Note Context" is provided, ensure your notes for the current segment flow logically and build upon that context. The notes should capture the key information and main ideas of the current segment. Output should be a clear summary, suitable for a note. Do not include any conversational preamble or markdown formatting unless it's part of the note content itself.`;
  
  let prompt = "";
  if (previousNoteSummary) {
    prompt += `Previous Note Context (summary of prior text):
\`\`\`
${previousNoteSummary}
\`\`\`

`;
  } else {
    prompt += `This is the first segment, so there is no previous note context.

`;
  }

  prompt += `Current Text Segment to summarize:
\`\`\`
${segment}
\`\`\`

Generate notes for the "Current Text Segment" only.`;

  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model: summarizationModel,
      contents: prompt,
      config: {
        systemInstruction: systemInstruction,
      }
    });
    return response.text.trim();
  } catch (error) {
    console.error("Gemini summarization error:", error);
    throw new Error(`Summarization API request failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}

export async function generateTitleForSession(textSnippet: string): Promise<string> {
  if (!API_KEY) return Promise.reject(new Error("API Key not configured."));
  if (!textSnippet.trim()) return Promise.resolve("Untitled Session");

  const prompt = `Based on the following text snippet, generate a very short and descriptive title (max 5-7 words). The title should be suitable for identifying a saved work session. Respond ONLY with the title itself, without any extra phrases or markdown.

Text snippet:
"${textSnippet.substring(0, 500)}${textSnippet.length > 500 ? '...' : ''}"`;

  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model: titleGenerationModel,
      contents: prompt,
       config: {
        // Optimize for speed for title generation, less critical for deep understanding
        thinkingConfig: { thinkingBudget: 0 } 
      }
    });
    let title = response.text.trim();
    // Remove potential surrounding quotes from the title
    title = title.replace(/^["']|["']$/g, '');
    return title || "Untitled Session"; // Fallback if AI returns empty string
  } catch (error) {
    console.error("Gemini title generation error:", error);
    // Don't throw, return a default title so saving doesn't fail completely
    return "Untitled Session (AI Error)";
  }
}
