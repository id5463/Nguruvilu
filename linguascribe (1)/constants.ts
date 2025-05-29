
import type { Language } from './types';

export const SUPPORTED_LANGUAGES: Language[] = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'ru', name: 'Russian' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'zh-CN', name: 'Chinese (Simplified)' },
  { code: 'ar', name: 'Arabic' },
  { code: 'hi', name: 'Hindi' },
];

export const DEFAULT_SOURCE_LANG: string = 'en';
export const DEFAULT_TARGET_LANG: string = 'es';
export const DEFAULT_UI_LANG: string = 'en';

export const SUPPORTED_UI_LANGS: Language[] = [
  { code: 'en', name: 'English (UI)' },
  { code: 'zh-CN', name: '简体中文 (界面)' },
];

export const NOTE_CHUNK_WORD_THRESHOLD: number = 500; 

// Debounce delays for translation modes
// STAGE1_TRANSLATION_DEBOUNCE_DELAY has been removed as Stage 1 is deprecated.
export const REALTIME_TRANSLATION_DEBOUNCE_DELAY: number = 150; // ms for speech-like immediacy (Stage 2 in realtime mode)
export const COMPOSED_TRANSLATION_DEBOUNCE_DELAY: number = 1200; // ms for typing longer texts (Stage 2 in composed mode)

// localStorage keys
export const LOCAL_STORAGE_KEYS = {
  SOURCE_TEXT: 'linguaScribe_sourceText',
  TRANSLATED_TEXT: 'linguaScribe_translatedText',
  NOTES: 'linguaScribe_notes',
  SOURCE_LANG: 'linguaScribe_sourceLang',
  TARGET_LANG: 'linguaScribe_targetLang',
  UI_LANG: 'linguaScribe_uiLang',
  TRANSLATION_MODE: 'linguaScribe_translationMode',
  HISTORY_ITEMS: 'linguaScribe_historyItems', // New key for session history
};

export type TranslationMode = 'realtime' | 'composed';
export const DEFAULT_TRANSLATION_MODE: TranslationMode = 'realtime';
