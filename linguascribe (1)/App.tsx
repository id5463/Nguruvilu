
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { SourceInputPanel } from './components/SourceInputPanel';
import { TranslationOutputPanel } from './components/TranslationOutputPanel';
import { NotesPanel } from './components/NotesPanel';
import { LanguageSelector } from './components/LanguageSelector';
import { HistoryPanel } from './components/HistoryPanel'; // New
import { translateText, summarizeSegmentForNote, generateTitleForSession } from './services/geminiService'; // Updated
import type { Note, Language, HistoryItem } from './types'; // Updated
import { 
  SUPPORTED_LANGUAGES, 
  DEFAULT_SOURCE_LANG, 
  DEFAULT_TARGET_LANG, 
  LOCAL_STORAGE_KEYS,
  SUPPORTED_UI_LANGS,
  REALTIME_TRANSLATION_DEBOUNCE_DELAY,
  COMPOSED_TRANSLATION_DEBOUNCE_DELAY,
  DEFAULT_TRANSLATION_MODE,
  type TranslationMode
} from './constants';
import { countWords, debounce } from './utils/textUtils';
import { useLocalization } from './contexts/LocalizationContext';
import { LoadingSpinner } from './components/LoadingSpinner'; // For saving session status

const App: React.FC = () => {
  const { t, language: uiLang, setLanguage: setUiLang, isLoadingLocales } = useLocalization();

  const [sourceText, setSourceText] = useState<string>(() => localStorage.getItem(LOCAL_STORAGE_KEYS.SOURCE_TEXT) || '');
  const [translatedFullText, setTranslatedFullText] = useState<string>(() => localStorage.getItem(LOCAL_STORAGE_KEYS.TRANSLATED_TEXT) || '');
  const [notes, setNotes] = useState<Note[]>(() => {
    const storedNotes = localStorage.getItem(LOCAL_STORAGE_KEYS.NOTES);
    return storedNotes ? JSON.parse(storedNotes) : [];
  });
  
  const [sourceLang, setSourceLang] = useState<string>(() => localStorage.getItem(LOCAL_STORAGE_KEYS.SOURCE_LANG) || DEFAULT_SOURCE_LANG);
  const [targetLang, setTargetLang] = useState<string>(() => localStorage.getItem(LOCAL_STORAGE_KEYS.TARGET_LANG) || DEFAULT_TARGET_LANG);
  const [translationMode, setTranslationMode] = useState<TranslationMode>(() => {
    const storedMode = localStorage.getItem(LOCAL_STORAGE_KEYS.TRANSLATION_MODE) as TranslationMode | null;
    return storedMode && ['realtime', 'composed'].includes(storedMode) ? storedMode : DEFAULT_TRANSLATION_MODE;
  });

  const [isTranslating, setIsTranslating] = useState<boolean>(false);
  const [isGeneratingNote, setIsGeneratingNote] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [currentInterimSTT, setCurrentInterimSTT] = useState<string>('');

  // History State
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>(() => {
    const storedHistory = localStorage.getItem(LOCAL_STORAGE_KEYS.HISTORY_ITEMS);
    try {
      return storedHistory ? JSON.parse(storedHistory) : [];
    } catch (e) {
      console.error(t('errorLoadingHistory'), e);
      return [];
    }
  });
  const [isHistoryPanelOpen, setIsHistoryPanelOpen] = useState<boolean>(false);
  const [historyMessage, setHistoryMessage] = useState<{type: 'success' | 'error', text: string} | null>(null);
  const [isSavingSession, setIsSavingSession] = useState<boolean>(false);


  const noteGenerationBufferRef = useRef<string>('');
  const processedTextLengthForNotesRef = useRef<number>(0);

  // Persist state to localStorage
  useEffect(() => { localStorage.setItem(LOCAL_STORAGE_KEYS.SOURCE_TEXT, sourceText); }, [sourceText]);
  useEffect(() => { localStorage.setItem(LOCAL_STORAGE_KEYS.TRANSLATED_TEXT, translatedFullText); }, [translatedFullText]);
  useEffect(() => { localStorage.setItem(LOCAL_STORAGE_KEYS.NOTES, JSON.stringify(notes)); }, [notes]);
  useEffect(() => { localStorage.setItem(LOCAL_STORAGE_KEYS.SOURCE_LANG, sourceLang); }, [sourceLang]);
  useEffect(() => { localStorage.setItem(LOCAL_STORAGE_KEYS.TARGET_LANG, targetLang); }, [targetLang]);
  useEffect(() => { localStorage.setItem(LOCAL_STORAGE_KEYS.TRANSLATION_MODE, translationMode); }, [translationMode]);
  useEffect(() => { localStorage.setItem(LOCAL_STORAGE_KEYS.HISTORY_ITEMS, JSON.stringify(historyItems));}, [historyItems]);


  const initializeNoteBufferAndProcessedLength = useCallback((currentTranslatedText: string) => {
    noteGenerationBufferRef.current = currentTranslatedText.trim();
    processedTextLengthForNotesRef.current = currentTranslatedText.length;
    console.log(`[NotesDebug] Note buffer initialized. Buffer words: ${countWords(noteGenerationBufferRef.current)}, processedLength: ${processedTextLengthForNotesRef.current}`);
  }, []);
  
  // Initialize note generation buffer refs on component mount or when loaded from history
  useEffect(() => {
    initializeNoteBufferAndProcessedLength(translatedFullText);
  }, [translatedFullText, initializeNoteBufferAndProcessedLength]);


  const getLanguageName = useCallback((code: string) => {
    return SUPPORTED_LANGUAGES.find(lang => lang.code === code)?.name || code;
  }, []);

  const currentDebounceDelay = translationMode === 'realtime' 
    ? REALTIME_TRANSLATION_DEBOUNCE_DELAY 
    : COMPOSED_TRANSLATION_DEBOUNCE_DELAY;

  const debouncedTranslate = useCallback(
    debounce(async (currentSourceText: string, currentSourceLang: string, currentTargetLang: string) => {
      if (!currentSourceText.trim()) {
        setTranslatedFullText('');
        setError(null);
        return;
      }
      setIsTranslating(true);
      setError(null);
      try {
        const translation = await translateText(
          currentSourceText, 
          getLanguageName(currentSourceLang), 
          getLanguageName(currentTargetLang)
        );
        setTranslatedFullText(translation);
      } catch (err) {
        console.error("Translation error:", err);
        setError(err instanceof Error ? err.message : "Translation failed.");
        setTranslatedFullText(prev => prev); 
      } finally {
        setIsTranslating(false);
      }
    }, currentDebounceDelay), 
    [getLanguageName, currentDebounceDelay] 
  );

  useEffect(() => {
    if (sourceText.trim()) {
      debouncedTranslate(sourceText, sourceLang, targetLang);
    } else {
      setTranslatedFullText('');
    }
  }, [sourceText, sourceLang, targetLang, debouncedTranslate]);

  useEffect(() => {
    console.log("[NotesDebug] translatedFullText effect for buffer accumulation.");
    console.log(`[NotesDebug] Before update: Buffer words: ${countWords(noteGenerationBufferRef.current)}, processedLength: ${processedTextLengthForNotesRef.current}, translatedFullText length: ${translatedFullText.length}`);

    if (translatedFullText.length > processedTextLengthForNotesRef.current) {
      const newTextForNotes = translatedFullText.substring(processedTextLengthForNotesRef.current);
      if (newTextForNotes.trim().length > 0) {
        const oldBufferContent = noteGenerationBufferRef.current;
        const space = (oldBufferContent.length > 0 && !oldBufferContent.endsWith(' ') && !newTextForNotes.startsWith(' ')) ? " " : "";
        noteGenerationBufferRef.current += space + newTextForNotes.trim();
      }
    } else if (translatedFullText.length < processedTextLengthForNotesRef.current) {
      noteGenerationBufferRef.current = translatedFullText.trim();
    }
    processedTextLengthForNotesRef.current = translatedFullText.length;
    console.log(`[NotesDebug] After update: Buffer words: ${countWords(noteGenerationBufferRef.current)}, processedLength: ${processedTextLengthForNotesRef.current}`);
  }, [translatedFullText]);


  const handleGenerateNoteManually = useCallback(async () => {
    const currentBufferContent = noteGenerationBufferRef.current;
    if (isGeneratingNote || !currentBufferContent.trim()) {
      if (!currentBufferContent.trim()) setError(t('noteBufferEmptyError'));
      return;
    }
    setIsGeneratingNote(true);
    setError(null); 
    const segmentToProcess = currentBufferContent;
    try {
      const previousNoteSummary = notes.length > 0 ? notes[notes.length - 1].summary : null;
      const noteSummary = await summarizeSegmentForNote(segmentToProcess, previousNoteSummary);
      setNotes(prevNotes => [
        ...prevNotes,
        { id: Date.now().toString(), timestamp: new Date().toISOString(), sourceSegment: segmentToProcess, summary: noteSummary },
      ]);
      noteGenerationBufferRef.current = "";
    } catch (err) {
      console.error("[NotesDebug] Manual Note generation API error:", err);
      setError(err instanceof Error ? err.message : t('errorGeneratingNote'));
    } finally {
      setIsGeneratingNote(false);
    }
  }, [isGeneratingNote, notes, t ]);

  const handleSourceTextChange = (text: string) => setSourceText(text);
  
  const handleClearAll = () => {
    setSourceText('');
    setTranslatedFullText(''); 
    setNotes([]);
    setCurrentInterimSTT(''); 
    setError(null);
    setHistoryMessage(null);
    // Note buffer and processed length will be reset by useEffect on translatedFullText
  };

  const toggleTranslationMode = () => setTranslationMode(prevMode => prevMode === 'realtime' ? 'composed' : 'realtime');
  const handleInterimResult = useCallback((interimText: string) => setCurrentInterimSTT(interimText), []);
  const handleFinalChunkResult = useCallback((finalChunkWithSpace: string) => {
    setSourceText(prevText => prevText + finalChunkWithSpace);
    setCurrentInterimSTT('');
  }, []);

  // --- History Functions ---
  const handleSaveCurrentSession = async () => {
    setIsSavingSession(true);
    setHistoryMessage(null);
    setError(null);
    const textForTitle = sourceText.trim() || translatedFullText.trim() || "New Session";
    let title = t('generatingTitleStatus');
    try {
      title = await generateTitleForSession(textForTitle);
    } catch (e) {
      console.error("Error generating title:", e);
      title = "Untitled Session (AI Error)";
      setHistoryMessage({type: 'error', text: t('errorGeneratingTitle')});
    }

    const newHistoryItem: HistoryItem = {
      id: Date.now().toString(),
      title,
      timestamp: new Date().toISOString(),
      sourceText,
      translatedFullText,
      notes,
      sourceLang,
      targetLang,
      uiLang, // Save current UI lang
      translationMode,
    };
    setHistoryItems(prev => [...prev, newHistoryItem]);
    if(!historyMessage || historyMessage.type !== 'error') { // Don't overwrite title gen error
        setHistoryMessage({type: 'success', text: t('sessionSavedSuccess')});
    }
    setIsSavingSession(false);
  };

  const handleLoadHistoryItem = (itemId: string) => {
    const itemToLoad = historyItems.find(item => item.id === itemId);
    if (itemToLoad) {
      setSourceText(itemToLoad.sourceText);
      setTranslatedFullText(itemToLoad.translatedFullText); // This will trigger buffer re-init via useEffect
      setNotes(itemToLoad.notes);
      setSourceLang(itemToLoad.sourceLang);
      setTargetLang(itemToLoad.targetLang);
      setUiLang(itemToLoad.uiLang); // Restore UI lang
      setTranslationMode(itemToLoad.translationMode as TranslationMode);
      
      // Explicitly re-initialize note buffer related refs after state updates propagate
      // Use a timeout to allow state updates to apply before re-initializing buffer.
      // This is a common pattern when an effect depends on state that's just been set.
      setTimeout(() => {
         initializeNoteBufferAndProcessedLength(itemToLoad.translatedFullText);
      }, 0);

      setIsHistoryPanelOpen(false);
      setHistoryMessage({type: 'success', text: t('sessionLoadedSuccess')});
      setError(null);
    }
  };

  const handleDeleteHistoryItem = (itemId: string) => {
    setHistoryItems(prev => prev.filter(item => item.id !== itemId));
    setHistoryMessage({type: 'success', text: t('sessionDeletedSuccess')});
  };

  const handleUpdateHistoryItemTitle = (itemId: string, newTitle: string) => {
    setHistoryItems(prev => prev.map(item => item.id === itemId ? { ...item, title: newTitle } : item));
    setHistoryMessage({type: 'success', text: t('titleUpdatedSuccess')});
  };
  // --- End History Functions ---

  useEffect(() => { // Clear history message after a few seconds
    if (historyMessage) {
      const timer = setTimeout(() => setHistoryMessage(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [historyMessage]);


  if (isLoadingLocales) {
    return (
      <div className="min-h-screen flex flex-col justify-center items-center bg-gray-900 text-gray-100 p-4">
        <h1 className="text-2xl font-bold text-purple-400">Loading application settings...</h1>
      </div>
    );
  }
  
  if (typeof process !== 'undefined' && process.env && !process.env.API_KEY) {
    console.warn("[App Debug] API_KEY is not set. Gemini API calls will likely fail.");
  }

  const canGenerateNote = noteGenerationBufferRef.current.trim().length > 0 && !isGeneratingNote;
  const currentModeText = translationMode === 'realtime' ? t('translationModeRealtime') : t('translationModeComposed');

  return (
    <div className="min-h-screen flex flex-col bg-gray-900 text-gray-100 p-4 space-y-4" lang={uiLang.split('-')[0]}>
      <header className="text-center py-4">
        <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-500 to-red-500">
          {t('appTitle')}
        </h1>
        <p className="text-gray-400 text-sm">{t('appSubtitle')}</p>
      </header>

      <div className="flex flex-wrap justify-between items-center gap-2 mb-1">
        <div className="w-full sm:w-auto sm:flex-1 sm:max-w-xs">
             <LanguageSelector
                id="ui-lang"
                label={t('uiLangLabel')}
                selectedLang={uiLang}
                onLangChange={setUiLang}
                languages={SUPPORTED_UI_LANGS}
             />
        </div>
        <div className="flex gap-2 flex-wrap items-center">
            <button
                onClick={handleSaveCurrentSession}
                disabled={isSavingSession}
                className="px-3 py-2 text-xs bg-teal-600 hover:bg-teal-700 text-white font-semibold rounded-lg shadow-md transition duration-150 ease-in-out flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
                title={t('saveSessionButton')}
            >
                {isSavingSession && <LoadingSpinner className="h-4 w-4 mr-2"/>}
                {isSavingSession ? t('savingSessionStatus') : t('saveSessionButton')}
            </button>
            <button
                onClick={() => setIsHistoryPanelOpen(true)}
                className="px-3 py-2 text-xs bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg shadow-md transition duration-150 ease-in-out"
                title={t('viewHistoryButton')}
            >
                {t('viewHistoryButton')}
            </button>
             <button
                onClick={toggleTranslationMode}
                className="px-3 py-2 text-xs bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg shadow-md transition duration-150 ease-in-out"
                title={translationMode === 'realtime' ? t('switchToComposedModeTitle') : t('switchToRealtimeModeTitle')}
             >
                {t('translationModeButtonLabel', { currentMode: currentModeText })}
             </button>
            <button
                onClick={handleGenerateNoteManually}
                disabled={!canGenerateNote}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg shadow-md transition duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label={t('generateNoteManuallyButton')}
                title={!canGenerateNote && !isGeneratingNote && noteGenerationBufferRef.current.trim().length === 0 ? t('noteBufferEmptyError') : t('generateNoteManuallyButton')}
            >
                {t('generateNoteManuallyButton')}
            </button>
            <button
                onClick={handleClearAll}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-lg shadow-md transition duration-150 ease-in-out"
                aria-label={t('clearAll')}
            >
                {t('clearAll')}
            </button>
        </div>
      </div>
      
      {historyMessage && (
        <div className={`border ${historyMessage.type === 'success' ? 'bg-green-800 border-green-600 text-green-100' : 'bg-red-800 border-red-600 text-red-100'} px-4 py-3 rounded-md relative`} role="alert">
          <span className="block sm:inline">{historyMessage.text}</span>
        </div>
      )}

      {error && (
        <div className="bg-red-800 border border-red-600 text-red-100 px-4 py-3 rounded-md relative" role="alert">
          <strong className="font-bold">{t('errorPrefix')}: </strong>
          <span className="block sm:inline">{error}</span>
        </div>
      )}

      {currentInterimSTT && (
        <div 
            className="text-sm text-gray-400 p-2 bg-gray-800 rounded-md border border-gray-700 shadow"
            aria-live="polite" 
            aria-atomic="true"
        >
            <span className="font-semibold">{t('liveCaptionPrefix')} </span>{currentInterimSTT}
        </div>
      )}


      <div className="flex-grow grid grid-cols-1 md:grid-cols-3 gap-4">
        <SourceInputPanel
          sourceText={sourceText}
          onSourceTextChange={handleSourceTextChange}
          onInterimResult={handleInterimResult}
          onFinalChunkResult={handleFinalChunkResult}
          sourceLang={sourceLang}
          onSourceLangChange={setSourceLang}
          targetLang={targetLang}
          onTargetLangChange={setTargetLang}
          supportedLanguages={SUPPORTED_LANGUAGES}
          isLoading={isTranslating}
        />
        <TranslationOutputPanel
          translatedText={translatedFullText}
          isLoading={isTranslating}
        />
        <NotesPanel
          notes={notes}
          isLoading={isGeneratingNote}
        />
      </div>
      
      <HistoryPanel
        isOpen={isHistoryPanelOpen}
        onClose={() => setIsHistoryPanelOpen(false)}
        historyItems={historyItems}
        onLoadItem={handleLoadHistoryItem}
        onDeleteItem={handleDeleteHistoryItem}
        onUpdateItemTitle={handleUpdateHistoryItemTitle}
        isLoadingHistoryAction={isSavingSession} // Could be more granular if needed
      />
      
      <footer className="text-center py-4 text-gray-500 text-xs">
        {t('footerText')}
      </footer>
    </div>
  );
};

export default App;
