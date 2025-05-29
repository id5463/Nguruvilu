import React, { useState, useEffect, useRef, useCallback } from 'react';
import { LanguageSelector } from './LanguageSelector';
import { LoadingSpinner } from './LoadingSpinner';
import type { Language } from '../types';
import { useLocalization } from '../contexts/LocalizationContext';

interface SourceInputPanelProps {
  sourceText: string; // This will now represent the cumulative FINALIZED source text
  onSourceTextChange: (text: string) => void; // For text area changes
  onInterimResult: (interimText: string) => void; // New: For live STT interim results
  onFinalChunkResult: (finalChunk: string) => void; // New: For a newly finalized STT chunk
  sourceLang: string;
  onSourceLangChange: (langCode: string) => void;
  targetLang: string;
  onTargetLangChange: (langCode: string) => void;
  supportedLanguages: Language[];
  isLoading: boolean; // ParentisLoading (e.g. for Stage 2 translation)
}

const SpeechRecognitionAPI = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
const recognitionAvailable = !!SpeechRecognitionAPI;

export const SourceInputPanel: React.FC<SourceInputPanelProps> = ({
  sourceText,
  onSourceTextChange,
  onInterimResult,
  onFinalChunkResult,
  sourceLang,
  onSourceLangChange,
  targetLang,
  onTargetLangChange,
  supportedLanguages,
  isLoading,
}) => {
  const { t } = useLocalization();
  const [isRecording, setIsRecording] = useState(false);
  const [speechError, setSpeechError] = useState<string | null>(null);
  const recognitionRef = useRef<any>(null);
  const isRecordingRef = useRef(isRecording);

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  const startRecording = useCallback(() => {
    if (!recognitionAvailable) {
      setSpeechError(t('speechRecognitionNotSupported'));
      return;
    }
    if (isLoading && isRecording) return; // Prevent issues if parent is already busy

    setSpeechError(null);
    setIsRecording(true);

    if (recognitionRef.current) {
        recognitionRef.current.stop();
        recognitionRef.current = null;
    }

    recognitionRef.current = new SpeechRecognitionAPI();
    recognitionRef.current.continuous = true;
    recognitionRef.current.interimResults = true;
    recognitionRef.current.lang = sourceLang; // Use the current sourceLang

    let lastFinalizedPosition = 0; // To track what has been finalized

    recognitionRef.current.onstart = () => {
      console.log("Speech recognition started with lang:", sourceLang);
      lastFinalizedPosition = 0; // Reset for new session
    };

    recognitionRef.current.onresult = (event: any) => {
      let interimTranscript = '';
      let finalTranscriptChunk = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const transcriptSegment = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscriptChunk += transcriptSegment;
        } else {
          interimTranscript += transcriptSegment;
        }
      }
      
      // Provide the full current interim transcript
      if (interimTranscript) {
        onInterimResult(interimTranscript);
      }

      // If there's a new finalized chunk, send it
      if (finalTranscriptChunk.trim()) {
        onFinalChunkResult(finalTranscriptChunk.trim() + ' '); // Add space for sentence separation
        onInterimResult(''); // Clear interim display as text is now final
      }
    };

    recognitionRef.current.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error, event.message);
      let msgKey = 'speechRecognitionError';
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        msgKey = 'microphoneAccessDenied';
      } else if (event.error === 'no-speech') {
        // This can happen, often recoverable by restart logic in onend
        setSpeechError(t('speechRecognitionError') + ": No speech. It might restart.");
        // Don't immediately set isRecording to false, let onend handle restart.
        return; 
      }
      setSpeechError(t(msgKey) + (event.error !== 'not-allowed' && event.error !== 'service-not-allowed' ? `: ${event.error}`: ''));
      setIsRecording(false); // Stop UI indication on other errors
      onInterimResult(''); // Clear interim display on error
    };

    recognitionRef.current.onend = () => {
      console.log("Speech recognition native onend called.");
      onInterimResult(''); // Clear any dangling interim text
      if (isRecordingRef.current) {
        console.log("Attempting to restart speech recognition as isRecordingRef is true...");
        if (recognitionRef.current) { 
          try {
            recognitionRef.current.start();
          } catch (err) {
            console.error("Error restarting speech recognition in onend:", err);
            setIsRecording(false);
          }
        }
      } else {
        setIsRecording(false); 
      }
    };
    
    try {
        recognitionRef.current.start();
    } catch (e) {
        console.error("Failed to start speech recognition:", e);
        setSpeechError(t('speechRecognitionError') + ": Could not start service.");
        setIsRecording(false);
    }

  }, [isLoading, sourceLang, onFinalChunkResult, onInterimResult, t]);

  const stopRecording = useCallback(() => {
    setIsRecording(false); 
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      console.log("Manual stop initiated for speech recognition.");
    }
    onInterimResult(''); // Clear interim display on manual stop
  }, [onInterimResult]);

  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        console.log("SourceInputPanel unmounting, stopping recognition.");
        recognitionRef.current.onresult = null; // Avoid errors after unmount
        recognitionRef.current.onerror = null;
        recognitionRef.current.onend = null;
        recognitionRef.current.stop();
        recognitionRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (isLoading && isRecording) {
      console.log("Parent isLoading became true, stopping speech recording if active.");
      stopRecording();
    }
  }, [isLoading, isRecording, stopRecording]);


  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };
  
  const MicIcon = ({ active }: { active: boolean }) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={`w-5 h-5 mr-2 ${active ? 'text-red-500 animate-pulse' : 'text-gray-400'}`}>
      <path d="M11.9999 14.942C13.5999 14.942 14.8999 13.642 14.8999 12.042V6.94204C14.8999 5.34204 13.5999 4.04204 11.9999 4.04204C10.3999 4.04204 9.09985 5.34204 9.09985 6.94204V12.042C9.09985 13.642 10.3999 14.942 11.9999 14.942Z" />
      <path d="M19.8 12C19.8 16.23 16.39 19.48 12 19.48C7.61 19.48 4.2 16.23 4.2 12H6C6 15.31 8.69 18 12 18C15.31 18 18 15.31 18 12H19.8Z" />
       <path d="M12 21.75C10.1775 21.75 8.505 21.0938 7.20375 19.9688L8.26875 18.9038C9.28125 19.7663 10.5825 20.25 12 20.25C13.4175 20.25 14.7188 19.7663 15.7313 18.9038L16.7963 19.9688C15.495 21.0938 13.8225 21.75 12 21.75Z"/>
    </svg>
  );

  return (
    <div className="bg-gray-800 p-4 rounded-lg shadow-xl flex flex-col space-y-4 h-full">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-purple-400 flex items-center">
          {t('sourceTextPanelTitle')}
          {/* isLoading here refers to Stage 2/Main translation, not Stage 1 */}
          {isLoading && <LoadingSpinner className="ml-2 h-5 w-5 text-purple-400" />}
        </h2>
        {recognitionAvailable && (
          <button
            onClick={toggleRecording}
            disabled={isLoading && isRecording} 
            className={`p-2 rounded-full flex items-center justify-center transition-colors 
                        ${isRecording ? 'bg-red-600 hover:bg-red-700 text-white' : 'bg-gray-700 hover:bg-gray-600 text-gray-300'}
                        disabled:opacity-50 disabled:cursor-not-allowed`}
            aria-label={isRecording ? t('stopRecording') : t('startRecording')}
            title={isRecording ? t('stopRecording') : t('startRecording')}
          >
            <MicIcon active={isRecording} />
            {isRecording ? t('recordingInProgress') : ''}
          </button>
        )}
      </div>
       {speechError && <p className="text-xs text-red-400" role="alert">{speechError}</p>}
       {!recognitionAvailable && <p className="text-xs text-yellow-400">{t('speechRecognitionNotSupported')}</p>}

      <div className="grid grid-cols-2 gap-4">
        <LanguageSelector
          id="source-lang"
          label={t('sourceLangLabel')}
          selectedLang={sourceLang}
          onLangChange={onSourceLangChange}
          languages={supportedLanguages}
          disabled={isLoading || isRecording}
        />
        <LanguageSelector
          id="target-lang"
          label={t('targetLangLabel')}
          selectedLang={targetLang}
          onLangChange={onTargetLangChange} 
          languages={supportedLanguages}
          disabled={isLoading || isRecording}
        />
      </div>
      <textarea
        value={sourceText} // Represents cumulative FINALIZED text
        onChange={(e) => onSourceTextChange(e.target.value)}
        placeholder={t('sourceTextPlaceholder')}
        className="flex-grow w-full p-3 border border-gray-600 rounded-md bg-gray-700 text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 resize-none min-h-[200px] md:min-h-0"
        disabled={isLoading && !isRecording} // Allow typing if recording, even if parent is loading
        lang={sourceLang.split('-')[0]} 
      />
    </div>
  );
};