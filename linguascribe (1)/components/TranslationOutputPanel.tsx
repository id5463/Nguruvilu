
import React, { useState } from 'react';
import { LoadingSpinner } from './LoadingSpinner';
import { useLocalization } from '../contexts/LocalizationContext';

interface TranslationOutputPanelProps {
  translatedText: string;
  isLoading: boolean; // isLoading refers to the main (Stage 2) translation
}

export const TranslationOutputPanel: React.FC<TranslationOutputPanelProps> = ({
  translatedText,
  isLoading,
}) => {
  const { t } = useLocalization();
  const [showCopyFeedback, setShowCopyFeedback] = useState(false);

  const handleCopyToClipboard = async () => {
    if (!translatedText) return;
    try {
      await navigator.clipboard.writeText(translatedText);
      setShowCopyFeedback(true);
      setTimeout(() => setShowCopyFeedback(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
      // Optionally, show an error message to the user
    }
  };

  const CopyIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path d="M7 3.5A1.5 1.5 0 0 1 8.5 2h3.879a1.5 1.5 0 0 1 1.06.44l3.122 3.12A1.5 1.5 0 0 1 17 6.622V16.5a1.5 1.5 0 0 1-1.5 1.5h-7A1.5 1.5 0 0 1 7 16.5v-13Zm0-2A3.5 3.5 0 0 0 3.5 5v11.5A3.5 3.5 0 0 0 7 20h7.5a3.5 3.5 0 0 0 3.5-3.5V6.622a3.5 3.5 0 0 0-1.025-2.475L13.854 1.025A3.5 3.5 0 0 0 11.379 0H8.5A3.5 3.5 0 0 0 5 3.5v1A1.5 1.5 0 0 0 7 6V3.5ZM8.5 4a.5.5 0 0 0 0 1h5a.5.5 0 0 0 0-1h-5Z" />
    </svg>
  );


  return (
    <div className="bg-gray-800 p-4 rounded-lg shadow-xl flex flex-col h-full">
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-xl font-semibold text-pink-500 flex items-center">
          {t('translationPanelTitle')}
          {/* Show spinner if loading and there's no existing text to display */}
          {isLoading && !translatedText && <LoadingSpinner className="ml-2 h-5 w-5 text-pink-500" />}
        </h2>
        {translatedText && (
          <button
            onClick={handleCopyToClipboard}
            className="p-1.5 text-gray-400 hover:text-pink-400 transition-colors relative"
            title={t('copiedToClipboard')}
            aria-label={t('copiedToClipboard')}
          >
            <CopyIcon />
            {showCopyFeedback && (
              <span className="absolute -top-7 right-0 bg-gray-900 text-white text-xs px-2 py-1 rounded-md shadow-lg">
                {t('copiedToClipboard')}
              </span>
            )}
          </button>
        )}
      </div>
      <div className="flex-grow p-3 border border-gray-600 rounded-md bg-gray-700 text-gray-200 overflow-y-auto whitespace-pre-wrap min-h-[200px] md:min-h-0">
        {/* Show "Translating..." if loading and there's no prior text to display */}
        {isLoading && !translatedText ? (
          <div className="flex justify-center items-center h-full">
            <p className="text-gray-400">{t('translatingStatus')}</p>
          </div>
        ) : translatedText ? (
          translatedText // Display the main translated text
        ) : (
          <p className="text-gray-500">{t('translationPlaceholder')}</p>
        )}
      </div>
    </div>
  );
};