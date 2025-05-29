import React, { useState } from 'react';
import type { Note } from '../types';
import { useLocalization } from '../contexts/LocalizationContext';

interface NoteCardProps {
  note: Note;
  noteNumber: number;
}

export const NoteCard: React.FC<NoteCardProps> = ({ note, noteNumber }) => {
  const { t, language: uiLocale } = useLocalization(); // Destructure language, aliased to uiLocale for clarity
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="bg-gray-700 p-3 rounded-md shadow-lg border border-gray-600 hover:border-red-500 transition-colors">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-lg font-semibold text-red-400">{t('noteCardTitle', { noteNumber })}</h3>
        <span className="text-xs text-gray-400">{new Date(note.timestamp).toLocaleString(uiLocale)}</span>
      </div>
      <p className="text-gray-200 whitespace-pre-wrap text-sm leading-relaxed">{note.summary}</p>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="text-xs text-purple-400 hover:text-purple-300 mt-2 focus:outline-none"
        aria-expanded={isExpanded}
      >
        {isExpanded ? t('hideOriginalSegment') : t('showOriginalSegment')}
      </button>
      {isExpanded && (
        <div className="mt-2 pt-2 border-t border-gray-600">
          <h4 className="text-xs font-semibold text-gray-400 mb-1">{t('originalTranslatedSegment')}</h4>
          <p className="text-xs text-gray-300 whitespace-pre-wrap max-h-32 overflow-y-auto custom-scrollbar">
            {note.sourceSegment}
          </p>
        </div>
      )}
    </div>
  );
};