
import React from 'react';
import type { Note } from '../types';
import { NoteCard } from './NoteCard';
import { LoadingSpinner } from './LoadingSpinner';
import { useLocalization } from '../contexts/LocalizationContext';

interface NotesPanelProps {
  notes: Note[];
  isLoading: boolean;
}

export const NotesPanel: React.FC<NotesPanelProps> = ({ notes, isLoading }) => {
  const { t } = useLocalization();

  return (
    <div className="bg-gray-800 p-4 rounded-lg shadow-xl flex flex-col h-full">
      <h2 className="text-xl font-semibold text-red-500 flex items-center mb-3">
        {t('notesPanelTitle')}
        {isLoading && <LoadingSpinner className="ml-2 h-5 w-5 text-red-500" />}
      </h2>
      {notes.length === 0 && !isLoading && (
        <div className="flex-grow flex items-center justify-center">
          <p className="text-gray-500">{t('notesPlaceholder')}</p>
        </div>
      )}
      {notes.length === 0 && isLoading && (
         <div className="flex-grow flex items-center justify-center">
            <p className="text-gray-400">{t('generatingFirstNoteStatus')}</p>
          </div>
      )}
      {notes.length > 0 && (
        <div className="flex-grow space-y-3 overflow-y-auto pr-1 custom-scrollbar">
          {notes.map((note, index) => (
            <NoteCard key={note.id} note={note} noteNumber={index + 1} />
          ))}
           {isLoading && notes.length > 0 && (
            <div className="flex items-center justify-center mt-2">
                <LoadingSpinner className="h-4 w-4 text-red-500 mr-2"/>
                <p className="text-sm text-gray-400">{t('generatingNoteStatus')}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
