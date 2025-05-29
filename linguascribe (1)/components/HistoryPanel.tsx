
import React, { useState, useEffect } from 'react';
import type { HistoryItem } from '../types';
import { useLocalization } from '../contexts/LocalizationContext';
import { LoadingSpinner } from './LoadingSpinner';

interface HistoryPanelProps {
  isOpen: boolean;
  onClose: () => void;
  historyItems: HistoryItem[];
  onLoadItem: (itemId: string) => void;
  onDeleteItem: (itemId: string) => void;
  onUpdateItemTitle: (itemId: string, newTitle: string) => void;
  isLoadingHistoryAction: boolean; // Generic loading state for any history panel action
}

export const HistoryPanel: React.FC<HistoryPanelProps> = ({
  isOpen,
  onClose,
  historyItems,
  onLoadItem,
  onDeleteItem,
  onUpdateItemTitle,
  isLoadingHistoryAction,
}) => {
  const { t, language: uiLang } = useLocalization();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState<string>('');
  const [itemToDelete, setItemToDelete] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      setEditingId(null);
      setEditingTitle('');
      setItemToDelete(null);
    }
  }, [isOpen]);

  const handleStartEdit = (item: HistoryItem) => {
    setEditingId(item.id);
    setEditingTitle(item.title);
  };

  const handleSaveEdit = () => {
    if (editingId && editingTitle.trim()) {
      onUpdateItemTitle(editingId, editingTitle.trim());
    }
    setEditingId(null);
    setEditingTitle('');
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditingTitle('');
  };

  const handleDeleteConfirm = (itemId: string) => {
    setItemToDelete(itemId);
  };
  
  const executeDelete = () => {
    if (itemToDelete) {
      onDeleteItem(itemToDelete);
    }
    setItemToDelete(null);
  };

  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center p-4 z-50 transition-opacity duration-300 ease-in-out"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="history-panel-title"
    >
      <div 
        className="bg-gray-800 p-6 rounded-lg shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col border border-gray-700"
        onClick={(e) => e.stopPropagation()} // Prevent closing modal when clicking inside
      >
        <div className="flex justify-between items-center mb-4">
          <h2 id="history-panel-title" className="text-2xl font-bold text-purple-400">{t('historyPanelTitle')}</h2>
          <button 
            onClick={onClose} 
            className="text-gray-400 hover:text-gray-200 p-1 rounded-full"
            aria-label={t('cancelEditButton')} // Using 'Cancel' as a generic close
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-6 h-6">
              <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
            </svg>
          </button>
        </div>

        {isLoadingHistoryAction && (
          <div className="absolute inset-0 bg-gray-800 bg-opacity-50 flex items-center justify-center z-10 rounded-lg">
            <LoadingSpinner className="h-8 w-8 text-purple-400" />
          </div>
        )}

        {itemToDelete && (
          <div className="fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center p-4 z-[60]">
             <div className="bg-gray-700 p-6 rounded-lg shadow-xl border border-gray-600">
                <p className="text-lg mb-4">{t('confirmDeleteSessionMessage')}</p>
                <div className="flex justify-end space-x-3">
                    <button onClick={() => setItemToDelete(null)} className="px-4 py-2 text-sm bg-gray-600 hover:bg-gray-500 rounded-md">{t('cancelEditButton')}</button>
                    <button onClick={executeDelete} className="px-4 py-2 text-sm bg-red-600 hover:bg-red-700 rounded-md">{t('deleteSessionButton')}</button>
                </div>
             </div>
          </div>
        )}

        {historyItems.length === 0 ? (
          <p className="text-gray-400 text-center py-8">{t('noSavedSessions')}</p>
        ) : (
          <div className="overflow-y-auto flex-grow space-y-3 pr-1 custom-scrollbar">
            {historyItems.slice().reverse().map((item) => ( // Display newest first
              <div key={item.id} className="bg-gray-700 p-3 rounded-md border border-gray-600 hover:border-purple-500 transition-colors">
                {editingId === item.id ? (
                  <div className="space-y-2">
                    <input
                      type="text"
                      value={editingTitle}
                      onChange={(e) => setEditingTitle(e.target.value)}
                      className="w-full p-2 bg-gray-600 text-gray-100 border border-gray-500 rounded-md focus:ring-purple-500 focus:border-purple-500"
                      placeholder={t('enterNewTitlePlaceholder')}
                      aria-label={t('enterNewTitlePlaceholder')}
                    />
                    <div className="flex space-x-2">
                      <button onClick={handleSaveEdit} className="px-3 py-1 text-xs bg-green-600 hover:bg-green-700 rounded-md">{t('saveTitleButton')}</button>
                      <button onClick={handleCancelEdit} className="px-3 py-1 text-xs bg-gray-500 hover:bg-gray-400 rounded-md">{t('cancelEditButton')}</button>
                    </div>
                  </div>
                ) : (
                  <div>
                    <h3 className="text-md font-semibold text-purple-300">{item.title}</h3>
                    <p className="text-xs text-gray-400 mb-2">{new Date(item.timestamp).toLocaleString(uiLang)}</p>
                    <div className="flex flex-wrap gap-2 items-center">
                      <button 
                        onClick={() => onLoadItem(item.id)}
                        className="px-3 py-1 text-xs bg-blue-600 hover:bg-blue-700 rounded-md"
                        aria-label={`${t('loadSessionButton')} ${item.title}`}
                      >
                        {t('loadSessionButton')}
                      </button>
                      <button 
                        onClick={() => handleStartEdit(item)}
                        className="px-3 py-1 text-xs bg-yellow-600 hover:bg-yellow-700 text-black rounded-md"
                        aria-label={`${t('editTitleButton')} ${item.title}`}
                      >
                        {t('editTitleButton')}
                      </button>
                      <button 
                        onClick={() => handleDeleteConfirm(item.id)}
                        className="px-3 py-1 text-xs bg-red-600 hover:bg-red-700 rounded-md"
                        aria-label={`${t('deleteSessionButton')} ${item.title}`}
                      >
                        {t('deleteSessionButton')}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
