
import React from 'react';
import type { Language } from '../types';

interface LanguageSelectorProps {
  id: string;
  label: string;
  selectedLang: string;
  onLangChange: (langCode: string) => void;
  languages: Language[];
  disabled?: boolean;
}

export const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  id,
  label,
  selectedLang,
  onLangChange,
  languages,
  disabled = false,
}) => {
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium text-gray-300 mb-1">
        {label}
      </label>
      <select
        id={id}
        name={id} // Good practice for forms, though not strictly necessary here
        value={selectedLang}
        onChange={(e) => onLangChange(e.target.value)}
        disabled={disabled}
        className="w-full p-2.5 border border-gray-600 rounded-md bg-gray-700 text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-colors disabled:opacity-70 disabled:cursor-not-allowed"
        aria-label={label}
      >
        {languages.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.name}
          </option>
        ))}
      </select>
    </div>
  );
};
