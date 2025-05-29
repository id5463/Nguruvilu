import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';
import { DEFAULT_UI_LANG, LOCAL_STORAGE_KEYS } from '../constants';

type Translations = Record<string, string>;
type LocaleData = Record<string, Translations>;

interface LocalizationContextType {
  language: string;
  setLanguage: (lang: string) => void;
  t: (key: string, params?: Record<string, string | number>) => string;
  isLoadingLocales: boolean;
}

const LocalizationContext = createContext<LocalizationContextType | undefined>(undefined);

const baseLocalePath = './locales'; // Relative to index.html

export const LocalizationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [language, setLanguageState] = useState<string>(() => {
    const storedLang = localStorage.getItem(LOCAL_STORAGE_KEYS.UI_LANG);
    // Validate storedLang against a known list if necessary, for now assume it's 'en' or 'zh-CN' if valid
    return ['en', 'zh-CN'].includes(storedLang || '') ? storedLang! : DEFAULT_UI_LANG;
  });
  
  const [translations, setTranslations] = useState<LocaleData>({});
  const [isLoadingLocales, setIsLoadingLocales] = useState<boolean>(true);

  useEffect(() => {
    localStorage.setItem(LOCAL_STORAGE_KEYS.UI_LANG, language);
    document.documentElement.lang = language.split('-')[0];
  }, [language]);

  const setLanguage = useCallback((lang: string) => {
    // Check if desired lang's translations are loaded, or if it's a supported lang
    if (translations[lang] || ['en', 'zh-CN'].includes(lang)) {
      setLanguageState(lang);
    } else {
      console.warn(`Language "${lang}" not supported or translations not loaded. Falling back to default.`);
      setLanguageState(DEFAULT_UI_LANG);
    }
  }, [translations]);
  
  useEffect(() => {
    const fetchLocales = async () => {
      setIsLoadingLocales(true);
      try {
        const [enResponse, zhCNResponse] = await Promise.all([
          fetch(`${baseLocalePath}/en.json`),
          fetch(`${baseLocalePath}/zh-CN.json`)
        ]);

        if (!enResponse.ok) throw new Error(`Failed to load en.json: ${enResponse.statusText}`);
        if (!zhCNResponse.ok) throw new Error(`Failed to load zh-CN.json: ${zhCNResponse.statusText}`);

        const enData = await enResponse.json();
        const zhCNData = await zhCNResponse.json();
        
        setTranslations({
          en: enData,
          'zh-CN': zhCNData,
        });
      } catch (error) {
        console.error("Failed to load translations:", error);
        // Fallback to empty or minimal translations if loading fails
        setTranslations({
          en: { appTitle: "App (Error Loading Names)" }, // Minimal fallback
          'zh-CN': { appTitle: "应用 (名称加载错误)"}
        });
      } finally {
        setIsLoadingLocales(false);
      }
    };

    // eslint-disable-next-line @typescript-eslint/no-floating-promises
    fetchLocales();
  }, []);


  const t = useCallback((key: string, params?: Record<string, string | number>): string => {
    if (isLoadingLocales) return key; // Or a loading indicator string

    let translation = translations[language]?.[key] || translations[DEFAULT_UI_LANG]?.[key] || key;
    
    if (params) {
      Object.keys(params).forEach(paramKey => {
        translation = translation.replace(new RegExp(`{${paramKey}}`, 'g'), String(params[paramKey]));
      });
    }
    return translation;
  }, [language, translations, isLoadingLocales]);

  useEffect(() => {
    document.documentElement.lang = language.split('-')[0];
  }, [language]);


  return (
    <LocalizationContext.Provider value={{ language, setLanguage, t, isLoadingLocales }}>
      {children}
    </LocalizationContext.Provider>
  );
};

export const useLocalization = (): LocalizationContextType => {
  const context = useContext(LocalizationContext);
  if (context === undefined) {
    throw new Error('useLocalization must be used within a LocalizationProvider');
  }
  return context;
};
