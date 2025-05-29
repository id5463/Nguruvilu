
export const countWords = (text: string): number => {
  if (!text || !text.trim()) {
    return 0;
  }
  // Matches sequences of non-whitespace characters.
  const words = text.match(/\S+/g);
  return words ? words.length : 0;
};

export function debounce<T extends (...args: any[]) => void>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  return (...args: Parameters<T>) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      func(...args);
    }, delay);
  };
}

/**
 * Extracts a segment of text up to a specified word limit.
 * @param text The input text.
 * @param wordLimit The maximum number of words for the segment.
 * @returns An object containing the extracted segment and the remainder of the text.
 */
export function extractSegmentByWords(text: string, wordLimit: number): { segment: string; remainder: string } {
  if (!text.trim()) {
    return { segment: "", remainder: "" };
  }

  const words = text.split(/(\s+)/); // Split by whitespace, keeping delimiters
  let currentWordCount = 0;
  let segmentEndIndex = -1;

  for (let i = 0; i < words.length; i++) {
    if (words[i].trim().length > 0) { // Count non-empty words
      currentWordCount++;
    }
    if (currentWordCount >= wordLimit) {
      segmentEndIndex = i;
      break;
    }
  }

  if (segmentEndIndex === -1 || currentWordCount < wordLimit ) { // Text is shorter than word limit or exactly word limit
    return { segment: text, remainder: "" };
  }

  // Find the end of the word/delimiter at segmentEndIndex
  let actualEndIndexInOriginal = 0;
  for(let i = 0; i <= segmentEndIndex; i++) {
    actualEndIndexInOriginal += words[i].length;
  }
  
  const segment = text.substring(0, actualEndIndexInOriginal).trimEnd();
  const remainder = text.substring(actualEndIndexInOriginal).trimStart();
  
  return { segment, remainder };
}
    