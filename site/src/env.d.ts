declare const __MATCH_VERSION__: string;
declare const __MATCH_WHEEL_NAME__: string;

interface Window {
  MathJax?: {
    tex?: {
      inlineMath?: string[][];
      displayMath?: string[][];
      processEscapes?: boolean;
    };
    typesetPromise?: () => Promise<void>;
  };
}
