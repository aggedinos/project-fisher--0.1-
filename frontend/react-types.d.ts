// Minimal React declarations for the hooks used by this frontend. The cloud
// environment ships React without @types/react, and npm access is unavailable.
// Replace with the official @types packages when the package registry is usable.
declare module 'react' {
  export type SetStateAction<S> = S | ((previous: S) => S);
  export type Dispatch<A> = (value: A) => void;
  export function useState<S>(initial: S | (() => S)): [S, Dispatch<SetStateAction<S>>];
  export function useEffect(effect: () => void | (() => void), dependencies?: readonly unknown[]): void;
  export function useMemo<T>(factory: () => T, dependencies: readonly unknown[]): T;
  export function useCallback<T extends (...args: never[]) => unknown>(callback: T, dependencies: readonly unknown[]): T;
  const React: { StrictMode: (props: { children?: unknown }) => unknown };
  export default React;
}

declare module 'react/jsx-runtime' {
  export namespace JSX {
    type Element = unknown;
    interface IntrinsicElements { [element: string]: Record<string, unknown> }
  }
  export const jsx: (type: unknown, props: unknown, key?: unknown) => JSX.Element;
  export const jsxs: typeof jsx;
}

declare module 'react-dom/client' {
  export function createRoot(element: Element): { render(node: unknown): void };
}
