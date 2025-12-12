import '@testing-library/jest-dom/vitest';
import { vi, beforeEach, afterEach } from 'vitest';

// Mock do fetch global para testes
global.fetch = vi.fn();

// Mock do window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock do ResizeObserver - Class-based para Recharts
class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

global.ResizeObserver = ResizeObserverMock;

// Mock IntersectionObserver
class IntersectionObserverMock {
  constructor() {}
  observe() {}
  unobserve() {}
  disconnect() {}
}

global.IntersectionObserver = IntersectionObserverMock;

// Reset mocks entre testes
beforeEach(() => {
  vi.clearAllMocks();
});

// Cleanup após cada teste
afterEach(() => {
  vi.restoreAllMocks();
});
