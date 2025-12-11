/**
 * Sankofa Enterprise Pro - API Store
 * Zustand-based API client with request interceptors
 * 
 * Features:
 * - Centralized API client
 * - Automatic token injection
 * - Request/response interceptors
 * - Error handling
 * - Retry logic
 */

import { create } from 'zustand';
import useAuthStore from './authStore';

// Default API configuration
const DEFAULT_CONFIG = {
    baseUrl: '/api',
    timeout: 30000,
    retries: 3,
    retryDelay: 1000,
};

/**
 * Sleep utility
 */
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * API Store with fetch wrapper
 */
const useApiStore = create((set, get) => ({
    // State
    config: { ...DEFAULT_CONFIG },
    pendingRequests: 0,
    lastError: null,
    
    // Request interceptors
    requestInterceptors: [],
    responseInterceptors: [],
    
    /**
     * Update configuration
     */
    setConfig: (config) => {
        set(state => ({
            config: { ...state.config, ...config },
        }));
    },
    
    /**
     * Add request interceptor
     */
    addRequestInterceptor: (interceptor) => {
        set(state => ({
            requestInterceptors: [...state.requestInterceptors, interceptor],
        }));
        
        // Return remove function
        return () => {
            set(state => ({
                requestInterceptors: state.requestInterceptors.filter(i => i !== interceptor),
            }));
        };
    },
    
    /**
     * Add response interceptor
     */
    addResponseInterceptor: (interceptor) => {
        set(state => ({
            responseInterceptors: [...state.responseInterceptors, interceptor],
        }));
        
        return () => {
            set(state => ({
                responseInterceptors: state.responseInterceptors.filter(i => i !== interceptor),
            }));
        };
    },
    
    /**
     * Core request method with interceptors
     */
    request: async (endpoint, options = {}) => {
        const { config, requestInterceptors, responseInterceptors } = get();
        const authStore = useAuthStore.getState();
        
        // Build URL
        const url = endpoint.startsWith('http')
            ? endpoint
            : `${config.baseUrl}${endpoint}`;
        
        // Default headers
        let headers = {
            'Content-Type': 'application/json',
            ...options.headers,
        };
        
        // Add auth token if available
        const token = authStore.getToken();
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        
        // Build request config
        let requestConfig = {
            ...options,
            headers,
        };
        
        // Apply request interceptors
        for (const interceptor of requestInterceptors) {
            requestConfig = await interceptor(requestConfig);
        }
        
        // Track pending requests
        set(state => ({ pendingRequests: state.pendingRequests + 1 }));
        
        let lastError = null;
        const maxRetries = options.retries ?? config.retries;
        
        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                // Create abort controller for timeout
                const controller = new AbortController();
                const timeoutId = setTimeout(
                    () => controller.abort(),
                    options.timeout ?? config.timeout
                );
                
                requestConfig.signal = controller.signal;
                
                const response = await fetch(url, requestConfig);
                clearTimeout(timeoutId);
                
                // Parse response
                let data;
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    data = await response.json();
                } else {
                    data = await response.text();
                }
                
                // Apply response interceptors
                let processedResponse = { response, data };
                for (const interceptor of responseInterceptors) {
                    processedResponse = await interceptor(processedResponse);
                }
                
                // Handle auth errors
                if (response.status === 401) {
                    // Try to refresh token
                    const refreshResult = await authStore.refreshToken();
                    if (refreshResult.success && attempt < maxRetries) {
                        // Retry with new token
                        await sleep(config.retryDelay);
                        continue;
                    }
                    
                    // Refresh failed - logout
                    authStore.logout();
                    throw new Error('Session expired. Please login again.');
                }
                
                // Track success
                set(state => ({
                    pendingRequests: state.pendingRequests - 1,
                    lastError: null,
                }));
                
                // Return data for successful responses
                if (response.ok) {
                    return processedResponse.data;
                }
                
                // Handle error responses
                const errorMessage = typeof processedResponse.data === 'object'
                    ? processedResponse.data.error?.message || processedResponse.data.error || 'Request failed'
                    : processedResponse.data;
                
                throw new Error(errorMessage);
                
            } catch (error) {
                lastError = error;
                
                // Don't retry on certain errors
                if (
                    error.name === 'AbortError' ||
                    error.message.includes('Session expired') ||
                    error.message.includes('Bad Request') ||
                    error.message.includes('Forbidden')
                ) {
                    break;
                }
                
                // Retry with exponential backoff
                if (attempt < maxRetries) {
                    await sleep(config.retryDelay * Math.pow(2, attempt));
                }
            }
        }
        
        // All retries failed
        set(state => ({
            pendingRequests: state.pendingRequests - 1,
            lastError: lastError?.message || 'Request failed',
        }));
        
        throw lastError;
    },
    
    /**
     * GET request
     */
    get: async (endpoint, options = {}) => {
        return get().request(endpoint, {
            ...options,
            method: 'GET',
        });
    },
    
    /**
     * POST request
     */
    post: async (endpoint, body, options = {}) => {
        return get().request(endpoint, {
            ...options,
            method: 'POST',
            body: JSON.stringify(body),
        });
    },
    
    /**
     * PUT request
     */
    put: async (endpoint, body, options = {}) => {
        return get().request(endpoint, {
            ...options,
            method: 'PUT',
            body: JSON.stringify(body),
        });
    },
    
    /**
     * PATCH request
     */
    patch: async (endpoint, body, options = {}) => {
        return get().request(endpoint, {
            ...options,
            method: 'PATCH',
            body: JSON.stringify(body),
        });
    },
    
    /**
     * DELETE request
     */
    delete: async (endpoint, options = {}) => {
        return get().request(endpoint, {
            ...options,
            method: 'DELETE',
        });
    },
    
    /**
     * Upload file
     */
    upload: async (endpoint, file, fieldName = 'file', additionalData = {}) => {
        const formData = new FormData();
        formData.append(fieldName, file);
        
        Object.entries(additionalData).forEach(([key, value]) => {
            formData.append(key, value);
        });
        
        return get().request(endpoint, {
            method: 'POST',
            body: formData,
            headers: {
                // Let browser set Content-Type with boundary for FormData
            },
        });
    },
    
    // Computed
    isLoading: () => get().pendingRequests > 0,
}));

export default useApiStore;

/**
 * Convenience hook for using API with loading state
 */
export function useApi() {
    const { get, post, put, patch, delete: del, upload, isLoading } = useApiStore();
    return { get, post, put, patch, delete: del, upload, isLoading };
}
