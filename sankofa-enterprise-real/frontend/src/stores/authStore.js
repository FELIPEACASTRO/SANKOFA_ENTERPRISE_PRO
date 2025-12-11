/**
 * Sankofa Enterprise Pro - Auth Store
 * Zustand-based authentication state management
 * 
 * Features:
 * - JWT token management with auto-refresh
 * - User session persistence
 * - Role-based access control helpers
 * - Secure token storage
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

// Token refresh threshold (5 minutes before expiration)
const REFRESH_THRESHOLD_MS = 5 * 60 * 1000;

/**
 * Parse JWT token to extract payload
 */
function parseJwt(token) {
    try {
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(
            atob(base64)
                .split('')
                .map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
                .join('')
        );
        return JSON.parse(jsonPayload);
    } catch (e) {
        console.error('Failed to parse JWT:', e);
        return null;
    }
}

/**
 * Check if token is expired or about to expire
 */
function isTokenExpiring(token, thresholdMs = REFRESH_THRESHOLD_MS) {
    if (!token) return true;
    
    const payload = parseJwt(token);
    if (!payload || !payload.exp) return true;
    
    const expiresAt = payload.exp * 1000; // Convert to milliseconds
    const now = Date.now();
    
    return now >= (expiresAt - thresholdMs);
}

/**
 * Auth Store Interface
 */
const useAuthStore = create(
    persist(
        (set, get) => ({
            // State
            token: null,
            user: null,
            isAuthenticated: false,
            isLoading: false,
            error: null,
            lastActivity: null,
            
            // Actions
            setAuth: (token, user) => {
                set({
                    token,
                    user,
                    isAuthenticated: true,
                    error: null,
                    lastActivity: Date.now(),
                });
            },
            
            clearAuth: () => {
                set({
                    token: null,
                    user: null,
                    isAuthenticated: false,
                    error: null,
                });
            },
            
            setLoading: (isLoading) => set({ isLoading }),
            setError: (error) => set({ error }),
            updateLastActivity: () => set({ lastActivity: Date.now() }),
            
            /**
             * Login action
             */
            login: async (username, password) => {
                set({ isLoading: true, error: null });
                
                try {
                    const response = await fetch('/api/auth/login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ username, password }),
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok || !data.success) {
                        throw new Error(data.error?.message || 'Login failed');
                    }
                    
                    const { token, user } = data.data;
                    
                    set({
                        token,
                        user,
                        isAuthenticated: true,
                        isLoading: false,
                        error: null,
                        lastActivity: Date.now(),
                    });
                    
                    return { success: true, user };
                } catch (error) {
                    set({
                        isLoading: false,
                        error: error.message,
                        isAuthenticated: false,
                    });
                    return { success: false, error: error.message };
                }
            },
            
            /**
             * Logout action
             */
            logout: () => {
                set({
                    token: null,
                    user: null,
                    isAuthenticated: false,
                    error: null,
                    lastActivity: null,
                });
            },
            
            /**
             * Refresh token if needed
             */
            refreshToken: async () => {
                const { token } = get();
                
                if (!token || !isTokenExpiring(token)) {
                    return { success: true };
                }
                
                try {
                    const response = await fetch('/api/auth/refresh', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${token}`,
                        },
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok || !data.success) {
                        throw new Error(data.error?.message || 'Token refresh failed');
                    }
                    
                    set({
                        token: data.data.token,
                        lastActivity: Date.now(),
                    });
                    
                    return { success: true };
                } catch (error) {
                    // Token refresh failed - logout
                    get().logout();
                    return { success: false, error: error.message };
                }
            },
            
            /**
             * Verify current token
             */
            verifyToken: async () => {
                const { token } = get();
                
                if (!token) {
                    return { success: false, error: 'No token' };
                }
                
                try {
                    const response = await fetch('/api/auth/verify', {
                        headers: {
                            'Authorization': `Bearer ${token}`,
                        },
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok || !data.valid) {
                        get().logout();
                        return { success: false, error: 'Invalid token' };
                    }
                    
                    return { success: true, user: data.data.user };
                } catch (error) {
                    return { success: false, error: error.message };
                }
            },
            
            // Computed getters
            getToken: () => get().token,
            getUser: () => get().user,
            
            /**
             * Check if user has specific permission
             */
            hasPermission: (permission) => {
                const { user } = get();
                if (!user) return false;
                
                const roles = user.roles || [user.role];
                
                // Admin has all permissions
                if (roles.includes('admin')) return true;
                
                // Check role-based permissions
                const rolePermissions = {
                    analyst: [
                        'fraud:view', 'fraud:predict', 'fraud:explain', 'fraud:feedback',
                        'transactions:view', 'transactions:search',
                        'alerts:view', 'alerts:acknowledge', 'alerts:update',
                        'reports:view', 'reports:generate',
                        'dashboard:view', 'metrics:view', 'model:view',
                        'investigation:view', 'audit:view', 'observability:view',
                    ],
                    operator: [
                        'fraud:view', 'fraud:predict', 'transactions:view',
                        'alerts:view', 'dashboard:view', 'metrics:view', 'observability:view',
                    ],
                    viewer: ['dashboard:view', 'metrics:view', 'transactions:view', 'alerts:view'],
                };
                
                for (const role of roles) {
                    const perms = rolePermissions[role] || [];
                    if (perms.includes(permission)) return true;
                    
                    // Check category wildcard
                    const category = permission.split(':')[0] + ':*';
                    if (perms.includes(category)) return true;
                }
                
                return false;
            },
            
            /**
             * Check if user has any of the specified roles
             */
            hasRole: (...requiredRoles) => {
                const { user } = get();
                if (!user) return false;
                
                const userRoles = user.roles || [user.role];
                return requiredRoles.some(role => userRoles.includes(role));
            },
        }),
        {
            name: 'sankofa-auth-storage',
            storage: createJSONStorage(() => localStorage),
            partialize: (state) => ({
                token: state.token,
                user: state.user,
                isAuthenticated: state.isAuthenticated,
            }),
        }
    )
);

export default useAuthStore;
export { parseJwt, isTokenExpiring };
