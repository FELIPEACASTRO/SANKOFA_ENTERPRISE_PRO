/**
 * Sankofa Enterprise Pro - Dashboard Store
 * Zustand-based dashboard state management
 * 
 * Features:
 * - Real-time KPIs with auto-refresh
 * - Transaction timeseries data
 * - Alerts management
 * - Channel statistics
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

// Auto-refresh intervals
const KPI_REFRESH_INTERVAL_MS = 30000;       // 30 seconds
const ALERTS_REFRESH_INTERVAL_MS = 15000;    // 15 seconds
const TIMESERIES_REFRESH_INTERVAL_MS = 60000; // 1 minute

/**
 * Dashboard Store
 */
const useDashboardStore = create(
    subscribeWithSelector((set, get) => ({
        // State
        kpis: null,
        timeseries: [],
        channelStats: [],
        alerts: [],
        investigations: [],
        
        // Loading states
        isLoadingKpis: false,
        isLoadingTimeseries: false,
        isLoadingAlerts: false,
        
        // Error states
        kpisError: null,
        timeseriesError: null,
        alertsError: null,
        
        // Last update timestamps
        lastKpisUpdate: null,
        lastTimeseriesUpdate: null,
        lastAlertsUpdate: null,
        
        // Refresh intervals (store refs for cleanup)
        _refreshIntervals: {},
        
        /**
         * Fetch KPIs from API
         */
        fetchKpis: async () => {
            set({ isLoadingKpis: true, kpisError: null });
            
            try {
                const response = await fetch('/api/dashboard/kpis');
                const data = await response.json();
                
                if (!response.ok || !data.success) {
                    throw new Error(data.error?.message || 'Failed to fetch KPIs');
                }
                
                set({
                    kpis: data.data,
                    isLoadingKpis: false,
                    lastKpisUpdate: Date.now(),
                });
                
                return data.data;
            } catch (error) {
                set({
                    isLoadingKpis: false,
                    kpisError: error.message,
                });
                throw error;
            }
        },
        
        /**
         * Fetch timeseries data
         */
        fetchTimeseries: async () => {
            set({ isLoadingTimeseries: true, timeseriesError: null });
            
            try {
                const response = await fetch('/api/dashboard/timeseries');
                const data = await response.json();
                
                if (!response.ok || !data.success) {
                    throw new Error(data.error?.message || 'Failed to fetch timeseries');
                }
                
                set({
                    timeseries: data.data || [],
                    isLoadingTimeseries: false,
                    lastTimeseriesUpdate: Date.now(),
                });
                
                return data.data;
            } catch (error) {
                set({
                    isLoadingTimeseries: false,
                    timeseriesError: error.message,
                });
                throw error;
            }
        },
        
        /**
         * Fetch channel statistics
         */
        fetchChannelStats: async () => {
            try {
                const response = await fetch('/api/dashboard/channels');
                const data = await response.json();
                
                if (response.ok && data.success) {
                    set({ channelStats: data.data || [] });
                }
                
                return data.data;
            } catch (error) {
                console.error('Failed to fetch channel stats:', error);
                return [];
            }
        },
        
        /**
         * Fetch alerts
         */
        fetchAlerts: async () => {
            set({ isLoadingAlerts: true, alertsError: null });
            
            try {
                const response = await fetch('/api/dashboard/alerts');
                const data = await response.json();
                
                if (!response.ok || !data.success) {
                    throw new Error(data.error?.message || 'Failed to fetch alerts');
                }
                
                set({
                    alerts: data.data || [],
                    isLoadingAlerts: false,
                    lastAlertsUpdate: Date.now(),
                });
                
                return data.data;
            } catch (error) {
                set({
                    isLoadingAlerts: false,
                    alertsError: error.message,
                });
                throw error;
            }
        },
        
        /**
         * Fetch investigations
         */
        fetchInvestigations: async () => {
            try {
                const response = await fetch('/api/investigations');
                const data = await response.json();
                
                if (response.ok && data.success) {
                    set({ investigations: data.data || data.investigations || [] });
                }
                
                return data.data || data.investigations;
            } catch (error) {
                console.error('Failed to fetch investigations:', error);
                return [];
            }
        },
        
        /**
         * Fetch all dashboard data
         */
        fetchAll: async () => {
            const { fetchKpis, fetchTimeseries, fetchChannelStats, fetchAlerts, fetchInvestigations } = get();
            
            await Promise.all([
                fetchKpis().catch(console.error),
                fetchTimeseries().catch(console.error),
                fetchChannelStats().catch(console.error),
                fetchAlerts().catch(console.error),
                fetchInvestigations().catch(console.error),
            ]);
        },
        
        /**
         * Acknowledge an alert
         */
        acknowledgeAlert: async (alertId) => {
            try {
                const response = await fetch(`/api/alerts/${alertId}/acknowledge`, {
                    method: 'POST',
                });
                const data = await response.json();
                
                if (response.ok && data.success) {
                    // Update local state
                    set(state => ({
                        alerts: state.alerts.map(alert =>
                            alert.id === alertId
                                ? { ...alert, status: 'acknowledged' }
                                : alert
                        ),
                    }));
                }
                
                return data;
            } catch (error) {
                console.error('Failed to acknowledge alert:', error);
                throw error;
            }
        },
        
        /**
         * Resolve an alert
         */
        resolveAlert: async (alertId, resolution) => {
            try {
                const response = await fetch(`/api/alerts/${alertId}/resolve`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ resolution }),
                });
                const data = await response.json();
                
                if (response.ok && data.success) {
                    set(state => ({
                        alerts: state.alerts.filter(alert => alert.id !== alertId),
                    }));
                }
                
                return data;
            } catch (error) {
                console.error('Failed to resolve alert:', error);
                throw error;
            }
        },
        
        /**
         * Start auto-refresh for all data
         */
        startAutoRefresh: () => {
            const { fetchKpis, fetchAlerts, fetchTimeseries, _refreshIntervals } = get();
            
            // Clear existing intervals
            get().stopAutoRefresh();
            
            // Start new intervals
            const intervals = {
                kpis: setInterval(fetchKpis, KPI_REFRESH_INTERVAL_MS),
                alerts: setInterval(fetchAlerts, ALERTS_REFRESH_INTERVAL_MS),
                timeseries: setInterval(fetchTimeseries, TIMESERIES_REFRESH_INTERVAL_MS),
            };
            
            set({ _refreshIntervals: intervals });
            
            console.log('Dashboard auto-refresh started');
        },
        
        /**
         * Stop auto-refresh
         */
        stopAutoRefresh: () => {
            const { _refreshIntervals } = get();
            
            Object.values(_refreshIntervals).forEach(clearInterval);
            set({ _refreshIntervals: {} });
            
            console.log('Dashboard auto-refresh stopped');
        },
        
        /**
         * Reset dashboard state
         */
        reset: () => {
            get().stopAutoRefresh();
            
            set({
                kpis: null,
                timeseries: [],
                channelStats: [],
                alerts: [],
                investigations: [],
                isLoadingKpis: false,
                isLoadingTimeseries: false,
                isLoadingAlerts: false,
                kpisError: null,
                timeseriesError: null,
                alertsError: null,
            });
        },
        
        // Selectors/computed values
        getActiveAlerts: () => {
            const { alerts } = get();
            return alerts.filter(alert => 
                alert.status === 'open' || alert.status === 'acknowledged'
            );
        },
        
        getCriticalAlerts: () => {
            const { alerts } = get();
            return alerts.filter(alert => alert.severity === 'critical');
        },
        
        getFraudRate: () => {
            const { kpis } = get();
            if (!kpis) return 0;
            const detected = kpis.fraudes_detectadas || 0;
            const total = kpis.transacoes_hoje || 1;
            return (detected / total) * 100;
        },
    }))
);

export default useDashboardStore;
export { KPI_REFRESH_INTERVAL_MS, ALERTS_REFRESH_INTERVAL_MS, TIMESERIES_REFRESH_INTERVAL_MS };
