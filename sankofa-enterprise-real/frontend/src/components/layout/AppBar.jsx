import { useState } from 'react';
import { Search, Bell, User, Moon, Sun, Menu } from 'lucide-react';
import { Button } from '@/components/ui/Button.jsx';
import { Input } from '@/components/ui/Input.jsx';
import { useTheme } from '@/providers/ThemeProvider';
import { cn } from '@/lib/utils';

export function AppBar({ onMenuToggle, className }) {
  const [searchQuery, setSearchQuery] = useState('');
  const { theme, toggleTheme } = useTheme();

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      // Implementar busca
      console.log('Buscar:', searchQuery);
    }
  };

  return (
    <header className={cn(
      'sticky top-0 z-50 w-full border-b border-[var(--color-border)] bg-[var(--color-surface)]/95 backdrop-blur supports-[backdrop-filter]:bg-[var(--color-surface)]/60',
      className
    )}>
      <div className="container flex h-16 items-center px-4">
        {/* Menu Toggle (Mobile) */}
        <Button
          variant="ghost"
          size="sm"
          className="mr-2 md:hidden"
          onClick={onMenuToggle}
          aria-label="Abrir menu de navegação"
        >
          <Menu className="h-5 w-5" />
        </Button>

        {/* Logo */}
        <div className="flex items-center space-x-3">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-xl blur-sm"></div>
            <img 
              src="/sankofa-logo.png" 
              alt="Sankofa Enterprise Pro" 
              className="relative h-12 w-12 object-contain drop-shadow-lg"
            />
          </div>
          <div className="hidden sm:block">
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Sankofa
              </h1>
              <span className="px-2 py-0.5 text-[10px] font-bold bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-full uppercase tracking-wide">
                Enterprise
              </span>
            </div>
            <p className="text-xs text-[var(--color-text-secondary)] font-medium">
              Sistema de Detecção de Fraudes Bancárias
            </p>
          </div>
        </div>

        {/* Search */}
        <div className="flex-1 max-w-md mx-4">
          <form onSubmit={handleSearch} className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--color-text-secondary)]" />
            <Input
              type="search"
              placeholder="Buscar transações, CPF, ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 pr-4"
              aria-label="Buscar transações"
            />
          </form>
        </div>

        {/* Actions */}
        <div className="flex items-center space-x-2">
          {/* Theme Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleTheme}
            aria-label={`Alternar para tema ${theme === 'light' ? 'escuro' : 'claro'}`}
          >
            {theme === 'light' ? (
              <Moon className="h-5 w-5" />
            ) : (
              <Sun className="h-5 w-5" />
            )}
          </Button>

          {/* Notifications */}
          <Button
            variant="ghost"
            size="sm"
            className="relative"
            aria-label="Notificações"
          >
            <Bell className="h-5 w-5" />
            <span className="absolute -top-1 -right-1 h-3 w-3 rounded-full bg-[var(--error-500)] text-[10px] text-white flex items-center justify-center">
              3
            </span>
          </Button>

          {/* User Menu */}
          <Button
            variant="ghost"
            size="sm"
            className="flex items-center space-x-2"
            aria-label="Menu do usuário"
          >
            <div className="h-6 w-6 rounded-full bg-[var(--color-brand)] flex items-center justify-center text-white text-xs">
              A
            </div>
            <span className="hidden md:inline text-sm">Analista</span>
          </Button>
        </div>
      </div>
    </header>
  );
}

