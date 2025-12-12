import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from '@/components/ui/Button';

describe('Button Component', () => {
  it('renderiza com texto correto', () => {
    render(<Button>Clique aqui</Button>);
    expect(screen.getByRole('button')).toHaveTextContent('Clique aqui');
  });

  it('aplica variante primary por padrão', () => {
    render(<Button>Primary</Button>);
    const button = screen.getByRole('button');
    expect(button).toHaveClass('bg-[var(--color-cta)]');
  });

  it('aplica variante secondary corretamente', () => {
    render(<Button variant="secondary">Secondary</Button>);
    const button = screen.getByRole('button');
    expect(button).toHaveClass('bg-[var(--neutral-100)]');
  });

  it('aplica variante danger corretamente', () => {
    render(<Button variant="danger">Danger</Button>);
    const button = screen.getByRole('button');
    expect(button).toHaveClass('bg-[var(--error-500)]');
  });

  it('desabilita botão quando disabled=true', () => {
    render(<Button disabled>Disabled</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });

  it('desabilita botão quando loading=true', () => {
    render(<Button loading>Loading</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
    expect(screen.getByRole('button')).toHaveAttribute('aria-busy', 'true');
  });

  it('mostra spinner quando loading', () => {
    render(<Button loading>Loading</Button>);
    expect(screen.getByRole('button').querySelector('.animate-spin')).toBeInTheDocument();
  });

  it('chama onClick quando clicado', () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>Click</Button>);
    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('não chama onClick quando disabled', () => {
    const handleClick = vi.fn();
    render(<Button disabled onClick={handleClick}>Click</Button>);
    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).not.toHaveBeenCalled();
  });

  it('aplica tamanhos corretamente', () => {
    const { rerender } = render(<Button size="sm">Small</Button>);
    expect(screen.getByRole('button')).toHaveClass('h-8');

    rerender(<Button size="md">Medium</Button>);
    expect(screen.getByRole('button')).toHaveClass('h-10');

    rerender(<Button size="lg">Large</Button>);
    expect(screen.getByRole('button')).toHaveClass('h-12');
  });

  it('possui focus ring para acessibilidade', () => {
    render(<Button>Focus Test</Button>);
    const button = screen.getByRole('button');
    expect(button).toHaveClass('focus-visible:ring-4');
  });
});
