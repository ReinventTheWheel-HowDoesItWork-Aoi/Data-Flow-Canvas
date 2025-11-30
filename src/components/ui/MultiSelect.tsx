/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown, Check, X } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface MultiSelectOption {
  value: string;
  label: string;
}

interface MultiSelectProps {
  label?: string;
  options: MultiSelectOption[];
  selected: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}

export const MultiSelect: React.FC<MultiSelectProps> = ({
  label,
  options,
  selected,
  onChange,
  placeholder = 'Select items...',
  className,
  disabled = false,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleToggle = (value: string) => {
    if (selected.includes(value)) {
      onChange(selected.filter((v) => v !== value));
    } else {
      onChange([...selected, value]);
    }
  };

  const handleRemove = (value: string, e: React.MouseEvent) => {
    e.stopPropagation();
    onChange(selected.filter((v) => v !== value));
  };

  const selectedLabels = selected
    .map((v) => options.find((o) => o.value === v)?.label || v)
    .slice(0, 3);

  return (
    <div className={cn('w-full', className)} ref={containerRef}>
      {label && (
        <label className="block text-small font-medium text-text-secondary mb-1.5">
          {label}
        </label>
      )}
      <div className="relative">
        <button
          type="button"
          onClick={() => !disabled && setIsOpen(!isOpen)}
          className={cn(
            'flex min-h-10 w-full items-center justify-between rounded-lg',
            'border border-border-default bg-bg-secondary px-3 py-2',
            'text-text-primary placeholder:text-text-muted',
            'focus:outline-none focus:ring-2 focus:ring-electric-indigo focus:border-transparent',
            'disabled:cursor-not-allowed disabled:opacity-50',
            isOpen && 'ring-2 ring-electric-indigo border-transparent'
          )}
          disabled={disabled}
        >
          <div className="flex flex-wrap gap-1 flex-1 text-left">
            {selected.length === 0 ? (
              <span className="text-text-muted">{placeholder}</span>
            ) : selected.length <= 3 ? (
              selected.map((value) => {
                const option = options.find((o) => o.value === value);
                return (
                  <span
                    key={value}
                    className="inline-flex items-center gap-1 px-2 py-0.5 bg-electric-indigo/20 text-electric-indigo rounded text-small"
                  >
                    {option?.label || value}
                    <X
                      className="h-3 w-3 cursor-pointer hover:text-electric-indigo/70"
                      onClick={(e) => handleRemove(value, e)}
                    />
                  </span>
                );
              })
            ) : (
              <span className="text-text-primary">
                {selected.length} items selected
              </span>
            )}
          </div>
          <ChevronDown
            className={cn(
              'h-4 w-4 text-text-muted transition-transform ml-2 flex-shrink-0',
              isOpen && 'rotate-180'
            )}
          />
        </button>

        {isOpen && (
          <div
            className={cn(
              'absolute z-50 mt-1 w-full max-h-60 overflow-auto rounded-lg',
              'border border-border-default bg-bg-secondary shadow-lg',
              'animate-fade-in'
            )}
          >
            <div className="p-1">
              {options.length === 0 ? (
                <div className="py-2 px-3 text-text-muted text-small">
                  No options available
                </div>
              ) : (
                options.map((option) => {
                  const isSelected = selected.includes(option.value);
                  return (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => handleToggle(option.value)}
                      className={cn(
                        'relative flex w-full cursor-pointer select-none items-center rounded-md py-1.5 pl-8 pr-2',
                        'text-text-primary outline-none text-left',
                        'hover:bg-electric-indigo/10 hover:text-electric-indigo',
                        isSelected && 'bg-electric-indigo/5'
                      )}
                    >
                      <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
                        {isSelected && <Check className="h-4 w-4 text-electric-indigo" />}
                      </span>
                      {option.label}
                    </button>
                  );
                })
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

MultiSelect.displayName = 'MultiSelect';
