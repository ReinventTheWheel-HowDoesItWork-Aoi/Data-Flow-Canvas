/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React from 'react';
import * as SelectPrimitive from '@radix-ui/react-select';
import { ChevronDown, Check } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

export const Select = SelectPrimitive.Root;
export const SelectGroup = SelectPrimitive.Group;
export const SelectValue = SelectPrimitive.Value;

interface SelectTriggerProps extends SelectPrimitive.SelectTriggerProps {
  label?: string;
}

export const SelectTrigger = React.forwardRef<
  HTMLButtonElement,
  SelectTriggerProps
>(({ className, children, label, ...props }, ref) => (
  <div className="w-full">
    {label && (
      <label className="block text-small font-medium text-text-secondary mb-1.5">
        {label}
      </label>
    )}
    <SelectPrimitive.Trigger
      ref={ref}
      className={cn(
        'flex h-10 w-full items-center justify-between rounded-lg',
        'border border-border-default bg-bg-secondary px-3 py-2',
        'text-text-primary placeholder:text-text-muted',
        'focus:outline-none focus:ring-2 focus:ring-electric-indigo focus:border-transparent',
        'disabled:cursor-not-allowed disabled:opacity-50',
        className
      )}
      {...props}
    >
      {children}
      <SelectPrimitive.Icon asChild>
        <ChevronDown className="h-4 w-4 text-text-muted" />
      </SelectPrimitive.Icon>
    </SelectPrimitive.Trigger>
  </div>
));
SelectTrigger.displayName = 'SelectTrigger';

export const SelectContent = React.forwardRef<
  HTMLDivElement,
  SelectPrimitive.SelectContentProps
>(({ className, children, position = 'popper', ...props }, ref) => (
  <SelectPrimitive.Portal>
    <SelectPrimitive.Content
      ref={ref}
      className={cn(
        'relative z-50 min-w-[8rem] overflow-hidden rounded-lg',
        'border border-border-default bg-bg-secondary shadow-lg',
        'data-[state=open]:animate-fade-in',
        position === 'popper' &&
          'data-[side=bottom]:translate-y-1 data-[side=left]:-translate-x-1 data-[side=right]:translate-x-1 data-[side=top]:-translate-y-1',
        className
      )}
      position={position}
      {...props}
    >
      <SelectPrimitive.Viewport
        className={cn(
          'p-1',
          position === 'popper' &&
            'h-[var(--radix-select-trigger-height)] w-full min-w-[var(--radix-select-trigger-width)]'
        )}
      >
        {children}
      </SelectPrimitive.Viewport>
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>
));
SelectContent.displayName = 'SelectContent';

export const SelectLabel = React.forwardRef<
  HTMLDivElement,
  SelectPrimitive.SelectLabelProps
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Label
    ref={ref}
    className={cn('py-1.5 pl-8 pr-2 text-small font-medium text-text-muted', className)}
    {...props}
  />
));
SelectLabel.displayName = 'SelectLabel';

export const SelectItem = React.forwardRef<
  HTMLDivElement,
  SelectPrimitive.SelectItemProps
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Item
    ref={ref}
    className={cn(
      'relative flex w-full cursor-pointer select-none items-center rounded-md py-1.5 pl-8 pr-2',
      'text-text-primary outline-none',
      'focus:bg-electric-indigo/10 focus:text-electric-indigo',
      'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
      className
    )}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <SelectPrimitive.ItemIndicator>
        <Check className="h-4 w-4" />
      </SelectPrimitive.ItemIndicator>
    </span>
    <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
  </SelectPrimitive.Item>
));
SelectItem.displayName = 'SelectItem';

export const SelectSeparator = React.forwardRef<
  HTMLDivElement,
  SelectPrimitive.SelectSeparatorProps
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Separator
    ref={ref}
    className={cn('-mx-1 my-1 h-px bg-border-default', className)}
    {...props}
  />
));
SelectSeparator.displayName = 'SelectSeparator';
