/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React from 'react';
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import { cn } from '@/lib/utils/cn';

export const TooltipProvider = TooltipPrimitive.Provider;
export const Tooltip = TooltipPrimitive.Root;
export const TooltipTrigger = TooltipPrimitive.Trigger;

export function TooltipContent({
  className,
  sideOffset = 4,
  children,
  ...props
}: TooltipPrimitive.TooltipContentProps) {
  return (
    <TooltipPrimitive.Portal>
      <TooltipPrimitive.Content
        sideOffset={sideOffset}
        className={cn(
          'z-50 px-3 py-1.5 text-small rounded-lg',
          'bg-deep-navy text-white shadow-md',
          'animate-fade-in',
          className
        )}
        {...props}
      >
        {children}
        <TooltipPrimitive.Arrow className="fill-deep-navy" />
      </TooltipPrimitive.Content>
    </TooltipPrimitive.Portal>
  );
}
