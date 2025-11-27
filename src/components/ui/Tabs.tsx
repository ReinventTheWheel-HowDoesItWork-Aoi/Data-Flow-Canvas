/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React from 'react';
import * as TabsPrimitive from '@radix-ui/react-tabs';
import { cn } from '@/lib/utils/cn';

export const Tabs = TabsPrimitive.Root;

export const TabsList = React.forwardRef<
  HTMLDivElement,
  TabsPrimitive.TabsListProps
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn(
      'inline-flex items-center gap-1 rounded-lg bg-bg-tertiary p-1',
      className
    )}
    {...props}
  />
));
TabsList.displayName = 'TabsList';

export const TabsTrigger = React.forwardRef<
  HTMLButtonElement,
  TabsPrimitive.TabsTriggerProps
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      'inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1.5',
      'text-small font-medium text-text-secondary',
      'ring-offset-bg-primary transition-all',
      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-electric-indigo focus-visible:ring-offset-2',
      'disabled:pointer-events-none disabled:opacity-50',
      'data-[state=active]:bg-bg-secondary data-[state=active]:text-text-primary data-[state=active]:shadow-sm',
      className
    )}
    {...props}
  />
));
TabsTrigger.displayName = 'TabsTrigger';

export const TabsContent = React.forwardRef<
  HTMLDivElement,
  TabsPrimitive.TabsContentProps
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content
    ref={ref}
    className={cn(
      'mt-2 ring-offset-bg-primary',
      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-electric-indigo focus-visible:ring-offset-2',
      'data-[state=active]:animate-fade-in',
      className
    )}
    {...props}
  />
));
TabsContent.displayName = 'TabsContent';
