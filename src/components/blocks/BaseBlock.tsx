/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils/cn';
import type { BlockData, BlockCategory, BlockState } from '@/types';
import { BlockHeader } from './BlockHeader';
import { BlockBody } from './BlockBody';

interface BaseBlockProps {
  data: BlockData;
  selected?: boolean;
}

const categoryColors: Record<BlockCategory, string> = {
  'data-input': 'border-l-electric-indigo',
  'transform': 'border-l-soft-violet',
  'analysis': 'border-l-fresh-teal',
  'visualization': 'border-l-golden-amber',
  'output': 'border-l-warm-coral',
};

const stateStyles: Record<BlockState, string> = {
  idle: 'border-border-default shadow-sm',
  selected: 'border-electric-indigo shadow-glow',
  executing: 'border-electric-indigo animate-pulse-border',
  success: 'border-fresh-teal shadow-glow-teal',
  error: 'border-warm-coral shadow-glow-coral',
};

export const BaseBlock = memo(({ data, selected }: BaseBlockProps) => {
  const { type, category, label, state, error } = data;

  return (
    <motion.div
      className={cn(
        'min-w-[180px] bg-bg-secondary rounded-xl',
        'border border-l-4',
        categoryColors[category],
        stateStyles[selected ? 'selected' : state],
        'transition-all duration-200'
      )}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
    >
      {/* Input Handle */}
      {category !== 'data-input' && (
        <Handle
          type="target"
          position={Position.Left}
          className={cn(
            'w-3 h-3 !bg-fresh-teal border-2 border-bg-primary',
            'hover:!bg-electric-indigo hover:scale-125 transition-transform'
          )}
        />
      )}

      <BlockHeader type={type} label={label} state={state} error={error} />
      <BlockBody type={type} data={data} />

      {/* Output Handle */}
      {category !== 'output' && (
        <Handle
          type="source"
          position={Position.Right}
          className={cn(
            'w-3 h-3 !bg-fresh-teal border-2 border-bg-primary',
            'hover:!bg-electric-indigo hover:scale-125 transition-transform'
          )}
        />
      )}
    </motion.div>
  );
});

BaseBlock.displayName = 'BaseBlock';
