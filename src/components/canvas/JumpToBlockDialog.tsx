/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { useReactFlow } from '@xyflow/react';
import { Search } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/Dialog';
import { Input } from '@/components/ui/Input';
import { useUIStore } from '@/stores/uiStore';
import { useCanvasStore } from '@/stores/canvasStore';
import { cn } from '@/lib/utils/cn';
import type { BlockData } from '@/types';

const categoryColors: Record<string, string> = {
  'data-input': 'bg-plasma-magenta/20 text-plasma-magenta border-plasma-magenta/30',
  'transform': 'bg-plasma-purple/20 text-plasma-purple border-plasma-purple/30',
  'analysis': 'bg-plasma-cyan/20 text-plasma-cyan border-plasma-cyan/30',
  'visualization': 'bg-plasma-yellow/20 text-plasma-yellow border-plasma-yellow/30',
  'output': 'bg-warm-coral/20 text-warm-coral border-warm-coral/30',
};

export function JumpToBlockDialog() {
  const { t } = useTranslation();
  const { fitView } = useReactFlow();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const isOpen = useUIStore((state) => state.isJumpToBlockOpen);
  const closeDialog = useUIStore((state) => state.closeJumpToBlock);
  const blocks = useCanvasStore((state) => state.blocks);
  const setSelectedBlocks = useCanvasStore((state) => state.setSelectedBlocks);

  // Filter blocks based on search query
  const filteredBlocks = useMemo(() => {
    if (!searchQuery.trim()) {
      return blocks;
    }
    const query = searchQuery.toLowerCase();
    return blocks.filter((block) => {
      const data = block.data as BlockData;
      return (
        data.label.toLowerCase().includes(query) ||
        data.type.toLowerCase().includes(query) ||
        data.category.toLowerCase().includes(query)
      );
    });
  }, [blocks, searchQuery]);

  // Reset selected index when filtered results change
  useEffect(() => {
    setSelectedIndex(0);
  }, [filteredBlocks.length]);

  // Focus input when dialog opens
  useEffect(() => {
    if (isOpen) {
      setSearchQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [isOpen]);

  // Scroll selected item into view
  useEffect(() => {
    if (listRef.current && filteredBlocks.length > 0) {
      const selectedElement = listRef.current.children[selectedIndex] as HTMLElement;
      if (selectedElement) {
        selectedElement.scrollIntoView({ block: 'nearest' });
      }
    }
  }, [selectedIndex, filteredBlocks.length]);

  const jumpToBlock = useCallback(
    (blockId: string) => {
      // Select the block
      setSelectedBlocks([blockId]);

      // Animate to the block
      fitView({
        nodes: [{ id: blockId }],
        duration: 300,
        padding: 0.5,
      });

      // Close the dialog
      closeDialog();
    },
    [fitView, setSelectedBlocks, closeDialog]
  );

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      switch (event.key) {
        case 'ArrowDown':
          event.preventDefault();
          setSelectedIndex((prev) =>
            prev < filteredBlocks.length - 1 ? prev + 1 : prev
          );
          break;
        case 'ArrowUp':
          event.preventDefault();
          setSelectedIndex((prev) => (prev > 0 ? prev - 1 : 0));
          break;
        case 'Enter':
          event.preventDefault();
          if (filteredBlocks[selectedIndex]) {
            jumpToBlock(filteredBlocks[selectedIndex].id);
          }
          break;
        case 'Escape':
          event.preventDefault();
          closeDialog();
          break;
      }
    },
    [filteredBlocks, selectedIndex, jumpToBlock, closeDialog]
  );

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && closeDialog()}>
      <DialogContent className="max-w-md" onKeyDown={handleKeyDown}>
        <DialogHeader>
          <DialogTitle>{t('jumpToBlock.title', 'Jump to Block')}</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <Input
            ref={inputRef}
            placeholder={t('jumpToBlock.searchPlaceholder', 'Search blocks by name or type...')}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            leftIcon={<Search size={16} />}
          />

          <div
            ref={listRef}
            className="max-h-64 overflow-y-auto space-y-1"
          >
            {filteredBlocks.length === 0 ? (
              <p className="text-text-muted text-center py-4">
                {blocks.length === 0
                  ? t('jumpToBlock.noBlocks', 'No blocks on canvas')
                  : t('jumpToBlock.noResults', 'No matching blocks found')}
              </p>
            ) : (
              filteredBlocks.map((block, index) => {
                const data = block.data as BlockData;
                const isSelected = index === selectedIndex;
                return (
                  <button
                    key={block.id}
                    className={cn(
                      'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left',
                      'transition-colors duration-100',
                      isSelected
                        ? 'bg-electric-indigo/20 border border-electric-indigo/50'
                        : 'hover:bg-bg-tertiary border border-transparent'
                    )}
                    onClick={() => jumpToBlock(block.id)}
                    onMouseEnter={() => setSelectedIndex(index)}
                  >
                    <span
                      className={cn(
                        'px-2 py-0.5 text-xs rounded border',
                        categoryColors[data.category] || 'bg-bg-tertiary text-text-secondary'
                      )}
                    >
                      {data.category}
                    </span>
                    <span className="text-text-primary font-medium truncate flex-1">
                      {data.label}
                    </span>
                  </button>
                );
              })
            )}
          </div>

          <div className="flex items-center justify-between text-xs text-text-muted border-t border-border-default pt-3">
            <span>{t('jumpToBlock.hint', 'Use arrow keys to navigate, Enter to select')}</span>
            <span className="text-text-secondary">
              {filteredBlocks.length} {t('jumpToBlock.blocks', 'block(s)')}
            </span>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
