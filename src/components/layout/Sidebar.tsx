/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React from 'react';
import { useTranslation } from 'react-i18next';
import {
  FileUp,
  Database,
  Filter,
  Columns,
  ArrowUpDown,
  Group,
  GitMerge,
  Plus,
  Eraser,
  BarChart3,
  TrendingUp,
  Network,
  PieChart,
  Table,
  Download,
  ChevronDown,
  ChevronRight,
  PenLine,
  TextCursorInput,
  Copy,
  Shuffle,
  ListFilter,
  RotateCcw,
  RotateCw,
  Layers,
  Scissors,
  Combine,
  GitBranch,
  Minimize2,
  AlertTriangle,
  GitFork,
  Activity,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils/cn';
import { blockCategories, blockDefinitions } from '@/constants';
import type { BlockType, BlockCategory } from '@/types';
import { useUIStore } from '@/stores/uiStore';

const icons: Record<string, React.ElementType> = {
  FileUp,
  Database,
  Filter,
  Columns,
  ArrowUpDown,
  Group,
  GitMerge,
  Plus,
  Eraser,
  BarChart3,
  TrendingUp,
  Network,
  PieChart,
  Table,
  Download,
  PenLine,
  TextCursorInput,
  Copy,
  Shuffle,
  ListFilter,
  RotateCcw,
  RotateCw,
  Layers,
  Scissors,
  Combine,
  GitBranch,
  Minimize2,
  AlertTriangle,
  GitFork,
  Activity,
};

const categoryIcons: Record<string, React.ElementType> = {
  'data-input': FileUp,
  'transform': Filter,
  'analysis': BarChart3,
  'visualization': PieChart,
  'output': Download,
};

const categoryColors: Record<BlockCategory, string> = {
  'data-input': 'text-electric-indigo',
  'transform': 'text-soft-violet',
  'analysis': 'text-fresh-teal',
  'visualization': 'text-golden-amber',
  'output': 'text-warm-coral',
};

// Map category IDs to translation keys
const categoryTranslationKeys: Record<string, string> = {
  'data-input': 'categories.dataInput',
  'transform': 'categories.transform',
  'analysis': 'categories.analysis',
  'visualization': 'categories.visualization',
  'output': 'categories.output',
};

// Map block types to translation keys
const blockTranslationKeys: Record<string, string> = {
  'load-data': 'blocks.loadData',
  'sample-data': 'blocks.sampleData',
  'create-dataset': 'blocks.createDataset',
  'filter-rows': 'blocks.filterRows',
  'select-columns': 'blocks.selectColumns',
  'sort': 'blocks.sort',
  'group-aggregate': 'blocks.groupAggregate',
  'join': 'blocks.join',
  'derive-column': 'blocks.deriveColumn',
  'handle-missing': 'blocks.handleMissing',
  'rename-columns': 'blocks.renameColumns',
  'deduplicate': 'blocks.deduplicate',
  'sample-rows': 'blocks.sampleRows',
  'limit-rows': 'blocks.limitRows',
  'pivot': 'blocks.pivot',
  'unpivot': 'blocks.unpivot',
  'union': 'blocks.union',
  'split-column': 'blocks.splitColumn',
  'merge-columns': 'blocks.mergeColumns',
  'conditional-column': 'blocks.conditionalColumn',
  'statistics': 'blocks.statistics',
  'regression': 'blocks.regression',
  'clustering': 'blocks.clustering',
  'pca': 'blocks.pca',
  'outlier-detection': 'blocks.outlierDetection',
  'classification': 'blocks.classification',
  'normality-test': 'blocks.normalityTest',
  'hypothesis-testing': 'blocks.hypothesisTesting',
  'time-series': 'blocks.timeSeries',
  'feature-importance': 'blocks.featureImportance',
  'cross-validation': 'blocks.crossValidation',
  'data-profiling': 'blocks.dataProfiling',
  'value-counts': 'blocks.valueCounts',
  'cross-tabulation': 'blocks.crossTabulation',
  'scaling': 'blocks.scaling',
  'encoding': 'blocks.encoding',
  'ab-test': 'blocks.abTest',
  'cohort-analysis': 'blocks.cohortAnalysis',
  'rfm-analysis': 'blocks.rfmAnalysis',
  'chart': 'blocks.chart',
  'table': 'blocks.table',
  'correlation-matrix': 'blocks.correlationMatrix',
  'violin-plot': 'blocks.violinPlot',
  'pair-plot': 'blocks.pairPlot',
  'area-chart': 'blocks.areaChart',
  'stacked-chart': 'blocks.stackedChart',
  'bubble-chart': 'blocks.bubbleChart',
  'qq-plot': 'blocks.qqPlot',
  'confusion-matrix': 'blocks.confusionMatrix',
  'roc-curve': 'blocks.rocCurve',
  'export': 'blocks.export',
};

export function Sidebar() {
  const { t } = useTranslation();
  const { isSidebarOpen } = useUIStore();
  const [expandedCategories, setExpandedCategories] = React.useState<string[]>(
    blockCategories.map((c) => c.id)
  );

  const toggleCategory = (categoryId: string) => {
    setExpandedCategories((prev) =>
      prev.includes(categoryId)
        ? prev.filter((id) => id !== categoryId)
        : [...prev, categoryId]
    );
  };

  const onDragStart = (
    event: React.DragEvent<HTMLDivElement>,
    blockType: BlockType
  ) => {
    event.dataTransfer.setData('application/dataflow-block', blockType);
    event.dataTransfer.effectAllowed = 'move';
  };

  if (!isSidebarOpen) {
    return null;
  }

  return (
    <aside className="w-64 bg-bg-secondary border-r border-border-default flex flex-col overflow-hidden">
      <div className="p-4 border-b border-border-default">
        <h2 className="text-h3 text-text-primary">{t('sidebar.blocks')}</h2>
        <p className="text-small text-text-muted mt-1">
          {t('sidebar.dragToCanvas')}
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {blockCategories.map((category) => {
          const isExpanded = expandedCategories.includes(category.id);
          const CategoryIcon = categoryIcons[category.id] || Database;

          return (
            <div key={category.id} className="mb-2">
              <button
                onClick={() => toggleCategory(category.id)}
                className={cn(
                  'w-full flex items-center gap-2 px-3 py-2 rounded-lg',
                  'text-text-primary hover:bg-bg-tertiary transition-colors',
                  'text-left'
                )}
              >
                <CategoryIcon
                  size={16}
                  className={categoryColors[category.id as BlockCategory]}
                />
                <span className="flex-1 text-small font-medium">
                  {t(categoryTranslationKeys[category.id] || category.label)}
                </span>
                {isExpanded ? (
                  <ChevronDown size={16} className="text-text-muted" />
                ) : (
                  <ChevronRight size={16} className="text-text-muted" />
                )}
              </button>

              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="pl-4 pr-2 py-1 space-y-1">
                      {category.blocks.map((blockType) => {
                        const definition = blockDefinitions[blockType];
                        const IconComponent =
                          icons[definition.icon] || Database;

                        return (
                          <div
                            key={blockType}
                            draggable
                            onDragStart={(e) => onDragStart(e, blockType)}
                            className={cn(
                              'flex items-center gap-2 px-3 py-2 rounded-lg',
                              'bg-bg-tertiary hover:bg-border-default',
                              'cursor-grab active:cursor-grabbing',
                              'transition-colors duration-100'
                            )}
                          >
                            <IconComponent
                              size={14}
                              className={
                                categoryColors[category.id as BlockCategory]
                              }
                            />
                            <div className="flex-1 min-w-0">
                              <p className="text-small text-text-primary truncate">
                                {t(blockTranslationKeys[blockType] || definition.label)}
                              </p>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>
    </aside>
  );
}
