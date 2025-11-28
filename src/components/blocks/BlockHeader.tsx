/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

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
  AlertCircle,
  CheckCircle,
  Loader2,
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
  FlaskConical,
  Clock,
  Award,
  Repeat,
  FileSearch,
  Hash,
  Grid3x3,
  Scale,
  Binary,
  Grid2x2,
  Music,
  LayoutGrid,
  AreaChart,
  Circle,
  ScatterChart,
  Users,
  Target,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { BlockType, BlockState } from '@/types';

interface BlockHeaderProps {
  type: BlockType;
  label: string;
  state: BlockState;
  error?: string;
}

const icons: Record<string, LucideIcon> = {
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
  FlaskConical,
  Clock,
  Award,
  Repeat,
  FileSearch,
  Hash,
  Grid3x3,
  Scale,
  Binary,
  Grid2x2,
  Music,
  LayoutGrid,
  AreaChart,
  Circle,
  ScatterChart,
  Users,
  Target,
};

const iconMap: Record<BlockType, string> = {
  'load-data': 'FileUp',
  'sample-data': 'Database',
  'create-dataset': 'PenLine',
  'filter-rows': 'Filter',
  'select-columns': 'Columns',
  'sort': 'ArrowUpDown',
  'group-aggregate': 'Group',
  'join': 'GitMerge',
  'derive-column': 'Plus',
  'handle-missing': 'Eraser',
  'rename-columns': 'TextCursorInput',
  'deduplicate': 'Copy',
  'sample-rows': 'Shuffle',
  'limit-rows': 'ListFilter',
  'pivot': 'RotateCcw',
  'unpivot': 'RotateCw',
  'union': 'Layers',
  'split-column': 'Scissors',
  'merge-columns': 'Combine',
  'conditional-column': 'GitBranch',
  'statistics': 'BarChart3',
  'regression': 'TrendingUp',
  'clustering': 'Network',
  'pca': 'Minimize2',
  'outlier-detection': 'AlertTriangle',
  'classification': 'GitFork',
  'normality-test': 'Activity',
  'hypothesis-testing': 'FlaskConical',
  'time-series': 'Clock',
  'feature-importance': 'Award',
  'cross-validation': 'Repeat',
  'data-profiling': 'FileSearch',
  'value-counts': 'Hash',
  'cross-tabulation': 'Grid3x3',
  'scaling': 'Scale',
  'encoding': 'Binary',
  'ab-test': 'FlaskConical',
  'cohort-analysis': 'Users',
  'rfm-analysis': 'Target',
  'chart': 'PieChart',
  'table': 'Table',
  'correlation-matrix': 'Grid2x2',
  'violin-plot': 'Music',
  'pair-plot': 'LayoutGrid',
  'area-chart': 'AreaChart',
  'stacked-chart': 'Layers',
  'bubble-chart': 'Circle',
  'qq-plot': 'ScatterChart',
  'confusion-matrix': 'Grid3x3',
  'roc-curve': 'TrendingUp',
  'export': 'Download',
};

// Map block types to translation keys
const blockTranslationKeys: Record<BlockType, string> = {
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

export function BlockHeader({ type, label, state, error }: BlockHeaderProps) {
  const { t } = useTranslation();
  const iconName = iconMap[type];
  const IconComponent = icons[iconName] || Database;

  // Use translated label, falling back to the original label if no translation exists
  const translatedLabel = t(blockTranslationKeys[type], { defaultValue: label });

  return (
    <div className="flex items-center gap-2 px-3 py-2 border-b border-border-default">
      <IconComponent size={16} className="text-text-muted" />
      <span className="text-small font-medium text-text-primary flex-1 truncate">
        {translatedLabel}
      </span>
      {state === 'executing' && (
        <Loader2 size={14} className="animate-spin text-electric-indigo" />
      )}
      {state === 'success' && (
        <CheckCircle size={14} className="text-fresh-teal" />
      )}
      {state === 'error' && (
        <span title={error}>
          <AlertCircle size={14} className="text-warm-coral" />
        </span>
      )}
    </div>
  );
}
