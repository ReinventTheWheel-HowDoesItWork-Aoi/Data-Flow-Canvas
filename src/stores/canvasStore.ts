/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { create } from 'zustand';
import { temporal } from 'zundo';
import { v4 as uuidv4 } from 'uuid';
import type { PipelineBlock, PipelineEdge, BlockData, BlockState, BlockType, BlockCategory } from '@/types';

interface CanvasState {
  blocks: PipelineBlock[];
  edges: PipelineEdge[];
  selectedBlockIds: string[];
  viewport: { x: number; y: number; zoom: number };

  // Actions
  addBlock: (type: BlockType, position: { x: number; y: number }) => string;
  addBlockDirect: (block: PipelineBlock) => void;
  updateBlock: (id: string, data: Partial<BlockData>) => void;
  removeBlock: (id: string) => void;
  setBlockState: (id: string, state: BlockState) => void;

  addEdge: (source: string, target: string, sourceHandle?: string, targetHandle?: string) => void;
  addEdgeDirect: (edge: PipelineEdge) => void;
  removeEdge: (id: string) => void;

  setBlocks: (blocks: PipelineBlock[]) => void;
  setEdges: (edges: PipelineEdge[]) => void;

  setSelectedBlocks: (ids: string[]) => void;
  clearSelection: () => void;

  setViewport: (viewport: { x: number; y: number; zoom: number }) => void;

  importPipeline: (blocks: PipelineBlock[], edges: PipelineEdge[]) => void;
  clearCanvas: () => void;
}

function getBlockCategory(type: BlockType): BlockCategory {
  const categories: Record<BlockType, BlockCategory> = {
    'load-data': 'data-input',
    'sample-data': 'data-input',
    'create-dataset': 'data-input',
    'filter-rows': 'transform',
    'select-columns': 'transform',
    'sort': 'transform',
    'group-aggregate': 'transform',
    'join': 'transform',
    'derive-column': 'transform',
    'handle-missing': 'transform',
    'rename-columns': 'transform',
    'deduplicate': 'transform',
    'sample-rows': 'transform',
    'limit-rows': 'transform',
    'pivot': 'transform',
    'unpivot': 'transform',
    'union': 'transform',
    'split-column': 'transform',
    'merge-columns': 'transform',
    'conditional-column': 'transform',
    'datetime-extract': 'transform',
    'string-operations': 'transform',
    'window-functions': 'transform',
    'bin-bucket': 'transform',
    'rank': 'transform',
    'type-conversion': 'transform',
    'fill-forward-backward': 'transform',
    'lag-lead': 'transform',
    'row-number': 'transform',
    'date-difference': 'transform',
    'transpose': 'transform',
    'string-pad': 'transform',
    'cumulative-operations': 'transform',
    'replace-values': 'transform',
    'percent-change': 'transform',
    'round-numbers': 'transform',
    'percent-of-total': 'transform',
    'absolute-value': 'transform',
    'column-math': 'transform',
    'extract-substring': 'transform',
    'parse-date': 'transform',
    'split-to-rows': 'transform',
    'clip-values': 'transform',
    'standardize-text': 'transform',
    'case-when': 'transform',
    'statistics': 'analysis',
    'regression': 'analysis',
    'clustering': 'analysis',
    'pca': 'analysis',
    'outlier-detection': 'analysis',
    'classification': 'analysis',
    'normality-test': 'analysis',
    'hypothesis-testing': 'analysis',
    'time-series': 'analysis',
    'feature-importance': 'analysis',
    'cross-validation': 'analysis',
    'data-profiling': 'analysis',
    'value-counts': 'analysis',
    'cross-tabulation': 'analysis',
    'scaling': 'analysis',
    'encoding': 'analysis',
    'ab-test': 'analysis',
    'cohort-analysis': 'analysis',
    'rfm-analysis': 'analysis',
    'anova': 'analysis',
    'chi-square-test': 'analysis',
    'correlation-analysis': 'analysis',
    'survival-analysis': 'analysis',
    'association-rules': 'analysis',
    'sentiment-analysis': 'analysis',
    'moving-average': 'analysis',
    'train-test-split': 'analysis',
    'model-evaluation': 'analysis',
    'knn': 'analysis',
    'naive-bayes': 'analysis',
    'gradient-boosting': 'analysis',
    'pareto-analysis': 'analysis',
    'trend-analysis': 'analysis',
    'forecasting': 'analysis',
    'percentile-analysis': 'analysis',
    'distribution-fit': 'analysis',
    'text-preprocessing': 'analysis',
    'tfidf-vectorization': 'analysis',
    'topic-modeling': 'analysis',
    'similarity-analysis': 'analysis',
    'svm': 'analysis',
    'xgboost': 'analysis',
    'model-explainability': 'analysis',
    'regression-diagnostics': 'analysis',
    'vif-analysis': 'analysis',
    'funnel-analysis': 'analysis',
    'customer-ltv': 'analysis',
    'churn-analysis': 'analysis',
    'growth-metrics': 'analysis',
    'attribution-modeling': 'analysis',
    'breakeven-analysis': 'analysis',
    'confidence-intervals': 'analysis',
    'bootstrap-analysis': 'analysis',
    'posthoc-tests': 'analysis',
    'power-analysis': 'analysis',
    'bayesian-inference': 'analysis',
    'data-quality-score': 'analysis',
    'changepoint-detection': 'analysis',
    'chart': 'visualization',
    'table': 'visualization',
    'correlation-matrix': 'visualization',
    'violin-plot': 'visualization',
    'pair-plot': 'visualization',
    'area-chart': 'visualization',
    'stacked-chart': 'visualization',
    'bubble-chart': 'visualization',
    'qq-plot': 'visualization',
    'confusion-matrix': 'visualization',
    'roc-curve': 'visualization',
    'export': 'output',
  };
  return categories[type];
}

function getBlockLabel(type: BlockType): string {
  const labels: Record<BlockType, string> = {
    'load-data': 'Load Data',
    'sample-data': 'Sample Data',
    'create-dataset': 'Create Dataset',
    'filter-rows': 'Filter Rows',
    'select-columns': 'Select Columns',
    'sort': 'Sort',
    'group-aggregate': 'Group & Aggregate',
    'join': 'Join',
    'derive-column': 'Derive Column',
    'handle-missing': 'Handle Missing',
    'rename-columns': 'Rename Columns',
    'deduplicate': 'Deduplicate',
    'sample-rows': 'Sample Rows',
    'limit-rows': 'Limit Rows',
    'pivot': 'Pivot',
    'unpivot': 'Unpivot',
    'union': 'Union',
    'split-column': 'Split Column',
    'merge-columns': 'Merge Columns',
    'conditional-column': 'Conditional Column',
    'datetime-extract': 'Date/Time Extract',
    'string-operations': 'String Operations',
    'window-functions': 'Window Functions',
    'bin-bucket': 'Bin/Bucket',
    'rank': 'Rank',
    'type-conversion': 'Type Conversion',
    'fill-forward-backward': 'Fill Forward/Backward',
    'lag-lead': 'Lag/Lead',
    'row-number': 'Row Number',
    'date-difference': 'Date Difference',
    'transpose': 'Transpose',
    'string-pad': 'String Pad',
    'cumulative-operations': 'Cumulative Operations',
    'replace-values': 'Replace Values',
    'percent-change': 'Percent Change',
    'round-numbers': 'Round Numbers',
    'percent-of-total': 'Percent of Total',
    'absolute-value': 'Absolute Value',
    'column-math': 'Column Math',
    'extract-substring': 'Extract Substring',
    'parse-date': 'Parse Date',
    'split-to-rows': 'Split to Rows',
    'clip-values': 'Clip Values',
    'standardize-text': 'Standardize Text',
    'case-when': 'Case When',
    'statistics': 'Statistics',
    'regression': 'Regression',
    'clustering': 'Clustering',
    'pca': 'PCA',
    'outlier-detection': 'Outlier Detection',
    'classification': 'Classification',
    'normality-test': 'Normality Test',
    'hypothesis-testing': 'Hypothesis Testing',
    'time-series': 'Time Series',
    'feature-importance': 'Feature Importance',
    'cross-validation': 'Cross-Validation',
    'data-profiling': 'Data Profiling',
    'value-counts': 'Value Counts',
    'cross-tabulation': 'Cross-Tabulation',
    'scaling': 'Scaling',
    'encoding': 'Encoding',
    'ab-test': 'A/B Test',
    'cohort-analysis': 'Cohort Analysis',
    'rfm-analysis': 'RFM Analysis',
    'anova': 'ANOVA',
    'chi-square-test': 'Chi-Square Test',
    'correlation-analysis': 'Correlation Analysis',
    'survival-analysis': 'Survival Analysis',
    'association-rules': 'Association Rules',
    'sentiment-analysis': 'Sentiment Analysis',
    'moving-average': 'Moving Average',
    'train-test-split': 'Train/Test Split',
    'model-evaluation': 'Model Evaluation',
    'knn': 'K-Nearest Neighbors',
    'naive-bayes': 'Naive Bayes',
    'gradient-boosting': 'Gradient Boosting',
    'pareto-analysis': 'Pareto Analysis',
    'trend-analysis': 'Trend Analysis',
    'forecasting': 'Forecasting',
    'percentile-analysis': 'Percentile Analysis',
    'distribution-fit': 'Distribution Fit',
    'text-preprocessing': 'Text Preprocessing',
    'tfidf-vectorization': 'TF-IDF Vectorization',
    'topic-modeling': 'Topic Modeling',
    'similarity-analysis': 'Similarity Analysis',
    'svm': 'SVM',
    'xgboost': 'XGBoost',
    'model-explainability': 'Model Explainability (SHAP)',
    'regression-diagnostics': 'Regression Diagnostics',
    'vif-analysis': 'VIF Analysis',
    'funnel-analysis': 'Funnel Analysis',
    'customer-ltv': 'Customer Lifetime Value',
    'churn-analysis': 'Churn Prediction',
    'growth-metrics': 'Growth Metrics',
    'attribution-modeling': 'Attribution Modeling',
    'breakeven-analysis': 'Break-even Analysis',
    'confidence-intervals': 'Confidence Intervals',
    'bootstrap-analysis': 'Bootstrap Analysis',
    'posthoc-tests': 'Post-hoc Tests',
    'power-analysis': 'Power Analysis',
    'bayesian-inference': 'Bayesian Inference',
    'data-quality-score': 'Data Quality Score',
    'changepoint-detection': 'Change Point Detection',
    'chart': 'Chart',
    'table': 'Table',
    'correlation-matrix': 'Correlation Matrix',
    'violin-plot': 'Violin Plot',
    'pair-plot': 'Pair Plot',
    'area-chart': 'Area Chart',
    'stacked-chart': 'Stacked Chart',
    'bubble-chart': 'Bubble Chart',
    'qq-plot': 'Q-Q Plot',
    'confusion-matrix': 'Confusion Matrix',
    'roc-curve': 'ROC Curve',
    'export': 'Export',
  };
  return labels[type];
}

function getDefaultConfig(type: BlockType): Record<string, unknown> {
  const configs: Partial<Record<BlockType, Record<string, unknown>>> = {
    'load-data': { fileType: 'csv', encoding: 'utf-8', delimiter: 'auto' },
    'sample-data': { dataset: 'iris' },
    'create-dataset': { columns: 'name,age,city', data: 'Alice,30,New York\nBob,25,Los Angeles\nCharlie,35,Chicago' },
    'filter-rows': { column: '', operator: 'equals', value: '' },
    'select-columns': { columns: [], rename: {} },
    'sort': { columns: [], ascending: true },
    'group-aggregate': { groupBy: [], aggregations: {} },
    'join': { how: 'inner', leftOn: '', rightOn: '' },
    'derive-column': { newColumn: '', expression: '' },
    'handle-missing': { strategy: 'drop', fillValue: '', columns: [] },
    'rename-columns': { renames: {} },
    'deduplicate': { columns: [], keep: 'first' },
    'sample-rows': { sampleType: 'count', count: 100, fraction: 0.1, seed: null },
    'limit-rows': { position: 'first', count: 10 },
    'pivot': { index: '', columns: '', values: '', aggFunc: 'mean' },
    'unpivot': { idColumns: [], valueColumns: [], varName: 'variable', valueName: 'value' },
    'union': { ignoreIndex: true },
    'split-column': { column: '', delimiter: ',', newColumns: [], keepOriginal: false },
    'merge-columns': { columns: [], separator: ' ', newColumn: 'merged', keepOriginal: true },
    'conditional-column': { newColumn: '', condition: '', trueValue: '', falseValue: '' },
    'statistics': { type: 'descriptive', columns: [] },
    'regression': { type: 'linear', features: [], target: '' },
    'clustering': { algorithm: 'kmeans', nClusters: 3, features: [] },
    'pca': { nComponents: 2, features: [], scaleData: true },
    'outlier-detection': { method: 'iqr', threshold: 1.5, columns: [] },
    'classification': { algorithm: 'decision_tree', features: [], target: '', testSize: 0.2 },
    'normality-test': { method: 'shapiro', columns: [], alpha: 0.05 },
    'hypothesis-testing': { testType: 'ttest_ind', column1: '', column2: '', groupColumn: '', alpha: 0.05 },
    'time-series': { dateColumn: '', valueColumn: '', analysis: 'moving_average', windowSize: 7 },
    'feature-importance': { features: [], target: '', taskType: 'auto' },
    'cross-validation': { features: [], target: '', modelType: 'random_forest', taskType: 'auto', nFolds: 5 },
    'data-profiling': {},
    'value-counts': { column: '', normalize: false, sortBy: 'count', topN: 0 },
    'cross-tabulation': { rowColumn: '', colColumn: '', normalize: 'none', showTotals: true },
    'scaling': { columns: [], method: 'standard', keepOriginal: false },
    'encoding': { columns: [], method: 'onehot', dropFirst: false },
    'chart': { chartType: 'bar', x: '', y: '', color: '', title: '' },
    'table': { pageSize: 100, sortable: true, filterable: true },
    'correlation-matrix': { columns: [], method: 'pearson', showValues: true },
    'violin-plot': { column: '', groupColumn: '', title: '' },
    'pair-plot': { columns: [], colorColumn: '', maxColumns: 5 },
    'area-chart': { x: '', y: '', color: '', stacked: false, title: '' },
    'stacked-chart': { x: '', yColumns: [], chartType: 'bar', normalize: false, title: '' },
    'bubble-chart': { x: '', y: '', size: '', color: '', title: '' },
    'qq-plot': { column: '', title: '' },
    'confusion-matrix': { actualColumn: '', predictedColumn: '', normalize: false, title: '' },
    'roc-curve': { actualColumn: '', probabilityColumn: '', positiveClass: '', title: '' },
    'export': { format: 'csv', filename: 'export' },
  };
  return configs[type] || {};
}

export const useCanvasStore = create<CanvasState>()(
  temporal(
    (set, get) => ({
      blocks: [],
      edges: [],
      selectedBlockIds: [],
      viewport: { x: 0, y: 0, zoom: 1 },

      addBlock: (type, position) => {
        const id = uuidv4();
        set((state) => ({
          blocks: [
            ...state.blocks,
            {
              id,
              type: 'custom',
              position,
              data: {
                type,
                category: getBlockCategory(type),
                label: getBlockLabel(type),
                config: getDefaultConfig(type),
                state: 'idle',
              },
            },
          ],
        }));
        return id;
      },

      addBlockDirect: (block) => {
        set((state) => {
          // Check if block already exists
          if (state.blocks.some((b) => b.id === block.id)) {
            return state;
          }
          return {
            blocks: [...state.blocks, block],
          };
        });
      },

      updateBlock: (id, data) => {
        set((state) => ({
          blocks: state.blocks.map((block) =>
            block.id === id
              ? { ...block, data: { ...block.data, ...data } }
              : block
          ),
        }));
      },

      removeBlock: (id) => {
        set((state) => ({
          blocks: state.blocks.filter((b) => b.id !== id),
          edges: state.edges.filter((e) => e.source !== id && e.target !== id),
          selectedBlockIds: state.selectedBlockIds.filter((i) => i !== id),
        }));
      },

      setBlockState: (id, blockState) => {
        set((state) => ({
          blocks: state.blocks.map((block) =>
            block.id === id
              ? { ...block, data: { ...block.data, state: blockState } }
              : block
          ),
        }));
      },

      addEdge: (source, target, sourceHandle, targetHandle) => {
        const id = `${source}-${target}`;
        set((state) => {
          if (state.edges.find((e) => e.id === id)) {
            return state;
          }
          return {
            edges: [
              ...state.edges,
              { id, source, target, sourceHandle, targetHandle, type: 'animated' },
            ],
          };
        });
      },

      addEdgeDirect: (edge) => {
        set((state) => {
          // Check if edge already exists
          if (state.edges.some((e) => e.id === edge.id)) {
            return state;
          }
          return {
            edges: [...state.edges, edge],
          };
        });
      },

      removeEdge: (id) => {
        set((state) => ({
          edges: state.edges.filter((e) => e.id !== id),
        }));
      },

      setBlocks: (blocks) => {
        set({ blocks });
      },

      setEdges: (edges) => {
        set({ edges });
      },

      setSelectedBlocks: (ids) => {
        set({ selectedBlockIds: ids });
      },

      clearSelection: () => {
        set({ selectedBlockIds: [] });
      },

      setViewport: (viewport) => {
        set({ viewport });
      },

      importPipeline: (blocks, edges) => {
        set({ blocks, edges });
      },

      clearCanvas: () => {
        set({
          blocks: [],
          edges: [],
          selectedBlockIds: [],
        });
      },
    }),
    {
      limit: 100,
    }
  )
);
