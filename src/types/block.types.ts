/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import type { Node, Edge } from '@xyflow/react';

export type BlockCategory =
  | 'data-input'
  | 'transform'
  | 'analysis'
  | 'visualization'
  | 'output';

export type BlockType =
  // Data Input
  | 'load-data'
  | 'sample-data'
  | 'create-dataset'
  // Transform
  | 'filter-rows'
  | 'select-columns'
  | 'sort'
  | 'group-aggregate'
  | 'join'
  | 'derive-column'
  | 'handle-missing'
  | 'rename-columns'
  | 'deduplicate'
  | 'sample-rows'
  | 'limit-rows'
  | 'pivot'
  | 'unpivot'
  | 'union'
  | 'split-column'
  | 'merge-columns'
  | 'conditional-column'
  | 'datetime-extract'
  | 'string-operations'
  | 'window-functions'
  | 'bin-bucket'
  | 'rank'
  | 'type-conversion'
  | 'fill-forward-backward'
  | 'lag-lead'
  | 'row-number'
  | 'date-difference'
  | 'transpose'
  | 'string-pad'
  | 'cumulative-operations'
  | 'replace-values'
  | 'percent-change'
  | 'round-numbers'
  | 'percent-of-total'
  | 'absolute-value'
  | 'column-math'
  | 'extract-substring'
  | 'parse-date'
  | 'split-to-rows'
  | 'clip-values'
  | 'standardize-text'
  | 'case-when'
  | 'explode-column'
  | 'add-constant-column'
  | 'drop-columns'
  | 'flatten-json'
  | 'coalesce-columns'
  | 'reorder-columns'
  | 'trim-text'
  | 'lookup-vlookup'
  | 'cross-join'
  | 'filter-expression'
  | 'number-format'
  | 'extract-pattern'
  | 'log-transform'
  | 'interpolate-missing'
  | 'date-truncate'
  | 'period-over-period'
  | 'hash-column'
  | 'expand-date-range'
  | 'string-similarity'
  | 'generate-sequence'
  | 'top-n-per-group'
  | 'first-last-per-group'
  // Analysis
  | 'statistics'
  | 'regression'
  | 'clustering'
  | 'pca'
  | 'outlier-detection'
  | 'classification'
  | 'normality-test'
  | 'hypothesis-testing'
  | 'time-series'
  | 'feature-importance'
  | 'cross-validation'
  | 'data-profiling'
  | 'value-counts'
  | 'cross-tabulation'
  | 'scaling'
  | 'encoding'
  | 'ab-test'
  | 'cohort-analysis'
  | 'rfm-analysis'
  | 'anova'
  | 'chi-square-test'
  | 'correlation-analysis'
  | 'survival-analysis'
  | 'association-rules'
  | 'sentiment-analysis'
  | 'moving-average'
  | 'train-test-split'
  | 'model-evaluation'
  | 'knn'
  | 'naive-bayes'
  | 'gradient-boosting'
  | 'pareto-analysis'
  | 'trend-analysis'
  | 'forecasting'
  | 'percentile-analysis'
  | 'distribution-fit'
  | 'text-preprocessing'
  | 'tfidf-vectorization'
  | 'topic-modeling'
  | 'similarity-analysis'
  | 'svm'
  | 'xgboost'
  | 'model-explainability'
  | 'regression-diagnostics'
  | 'vif-analysis'
  | 'funnel-analysis'
  | 'customer-ltv'
  | 'churn-analysis'
  | 'growth-metrics'
  | 'attribution-modeling'
  | 'breakeven-analysis'
  | 'confidence-intervals'
  | 'bootstrap-analysis'
  | 'posthoc-tests'
  | 'power-analysis'
  | 'bayesian-inference'
  | 'data-quality-score'
  | 'changepoint-detection'
  // Visualization
  | 'chart'
  | 'table'
  | 'correlation-matrix'
  | 'violin-plot'
  | 'pair-plot'
  | 'area-chart'
  | 'stacked-chart'
  | 'bubble-chart'
  | 'qq-plot'
  | 'confusion-matrix'
  | 'roc-curve'
  | 'funnel-chart'
  | 'sankey-diagram'
  | 'treemap'
  | 'sunburst-chart'
  | 'gauge-chart'
  | 'radar-chart'
  | 'waterfall-chart'
  | 'candlestick-chart'
  | 'choropleth-map'
  | 'word-cloud'
  | 'pareto-chart'
  | 'parallel-coordinates'
  | 'dendrogram'
  // Output
  | 'export';

export type BlockState = 'idle' | 'selected' | 'executing' | 'success' | 'error';

export interface BlockConfig {
  [key: string]: unknown;
}

export interface BlockData extends Record<string, unknown> {
  type: BlockType;
  category: BlockCategory;
  label: string;
  config: BlockConfig;
  state: BlockState;
  error?: string;
  pythonCode?: string;
}

export type PipelineBlock = Node<BlockData>;
export type PipelineEdge = Edge;

export interface BlockDefinition {
  type: BlockType;
  category: BlockCategory;
  label: string;
  description: string;
  icon: string;
  defaultConfig: BlockConfig;
  inputs: number;
  outputs: number;
  pythonTemplate: string;
}
