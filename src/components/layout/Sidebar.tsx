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
  Search,
  X,
  Cpu,
  Clock,
  MapPin,
  Sparkles,
  Sigma,
  BarChart,
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
  Cpu,
  Clock,
  MapPin,
  Sparkles,
  Sigma,
  BarChart,
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
  // Data Input
  'load-data': 'blocks.loadData',
  'sample-data': 'blocks.sampleData',
  'create-dataset': 'blocks.createDataset',
  // Transform
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
  'datetime-extract': 'blocks.datetimeExtract',
  'string-operations': 'blocks.stringOperations',
  'window-functions': 'blocks.windowFunctions',
  'bin-bucket': 'blocks.binBucket',
  'rank': 'blocks.rank',
  'type-conversion': 'blocks.typeConversion',
  'fill-forward-backward': 'blocks.fillForwardBackward',
  'lag-lead': 'blocks.lagLead',
  'row-number': 'blocks.rowNumber',
  'date-difference': 'blocks.dateDifference',
  'transpose': 'blocks.transpose',
  'string-pad': 'blocks.stringPad',
  'cumulative-operations': 'blocks.cumulativeOperations',
  'replace-values': 'blocks.replaceValues',
  'percent-change': 'blocks.percentChange',
  'round-numbers': 'blocks.roundNumbers',
  'percent-of-total': 'blocks.percentOfTotal',
  'absolute-value': 'blocks.absoluteValue',
  'column-math': 'blocks.columnMath',
  'extract-substring': 'blocks.extractSubstring',
  'parse-date': 'blocks.parseDate',
  'split-to-rows': 'blocks.splitToRows',
  'clip-values': 'blocks.clipValues',
  'standardize-text': 'blocks.standardizeText',
  'case-when': 'blocks.caseWhen',
  'explode-column': 'blocks.explodeColumn',
  'add-constant-column': 'blocks.addConstantColumn',
  'drop-columns': 'blocks.dropColumns',
  'flatten-json': 'blocks.flattenJson',
  'coalesce-columns': 'blocks.coalesceColumns',
  'reorder-columns': 'blocks.reorderColumns',
  'trim-text': 'blocks.trimText',
  'lookup-vlookup': 'blocks.lookupVlookup',
  'cross-join': 'blocks.crossJoin',
  'filter-expression': 'blocks.filterExpression',
  'number-format': 'blocks.numberFormat',
  'extract-pattern': 'blocks.extractPattern',
  'log-transform': 'blocks.logTransform',
  'interpolate-missing': 'blocks.interpolateMissing',
  'date-truncate': 'blocks.dateTruncate',
  'period-over-period': 'blocks.periodOverPeriod',
  'hash-column': 'blocks.hashColumn',
  'expand-date-range': 'blocks.expandDateRange',
  'string-similarity': 'blocks.stringSimilarity',
  'generate-sequence': 'blocks.generateSequence',
  'top-n-per-group': 'blocks.topNPerGroup',
  'first-last-per-group': 'blocks.firstLastPerGroup',
  'one-hot-encode': 'blocks.oneHotEncode',
  'label-encode': 'blocks.labelEncode',
  'ordinal-encode': 'blocks.ordinalEncode',
  'min-max-normalize': 'blocks.minMaxNormalize',
  'z-score-standardize': 'blocks.zScoreStandardize',
  'rolling-statistics': 'blocks.rollingStatistics',
  'resample-timeseries': 'blocks.resampleTimeseries',
  'regex-replace': 'blocks.regexReplace',
  'expand-json-column': 'blocks.expandJsonColumn',
  'add-unique-id': 'blocks.addUniqueId',
  'missing-indicator': 'blocks.missingIndicator',
  'quantile-transform': 'blocks.quantileTransform',
  // New Data Science Transform Blocks
  'fuzzy-join': 'blocks.fuzzyJoin',
  'memory-optimizer': 'blocks.memoryOptimizer',
  'cyclical-time-encoder': 'blocks.cyclicalTimeEncoder',
  'geographic-distance': 'blocks.geographicDistance',
  'rare-category-combiner': 'blocks.rareCategoryCombiner',
  'smart-auto-cleaner': 'blocks.smartAutoCleaner',
  'interaction-generator': 'blocks.interactionGenerator',
  'fuzzy-deduplicator': 'blocks.fuzzyDeduplicator',
  'array-aggregator': 'blocks.arrayAggregator',
  'target-aware-binning': 'blocks.targetAwareBinning',
  // Analysis
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
  'anova': 'blocks.anova',
  'chi-square-test': 'blocks.chiSquareTest',
  'correlation-analysis': 'blocks.correlationAnalysis',
  'survival-analysis': 'blocks.survivalAnalysis',
  'association-rules': 'blocks.associationRules',
  'sentiment-analysis': 'blocks.sentimentAnalysis',
  'moving-average': 'blocks.movingAverage',
  'train-test-split': 'blocks.trainTestSplit',
  'model-evaluation': 'blocks.modelEvaluation',
  'knn': 'blocks.knn',
  'naive-bayes': 'blocks.naiveBayes',
  'gradient-boosting': 'blocks.gradientBoosting',
  'pareto-analysis': 'blocks.paretoAnalysis',
  'trend-analysis': 'blocks.trendAnalysis',
  'forecasting': 'blocks.forecasting',
  'percentile-analysis': 'blocks.percentileAnalysis',
  'distribution-fit': 'blocks.distributionFit',
  'text-preprocessing': 'blocks.textPreprocessing',
  'tfidf-vectorization': 'blocks.tfidfVectorization',
  'topic-modeling': 'blocks.topicModeling',
  'similarity-analysis': 'blocks.similarityAnalysis',
  'svm': 'blocks.svm',
  'xgboost': 'blocks.xgboost',
  'model-explainability': 'blocks.modelExplainability',
  'regression-diagnostics': 'blocks.regressionDiagnostics',
  'vif-analysis': 'blocks.vifAnalysis',
  'funnel-analysis': 'blocks.funnelAnalysis',
  'customer-ltv': 'blocks.customerLtv',
  'churn-analysis': 'blocks.churnAnalysis',
  'growth-metrics': 'blocks.growthMetrics',
  'attribution-modeling': 'blocks.attributionModeling',
  'breakeven-analysis': 'blocks.breakevenAnalysis',
  'confidence-intervals': 'blocks.confidenceIntervals',
  'bootstrap-analysis': 'blocks.bootstrapAnalysis',
  'posthoc-tests': 'blocks.posthocTests',
  'power-analysis': 'blocks.powerAnalysis',
  'bayesian-inference': 'blocks.bayesianInference',
  'data-quality-score': 'blocks.dataQualityScore',
  'changepoint-detection': 'blocks.changepointDetection',
  'isolation-forest': 'blocks.isolationForest',
  'arima-forecasting': 'blocks.arimaForecasting',
  'seasonal-decomposition': 'blocks.seasonalDecomposition',
  'monte-carlo-simulation': 'blocks.monteCarloSimulation',
  'propensity-score-matching': 'blocks.propensityScoreMatching',
  'difference-in-differences': 'blocks.differenceInDifferences',
  'factor-analysis': 'blocks.factorAnalysis',
  'dbscan-clustering': 'blocks.dbscanClustering',
  'elastic-net': 'blocks.elasticNet',
  'var-analysis': 'blocks.varAnalysis',
  'interrupted-time-series': 'blocks.interruptedTimeSeries',
  'granger-causality': 'blocks.grangerCausality',
  'local-outlier-factor': 'blocks.localOutlierFactor',
  'feature-selection': 'blocks.featureSelection',
  'outlier-treatment': 'blocks.outlierTreatment',
  'data-drift': 'blocks.dataDrift',
  'polynomial-features': 'blocks.polynomialFeatures',
  'multi-output': 'blocks.multiOutput',
  'probability-calibration': 'blocks.probabilityCalibration',
  'tsne-reduction': 'blocks.tsneReduction',
  'statistical-tests': 'blocks.statisticalTests',
  'optimal-binning': 'blocks.optimalBinning',
  'correlation-finder': 'blocks.correlationFinder',
  'ab-test-calculator': 'blocks.abTestCalculator',
  'target-encoding': 'blocks.targetEncoding',
  'learning-curves': 'blocks.learningCurves',
  'imbalanced-data-handler': 'blocks.imbalancedDataHandler',
  'hyperparameter-tuning': 'blocks.hyperparameterTuning',
  'ensemble-stacking': 'blocks.ensembleStacking',
  'advanced-imputation': 'blocks.advancedImputation',
  'umap-reduction': 'blocks.umapReduction',
  'cluster-validation': 'blocks.clusterValidation',
  'model-comparison': 'blocks.modelComparison',
  'time-series-cv': 'blocks.timeSeriesCv',
  'uplift-modeling': 'blocks.upliftModeling',
  'quantile-regression': 'blocks.quantileRegression',
  'adversarial-validation': 'blocks.adversarialValidation',
  'custom-python-code': 'blocks.customPythonCode',
  'sql-query': 'blocks.sqlQuery',
  'auto-eda': 'blocks.autoEda',
  'data-validation': 'blocks.dataValidation',
  'neural-network': 'blocks.neuralNetwork',
  'auto-feature-engineering': 'blocks.autoFeatureEngineering',
  'shap-interpretation': 'blocks.shapInterpretation',
  'automl': 'blocks.automl',
  'pipeline-export': 'blocks.pipelineExport',
  'multivariate-anomaly': 'blocks.multivariateAnomaly',
  'causal-impact': 'blocks.causalImpact',
  'model-registry': 'blocks.modelRegistry',
  // Visualization
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
  'funnel-chart': 'blocks.funnelChart',
  'sankey-diagram': 'blocks.sankeyDiagram',
  'treemap': 'blocks.treemap',
  'sunburst-chart': 'blocks.sunburstChart',
  'gauge-chart': 'blocks.gaugeChart',
  'radar-chart': 'blocks.radarChart',
  'waterfall-chart': 'blocks.waterfallChart',
  'candlestick-chart': 'blocks.candlestickChart',
  'choropleth-map': 'blocks.choroplethMap',
  'word-cloud': 'blocks.wordCloud',
  'pareto-chart': 'blocks.paretoChart',
  'parallel-coordinates': 'blocks.parallelCoordinates',
  'dendrogram': 'blocks.dendrogram',
  'box-plot': 'blocks.boxPlot',
  'heatmap': 'blocks.heatmap',
  'scatter-map': 'blocks.scatterMap',
  'grouped-histogram': 'blocks.groupedHistogram',
  'network-graph': 'blocks.networkGraph',
  'calendar-heatmap': 'blocks.calendarHeatmap',
  'faceted-chart': 'blocks.facetedChart',
  'density-plot': 'blocks.densityPlot',
  'error-bar-chart': 'blocks.errorBarChart',
  'dot-plot': 'blocks.dotPlot',
  'slope-chart': 'blocks.slopeChart',
  'grouped-bar-chart': 'blocks.groupedBarChart',
  'bump-chart': 'blocks.bumpChart',
  'donut-chart': 'blocks.donutChart',
  'horizontal-bar-chart': 'blocks.horizontalBarChart',
  'scatter-3d': 'blocks.scatter3d',
  'contour-plot': 'blocks.contourPlot',
  'hexbin-plot': 'blocks.hexbinPlot',
  'ridge-plot': 'blocks.ridgePlot',
  'strip-plot': 'blocks.stripPlot',
  'bullet-chart': 'blocks.bulletChart',
  'pyramid-chart': 'blocks.pyramidChart',
  'timeline-chart': 'blocks.timelineChart',
  'surface-3d': 'blocks.surface3d',
  'marginal-histogram': 'blocks.marginalHistogram',
  'dumbbell-chart': 'blocks.dumbbellChart',
  'shap-summary-plot': 'blocks.shapSummaryPlot',
  'partial-dependence-plot': 'blocks.partialDependencePlot',
  'feature-importance-plot': 'blocks.featureImportancePlot',
  'ice-plot': 'blocks.icePlot',
  'precision-recall-curve': 'blocks.precisionRecallCurve',
  'learning-curve-plot': 'blocks.learningCurvePlot',
  'residual-plot': 'blocks.residualPlot',
  'actual-vs-predicted-plot': 'blocks.actualVsPredictedPlot',
  'calibration-curve': 'blocks.calibrationCurve',
  'lift-chart': 'blocks.liftChart',
  'elbow-plot': 'blocks.elbowPlot',
  'silhouette-plot': 'blocks.silhouettePlot',
  'tsne-umap-plot': 'blocks.tsneUmapPlot',
  'missing-value-heatmap': 'blocks.missingValueHeatmap',
  'outlier-detection-plot': 'blocks.outlierDetectionPlot',
  'distribution-comparison-plot': 'blocks.distributionComparisonPlot',
  'ecdf-plot': 'blocks.ecdfPlot',
  'andrews-curves': 'blocks.andrewsCurves',
  'cv-results-plot': 'blocks.cvResultsPlot',
  'hyperparameter-heatmap': 'blocks.hyperparameterHeatmap',
  // Output
  'export': 'blocks.export',
};

export function Sidebar() {
  const { t } = useTranslation();
  const { isSidebarOpen } = useUIStore();
  const [expandedCategories, setExpandedCategories] = React.useState<string[]>(
    blockCategories.map((c) => c.id)
  );
  const [searchQuery, setSearchQuery] = React.useState('');

  const toggleCategory = (categoryId: string) => {
    setExpandedCategories((prev) =>
      prev.includes(categoryId)
        ? prev.filter((id) => id !== categoryId)
        : [...prev, categoryId]
    );
  };

  // Filter blocks based on search query
  const filteredCategories = React.useMemo(() => {
    if (!searchQuery.trim()) {
      return blockCategories;
    }

    const query = searchQuery.toLowerCase().trim();

    return blockCategories
      .map((category) => {
        const filteredBlocks = category.blocks.filter((blockType) => {
          const definition = blockDefinitions[blockType];
          const label = definition.label.toLowerCase();
          const description = definition.description.toLowerCase();
          const translatedLabel = t(blockTranslationKeys[blockType] || definition.label).toLowerCase();

          return (
            label.includes(query) ||
            description.includes(query) ||
            translatedLabel.includes(query) ||
            blockType.includes(query)
          );
        });

        return {
          ...category,
          blocks: filteredBlocks,
        };
      })
      .filter((category) => category.blocks.length > 0);
  }, [searchQuery, t]);

  // Auto-expand categories when searching
  React.useEffect(() => {
    if (searchQuery.trim()) {
      setExpandedCategories(filteredCategories.map((c) => c.id));
    }
  }, [searchQuery, filteredCategories]);

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

      {/* Search Bar */}
      <div className="px-3 py-2 border-b border-border-default">
        <div className="relative">
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted"
          />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={t('sidebar.searchBlocks', 'Search blocks...')}
            className={cn(
              'w-full pl-9 pr-8 py-2 rounded-lg',
              'bg-bg-tertiary border border-border-default',
              'text-small text-text-primary placeholder:text-text-muted',
              'focus:outline-none focus:ring-2 focus:ring-electric-indigo focus:border-transparent',
              'transition-all duration-200'
            )}
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1 hover:bg-bg-secondary rounded"
            >
              <X size={14} className="text-text-muted" />
            </button>
          )}
        </div>
        {searchQuery && (
          <p className="text-xs text-text-muted mt-1">
            {filteredCategories.reduce((acc, cat) => acc + cat.blocks.length, 0)} {t('sidebar.blocksFound', 'blocks found')}
          </p>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {filteredCategories.length === 0 && searchQuery ? (
          <div className="text-center py-8 text-text-muted">
            <Search size={32} className="mx-auto mb-2 opacity-50" />
            <p className="text-small">{t('sidebar.noBlocksFound', 'No blocks found')}</p>
            <p className="text-xs mt-1">{t('sidebar.tryDifferentSearch', 'Try a different search term')}</p>
          </div>
        ) : (
          filteredCategories.map((category) => {
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
        })
        )}
      </div>
    </aside>
  );
}
