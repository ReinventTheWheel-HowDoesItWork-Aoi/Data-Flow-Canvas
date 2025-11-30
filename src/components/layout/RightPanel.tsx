/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { X, Upload, Database, BarChart3, Table2, Loader2, CheckCircle, AlertCircle, AlertTriangle, Download } from 'lucide-react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/Select';
import { useUIStore } from '@/stores/uiStore';
import { useCanvasStore } from '@/stores/canvasStore';
import { useExecutionStore } from '@/stores/executionStore';
import { usePyodide } from '@/lib/pyodide/PyodideProvider';
import { blockDefinitions } from '@/constants';
import { sampleDatasets } from '@/constants/sampleDatasets';
import { ChartVisualization, TableVisualization } from '@/components/visualization';
import type { BlockType, BlockConfig, PipelineBlock } from '@/types';
import { cn } from '@/lib/utils/cn';

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
  'export': 'blocks.export',
};

// Map block types to description translation keys
const blockDescriptionKeys: Record<BlockType, string> = {
  'load-data': 'blockDescriptions.loadData',
  'sample-data': 'blockDescriptions.sampleData',
  'create-dataset': 'blockDescriptions.createDataset',
  'filter-rows': 'blockDescriptions.filterRows',
  'select-columns': 'blockDescriptions.selectColumns',
  'sort': 'blockDescriptions.sort',
  'group-aggregate': 'blockDescriptions.groupAggregate',
  'join': 'blockDescriptions.join',
  'derive-column': 'blockDescriptions.deriveColumn',
  'handle-missing': 'blockDescriptions.handleMissing',
  'rename-columns': 'blockDescriptions.renameColumns',
  'deduplicate': 'blockDescriptions.deduplicate',
  'sample-rows': 'blockDescriptions.sampleRows',
  'limit-rows': 'blockDescriptions.limitRows',
  'pivot': 'blockDescriptions.pivot',
  'unpivot': 'blockDescriptions.unpivot',
  'union': 'blockDescriptions.union',
  'split-column': 'blockDescriptions.splitColumn',
  'merge-columns': 'blockDescriptions.mergeColumns',
  'conditional-column': 'blockDescriptions.conditionalColumn',
  'datetime-extract': 'blockDescriptions.datetimeExtract',
  'string-operations': 'blockDescriptions.stringOperations',
  'window-functions': 'blockDescriptions.windowFunctions',
  'bin-bucket': 'blockDescriptions.binBucket',
  'rank': 'blockDescriptions.rank',
  'type-conversion': 'blockDescriptions.typeConversion',
  'fill-forward-backward': 'blockDescriptions.fillForwardBackward',
  'lag-lead': 'blockDescriptions.lagLead',
  'row-number': 'blockDescriptions.rowNumber',
  'date-difference': 'blockDescriptions.dateDifference',
  'transpose': 'blockDescriptions.transpose',
  'string-pad': 'blockDescriptions.stringPad',
  'cumulative-operations': 'blockDescriptions.cumulativeOperations',
  'replace-values': 'blockDescriptions.replaceValues',
  'percent-change': 'blockDescriptions.percentChange',
  'round-numbers': 'blockDescriptions.roundNumbers',
  'percent-of-total': 'blockDescriptions.percentOfTotal',
  'absolute-value': 'blockDescriptions.absoluteValue',
  'column-math': 'blockDescriptions.columnMath',
  'extract-substring': 'blockDescriptions.extractSubstring',
  'parse-date': 'blockDescriptions.parseDate',
  'split-to-rows': 'blockDescriptions.splitToRows',
  'clip-values': 'blockDescriptions.clipValues',
  'standardize-text': 'blockDescriptions.standardizeText',
  'case-when': 'blockDescriptions.caseWhen',
  'explode-column': 'blockDescriptions.explodeColumn',
  'add-constant-column': 'blockDescriptions.addConstantColumn',
  'drop-columns': 'blockDescriptions.dropColumns',
  'flatten-json': 'blockDescriptions.flattenJson',
  'coalesce-columns': 'blockDescriptions.coalesceColumns',
  'reorder-columns': 'blockDescriptions.reorderColumns',
  'trim-text': 'blockDescriptions.trimText',
  'lookup-vlookup': 'blockDescriptions.lookupVlookup',
  'cross-join': 'blockDescriptions.crossJoin',
  'filter-expression': 'blockDescriptions.filterExpression',
  'number-format': 'blockDescriptions.numberFormat',
  'extract-pattern': 'blockDescriptions.extractPattern',
  'log-transform': 'blockDescriptions.logTransform',
  'interpolate-missing': 'blockDescriptions.interpolateMissing',
  'date-truncate': 'blockDescriptions.dateTruncate',
  'period-over-period': 'blockDescriptions.periodOverPeriod',
  'hash-column': 'blockDescriptions.hashColumn',
  'expand-date-range': 'blockDescriptions.expandDateRange',
  'string-similarity': 'blockDescriptions.stringSimilarity',
  'generate-sequence': 'blockDescriptions.generateSequence',
  'top-n-per-group': 'blockDescriptions.topNPerGroup',
  'first-last-per-group': 'blockDescriptions.firstLastPerGroup',
  'statistics': 'blockDescriptions.statistics',
  'regression': 'blockDescriptions.regression',
  'clustering': 'blockDescriptions.clustering',
  'pca': 'blockDescriptions.pca',
  'outlier-detection': 'blockDescriptions.outlierDetection',
  'classification': 'blockDescriptions.classification',
  'normality-test': 'blockDescriptions.normalityTest',
  'hypothesis-testing': 'blockDescriptions.hypothesisTesting',
  'time-series': 'blockDescriptions.timeSeries',
  'feature-importance': 'blockDescriptions.featureImportance',
  'cross-validation': 'blockDescriptions.crossValidation',
  'data-profiling': 'blockDescriptions.dataProfiling',
  'value-counts': 'blockDescriptions.valueCounts',
  'cross-tabulation': 'blockDescriptions.crossTabulation',
  'scaling': 'blockDescriptions.scaling',
  'encoding': 'blockDescriptions.encoding',
  'ab-test': 'blockDescriptions.abTest',
  'cohort-analysis': 'blockDescriptions.cohortAnalysis',
  'rfm-analysis': 'blockDescriptions.rfmAnalysis',
  'anova': 'blockDescriptions.anova',
  'chi-square-test': 'blockDescriptions.chiSquareTest',
  'correlation-analysis': 'blockDescriptions.correlationAnalysis',
  'survival-analysis': 'blockDescriptions.survivalAnalysis',
  'association-rules': 'blockDescriptions.associationRules',
  'sentiment-analysis': 'blockDescriptions.sentimentAnalysis',
  'moving-average': 'blockDescriptions.movingAverage',
  'train-test-split': 'blockDescriptions.trainTestSplit',
  'model-evaluation': 'blockDescriptions.modelEvaluation',
  'knn': 'blockDescriptions.knn',
  'naive-bayes': 'blockDescriptions.naiveBayes',
  'gradient-boosting': 'blockDescriptions.gradientBoosting',
  'pareto-analysis': 'blockDescriptions.paretoAnalysis',
  'trend-analysis': 'blockDescriptions.trendAnalysis',
  'forecasting': 'blockDescriptions.forecasting',
  'percentile-analysis': 'blockDescriptions.percentileAnalysis',
  'distribution-fit': 'blockDescriptions.distributionFit',
  'text-preprocessing': 'blockDescriptions.textPreprocessing',
  'tfidf-vectorization': 'blockDescriptions.tfidfVectorization',
  'topic-modeling': 'blockDescriptions.topicModeling',
  'similarity-analysis': 'blockDescriptions.similarityAnalysis',
  'svm': 'blockDescriptions.svm',
  'xgboost': 'blockDescriptions.xgboost',
  'model-explainability': 'blockDescriptions.modelExplainability',
  'regression-diagnostics': 'blockDescriptions.regressionDiagnostics',
  'vif-analysis': 'blockDescriptions.vifAnalysis',
  'funnel-analysis': 'blockDescriptions.funnelAnalysis',
  'customer-ltv': 'blockDescriptions.customerLtv',
  'churn-analysis': 'blockDescriptions.churnAnalysis',
  'growth-metrics': 'blockDescriptions.growthMetrics',
  'attribution-modeling': 'blockDescriptions.attributionModeling',
  'breakeven-analysis': 'blockDescriptions.breakevenAnalysis',
  'confidence-intervals': 'blockDescriptions.confidenceIntervals',
  'bootstrap-analysis': 'blockDescriptions.bootstrapAnalysis',
  'posthoc-tests': 'blockDescriptions.posthocTests',
  'power-analysis': 'blockDescriptions.powerAnalysis',
  'bayesian-inference': 'blockDescriptions.bayesianInference',
  'data-quality-score': 'blockDescriptions.dataQualityScore',
  'changepoint-detection': 'blockDescriptions.changepointDetection',
  'chart': 'blockDescriptions.chart',
  'table': 'blockDescriptions.table',
  'correlation-matrix': 'blockDescriptions.correlationMatrix',
  'violin-plot': 'blockDescriptions.violinPlot',
  'pair-plot': 'blockDescriptions.pairPlot',
  'area-chart': 'blockDescriptions.areaChart',
  'stacked-chart': 'blockDescriptions.stackedChart',
  'bubble-chart': 'blockDescriptions.bubbleChart',
  'qq-plot': 'blockDescriptions.qqPlot',
  'confusion-matrix': 'blockDescriptions.confusionMatrix',
  'roc-curve': 'blockDescriptions.rocCurve',
  'funnel-chart': 'blockDescriptions.funnelChart',
  'sankey-diagram': 'blockDescriptions.sankeyDiagram',
  'treemap': 'blockDescriptions.treemap',
  'sunburst-chart': 'blockDescriptions.sunburstChart',
  'gauge-chart': 'blockDescriptions.gaugeChart',
  'radar-chart': 'blockDescriptions.radarChart',
  'waterfall-chart': 'blockDescriptions.waterfallChart',
  'candlestick-chart': 'blockDescriptions.candlestickChart',
  'choropleth-map': 'blockDescriptions.choroplethMap',
  'word-cloud': 'blockDescriptions.wordCloud',
  'pareto-chart': 'blockDescriptions.paretoChart',
  'parallel-coordinates': 'blockDescriptions.parallelCoordinates',
  'dendrogram': 'blockDescriptions.dendrogram',
  'export': 'blockDescriptions.export',
};

// Helper function to convert ArrayBuffer to Base64 (handles large files)
function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 8192;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
    binary += String.fromCharCode.apply(null, Array.from(chunk));
  }
  return btoa(binary);
}

export function RightPanel() {
  const { t } = useTranslation();
  const { isRightPanelOpen, rightPanelTab, setRightPanelTab, toggleRightPanel } =
    useUIStore();
  const { blocks, selectedBlockIds, updateBlock } = useCanvasStore();
  const { results } = useExecutionStore();

  const selectedBlock =
    selectedBlockIds.length === 1
      ? blocks.find((b) => b.id === selectedBlockIds[0])
      : null;

  const selectedResult = selectedBlock
    ? results.get(selectedBlock.id)
    : null;

  if (!isRightPanelOpen) {
    return null;
  }

  return (
    <aside className="w-80 bg-bg-secondary border-l border-border-default flex flex-col">
      <div className="flex items-center justify-between p-4 border-b border-border-default">
        <h2 className="text-h3 text-text-primary">
          {selectedBlock
            ? t(blockTranslationKeys[selectedBlock.data.type], { defaultValue: blockDefinitions[selectedBlock.data.type].label })
            : t('rightPanel.properties')}
        </h2>
        <Button variant="ghost" size="sm" onClick={toggleRightPanel}>
          <X size={18} />
        </Button>
      </div>

      <Tabs
        value={rightPanelTab}
        onValueChange={(v) => setRightPanelTab(v as typeof rightPanelTab)}
        className="flex-1 flex flex-col"
      >
        <TabsList className="mx-4 mt-2">
          <TabsTrigger value="config">{t('rightPanel.config')}</TabsTrigger>
          <TabsTrigger value="preview">{t('rightPanel.preview')}</TabsTrigger>
          <TabsTrigger value="visualization">{t('rightPanel.visualization')}</TabsTrigger>
        </TabsList>

        <TabsContent value="config" className="flex-1 p-4 overflow-y-auto">
          {selectedBlock ? (
            <BlockConfigEditor
              block={selectedBlock}
              onUpdate={(config) =>
                updateBlock(selectedBlock.id, { config })
              }
            />
          ) : (
            <div className="text-center text-text-muted py-8">
              <p>{t('rightPanel.selectBlock')}</p>
            </div>
          )}
        </TabsContent>

        <TabsContent value="preview" className="flex-1 p-4 overflow-y-auto">
          {selectedResult?.success && selectedResult.data ? (
            <DataPreview data={selectedResult.data} />
          ) : selectedResult?.error ? (
            <div className="text-center py-8">
              <AlertCircle size={48} className="mx-auto mb-4 text-warm-coral opacity-70" />
              <p className="text-warm-coral font-medium mb-2">{t('rightPanel.executionError')}</p>
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg text-left overflow-auto max-h-40">
                {selectedResult.error}
              </p>
            </div>
          ) : (
            <div className="text-center text-text-muted py-8">
              <p>{t('rightPanel.runToPreview')}</p>
            </div>
          )}
        </TabsContent>

        <TabsContent value="visualization" className="flex-1 p-4 overflow-y-auto">
          <VisualizationPanel
            block={selectedBlock ?? null}
            result={selectedResult ?? null}
          />
        </TabsContent>
      </Tabs>
    </aside>
  );
}

interface BlockConfigEditorProps {
  block: PipelineBlock;
  onUpdate: (config: BlockConfig) => void;
}

function BlockConfigEditor({ block, onUpdate }: BlockConfigEditorProps) {
  const { t } = useTranslation();
  const { type, config } = block.data;
  const definition = blockDefinitions[type];
  const { edges } = useCanvasStore();
  const { results } = useExecutionStore();

  // Get columns from upstream block's data
  const upstreamColumns = React.useMemo(() => {
    const inputEdge = edges.find((e) => e.target === block.id);
    if (!inputEdge) return [];

    const upstreamResult = results.get(inputEdge.source);
    if (!upstreamResult?.success || !upstreamResult.data) return [];

    // Extract column names from the first row of data
    const data = upstreamResult.data;
    if (Array.isArray(data) && data.length > 0 && typeof data[0] === 'object') {
      return Object.keys(data[0]);
    }
    return [];
  }, [edges, block.id, results]);

  const handleChange = useCallback(
    (key: string, value: unknown) => {
      onUpdate({ ...config, [key]: value });
    },
    [config, onUpdate]
  );

  // Get translated label and description
  const translatedLabel = t(blockTranslationKeys[type], { defaultValue: definition.label });
  const translatedDescription = t(blockDescriptionKeys[type], { defaultValue: definition.description });

  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-small font-medium text-text-primary mb-1">
          {translatedLabel}
        </h4>
        <p className="text-small text-text-muted">{translatedDescription}</p>
      </div>

      <div className="space-y-4">
        {renderConfigFields(type, config, handleChange, onUpdate, upstreamColumns)}
      </div>
    </div>
  );
}

// Separate component for CSV file upload with state management
function LoadDataConfig({
  config,
  onChange,
  onBatchUpdate
}: {
  config: BlockConfig;
  onChange: (key: string, value: unknown) => void;
  onBatchUpdate: (updates: BlockConfig) => void;
}) {
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string>('');

  // Derive upload status from config - file is loaded if we have both fileName and fileContent
  const hasFile = Boolean(config.fileName && config.fileContent);
  const uploadStatus = isUploading ? 'uploading' : errorMessage ? 'error' : hasFile ? 'success' : 'idle';

  const handleFileUpload = useCallback((file: File) => {
    setIsUploading(true);
    setErrorMessage('');

    // Capture current config values synchronously to avoid stale closure issues
    const currentEncoding = config.encoding || 'utf-8';

    const reader = new FileReader();

    reader.onload = (event) => {
      try {
        const content = event.target?.result;
        if (content instanceof ArrayBuffer) {
          // Use chunked base64 encoding for binary files
          const base64 = arrayBufferToBase64(content);
          // Batch all updates into a single call to avoid stale closure issues
          onBatchUpdate({
            fileName: file.name,
            fileType: 'csv',
            fileContent: base64,
            encoding: currentEncoding,
          });
          setIsUploading(false);
        } else if (typeof content === 'string') {
          // For text files, properly encode to Base64
          try {
            const base64 = btoa(unescape(encodeURIComponent(content)));
            onBatchUpdate({
              fileName: file.name,
              fileType: 'csv',
              fileContent: base64,
              encoding: currentEncoding,
            });
            setIsUploading(false);
          } catch {
            // Fallback: read as binary
            const textReader = new FileReader();
            textReader.onload = (e) => {
              if (e.target?.result instanceof ArrayBuffer) {
                const base64 = arrayBufferToBase64(e.target.result);
                onBatchUpdate({
                  fileName: file.name,
                  fileType: 'csv',
                  fileContent: base64,
                  encoding: currentEncoding,
                });
                setIsUploading(false);
              }
            };
            textReader.onerror = () => {
              setIsUploading(false);
              setErrorMessage('Failed to read file. Please try again.');
            };
            textReader.readAsArrayBuffer(file);
          }
        }
      } catch (error) {
        setIsUploading(false);
        setErrorMessage(error instanceof Error ? error.message : 'Failed to read file');
      }
    };

    reader.onerror = () => {
      setIsUploading(false);
      setErrorMessage('Failed to read file. Please try again.');
    };

    // Always read as ArrayBuffer for consistency
    reader.readAsArrayBuffer(file);
  }, [config.encoding, onBatchUpdate]);

  return (
    <>
      <div>
        <label className="block text-small font-medium text-text-secondary mb-1.5">
          CSV File
        </label>
        <label
          className={cn(
            'flex flex-col items-center justify-center',
            'w-full h-24 px-4 py-6',
            'border-2 border-dashed rounded-lg',
            'cursor-pointer transition-colors',
            'bg-bg-tertiary',
            uploadStatus === 'success' && 'border-fresh-teal',
            uploadStatus === 'error' && 'border-warm-coral',
            uploadStatus === 'idle' && 'border-border-default hover:border-electric-indigo',
            uploadStatus === 'uploading' && 'border-electric-indigo'
          )}
        >
          {uploadStatus === 'uploading' ? (
            <Loader2 size={24} className="text-electric-indigo mb-2 animate-spin" />
          ) : uploadStatus === 'success' ? (
            <CheckCircle size={24} className="text-fresh-teal mb-2" />
          ) : uploadStatus === 'error' ? (
            <AlertCircle size={24} className="text-warm-coral mb-2" />
          ) : (
            <Upload size={24} className="text-text-muted mb-2" />
          )}
          <span className={cn(
            'text-small text-center',
            uploadStatus === 'success' && 'text-fresh-teal',
            uploadStatus === 'error' && 'text-warm-coral',
            uploadStatus === 'idle' && 'text-text-muted',
            uploadStatus === 'uploading' && 'text-electric-indigo'
          )}>
            {uploadStatus === 'uploading' ? 'Uploading...' :
             uploadStatus === 'error' ? errorMessage :
             config.fileName ? (config.fileName as string) :
             'Click to upload CSV file'}
          </span>
          <input
            type="file"
            className="hidden"
            accept=".csv"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                handleFileUpload(file);
              }
            }}
          />
        </label>
        {hasFile && (
          <p className="text-small text-text-muted mt-1">
            File loaded successfully
          </p>
        )}
      </div>

      <Input
        label="Encoding"
        value={(config.encoding as string) || 'utf-8'}
        onChange={(e) => onChange('encoding', e.target.value)}
        placeholder="utf-8"
      />
    </>
  );
}

// Separate component for Rename Columns to properly use useState hooks
function RenameColumnsConfig({
  config,
  onChange,
  availableColumns,
}: {
  config: BlockConfig;
  onChange: (key: string, value: unknown) => void;
  availableColumns: string[];
}) {
  const [newOldName, setNewOldName] = useState('');
  const [newNewName, setNewNewName] = useState('');
  const renames = (config.renames as Record<string, string>) || {};

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <label className="block text-small font-medium text-text-secondary">
          Column Renames
        </label>
        {Object.entries(renames).length > 0 ? (
          <div className="space-y-2 bg-bg-tertiary rounded-lg p-3">
            {Object.entries(renames).map(([oldName, newName]) => (
              <div key={oldName} className="flex items-center gap-2">
                <span className="text-small font-mono text-text-muted">{oldName}</span>
                <span className="text-text-muted">→</span>
                <span className="text-small font-mono text-text-primary flex-1">{newName}</span>
                <button
                  type="button"
                  onClick={() => {
                    const newRenames = { ...renames };
                    delete newRenames[oldName];
                    onChange('renames', newRenames);
                  }}
                  className="text-warm-coral hover:text-warm-coral/80 text-small"
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
            No renames configured
          </p>
        )}
      </div>

      {availableColumns.length > 0 ? (
        <div className="space-y-2">
          <Select
            value={newOldName}
            onValueChange={setNewOldName}
          >
            <SelectTrigger label="Column to Rename">
              <SelectValue placeholder="Select column" />
            </SelectTrigger>
            <SelectContent>
              {availableColumns.filter(col => !renames[col]).map((col) => (
                <SelectItem key={col} value={col}>
                  {col}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Input
            label="New Name"
            value={newNewName}
            onChange={(e) => setNewNewName(e.target.value)}
            placeholder="Enter new column name"
          />
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              if (newOldName && newNewName) {
                onChange('renames', { ...renames, [newOldName]: newNewName });
                setNewOldName('');
                setNewNewName('');
              }
            }}
            disabled={!newOldName || !newNewName}
          >
            Add Rename
          </Button>
        </div>
      ) : (
        <p className="text-small text-text-muted">
          Run the pipeline first to see available columns
        </p>
      )}
    </div>
  );
}

function renderConfigFields(
  type: BlockType,
  config: BlockConfig,
  onChange: (key: string, value: unknown) => void,
  onBatchUpdate: (config: BlockConfig) => void,
  availableColumns: string[] = []
) {
  switch (type) {
    case 'load-data':
      return (
        <LoadDataConfig config={config} onChange={onChange} onBatchUpdate={onBatchUpdate} />
      );

    case 'sample-data':
      return (
        <Select
          value={(config.dataset as string) || 'iris'}
          onValueChange={(v) => onChange('dataset', v)}
        >
          <SelectTrigger label="Dataset">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {sampleDatasets.map((ds) => (
              <SelectItem key={ds.id} value={ds.id}>
                <div className="flex items-center gap-2">
                  <Database size={14} />
                  <span>{ds.name}</span>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      );

    case 'create-dataset':
      return (
        <>
          <Input
            label="Column Names (comma-separated)"
            value={(config.columns as string) || ''}
            onChange={(e) => onChange('columns', e.target.value)}
            placeholder="e.g., name,age,city"
          />
          <div>
            <label className="block text-small font-medium text-text-secondary mb-1.5">
              Data (CSV format, one row per line)
            </label>
            <textarea
              className={cn(
                'w-full min-h-[150px] px-3 py-2',
                'bg-bg-tertiary border border-border-default rounded-lg',
                'text-text-primary text-small font-mono',
                'focus:outline-none focus:ring-2 focus:ring-electric-indigo/50 focus:border-electric-indigo',
                'placeholder:text-text-muted resize-y'
              )}
              value={(config.data as string) || ''}
              onChange={(e) => onChange('data', e.target.value)}
              placeholder="Alice,30,New York&#10;Bob,25,Los Angeles&#10;Charlie,35,Chicago"
            />
            <p className="text-small text-text-muted mt-1">
              Enter data in CSV format. Each line is a row.
            </p>
          </div>
        </>
      );

    case 'filter-rows':
      return (
        <>
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select a column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Select
            value={(config.operator as string) || 'equals'}
            onValueChange={(v) => onChange('operator', v)}
          >
            <SelectTrigger label="Operator">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="equals">Equals</SelectItem>
              <SelectItem value="not_equals">Not Equals</SelectItem>
              <SelectItem value="greater_than">Greater Than</SelectItem>
              <SelectItem value="less_than">Less Than</SelectItem>
              <SelectItem value="contains">Contains</SelectItem>
              <SelectItem value="starts_with">Starts With</SelectItem>
              <SelectItem value="ends_with">Ends With</SelectItem>
              <SelectItem value="is_null">Is Null</SelectItem>
              <SelectItem value="is_not_null">Is Not Null</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Value"
            value={(config.value as string) || ''}
            onChange={(e) => onChange('value', e.target.value)}
            placeholder="e.g., 25"
          />
        </>
      );

    case 'select-columns': {
      const selectedCols = (config.columns as string[]) || [];
      return (
        <div className="space-y-3">
          {availableColumns.length > 0 ? (
            <>
              <label className="block text-small font-medium text-text-secondary">
                Select Columns
              </label>
              <div className="space-y-2 max-h-60 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={selectedCols.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...selectedCols, col]
                          : selectedCols.filter((c) => c !== col);
                        onChange('columns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
              <p className="text-small text-text-muted">
                {selectedCols.length} of {availableColumns.length} columns selected
              </p>
            </>
          ) : (
            <div className="text-center py-4 bg-bg-tertiary rounded-lg">
              <p className="text-small text-text-muted">
                Run the pipeline first to see available columns
              </p>
            </div>
          )}
        </div>
      );
    }

    case 'sort': {
      const sortColumns = (config.columns as string[]) || [];
      return (
        <>
          {availableColumns.length > 0 ? (
            <Select
              value={sortColumns[0] || ''}
              onValueChange={(v) => onChange('columns', [v])}
            >
              <SelectTrigger label="Sort by Column">
                <SelectValue placeholder="Select a column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Sort by column"
              value={sortColumns.join(', ')}
              onChange={(e) =>
                onChange(
                  'columns',
                  e.target.value.split(',').map((s) => s.trim()).filter(Boolean)
                )
              }
              placeholder="Run pipeline to see columns"
            />
          )}

          <Select
            value={(config.ascending as boolean) !== false ? 'asc' : 'desc'}
            onValueChange={(v) => onChange('ascending', v === 'asc')}
          >
            <SelectTrigger label="Order">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="asc">Ascending</SelectItem>
              <SelectItem value="desc">Descending</SelectItem>
            </SelectContent>
          </Select>
        </>
      );
    }

    case 'group-aggregate': {
      const groupByCols = (config.groupBy as string[]) || [];
      const aggregations = (config.aggregations as Record<string, string>) || {};
      return (
        <div className="space-y-4">
          {/* Group By Columns */}
          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Group By Columns
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={groupByCols.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...groupByCols, col]
                          : groupByCols.filter((c) => c !== col);
                        onChange('groupBy', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          {/* Aggregations */}
          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Aggregations
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-48 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns
                  .filter((col) => !groupByCols.includes(col))
                  .map((col) => (
                    <div key={col} className="flex items-center gap-2">
                      <span className="text-small text-text-primary font-mono flex-1 truncate">{col}</span>
                      <select
                        value={aggregations[col] || ''}
                        onChange={(e) => {
                          const newAgg = { ...aggregations };
                          if (e.target.value) {
                            newAgg[col] = e.target.value;
                          } else {
                            delete newAgg[col];
                          }
                          onChange('aggregations', newAgg);
                        }}
                        className="bg-bg-secondary border border-border-default rounded px-2 py-1 text-small text-text-primary"
                      >
                        <option value="">None</option>
                        <option value="sum">Sum</option>
                        <option value="mean">Mean</option>
                        <option value="count">Count</option>
                        <option value="min">Min</option>
                        <option value="max">Max</option>
                        <option value="std">Std Dev</option>
                        <option value="first">First</option>
                        <option value="last">Last</option>
                      </select>
                    </div>
                  ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>
        </div>
      );
    }

    case 'derive-column': {
      const currentExpression = (config.expression as string) || '';
      return (
        <div className="space-y-4">
          <Input
            label="New Column Name"
            value={(config.newColumn as string) || ''}
            onChange={(e) => onChange('newColumn', e.target.value)}
            placeholder="e.g., total_price"
          />

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Expression
            </label>
            <textarea
              className="w-full min-h-[80px] px-3 py-2 bg-bg-tertiary border border-border-default rounded-lg text-text-primary text-small font-mono focus:outline-none focus:ring-2 focus:ring-electric-indigo/50 focus:border-electric-indigo resize-y"
              value={currentExpression}
              onChange={(e) => onChange('expression', e.target.value)}
              placeholder="e.g., df['price'] * df['quantity']"
            />
          </div>

          {availableColumns.length > 0 && (
            <div className="space-y-2">
              <label className="block text-small font-medium text-text-secondary">
                Click to insert column
              </label>
              <div className="flex flex-wrap gap-1.5">
                {availableColumns.map((col) => (
                  <button
                    key={col}
                    type="button"
                    onClick={() => onChange('expression', currentExpression + `df['${col}']`)}
                    className="px-2 py-1 text-small font-mono bg-bg-tertiary hover:bg-electric-indigo/20 border border-border-default rounded transition-colors text-text-primary"
                  >
                    {col}
                  </button>
                ))}
              </div>
            </div>
          )}

          <p className="text-small text-text-muted">
            Use Python/pandas syntax. Example: df['price'] * df['qty']
          </p>
        </div>
      );
    }

    case 'handle-missing': {
      const selectedMissingCols = (config.columns as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.strategy as string) || 'drop'}
            onValueChange={(v) => onChange('strategy', v)}
          >
            <SelectTrigger label="Strategy">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="drop">Drop Rows</SelectItem>
              <SelectItem value="fill_value">Fill with Value</SelectItem>
              <SelectItem value="fill_mean">Fill with Mean</SelectItem>
              <SelectItem value="fill_median">Fill with Median</SelectItem>
              <SelectItem value="fill_mode">Fill with Mode</SelectItem>
              <SelectItem value="interpolate">Interpolate</SelectItem>
            </SelectContent>
          </Select>

          {config.strategy === 'fill_value' && (
            <Input
              label="Fill Value"
              value={(config.fillValue as string) || ''}
              onChange={(e) => onChange('fillValue', e.target.value)}
              placeholder="e.g., 0"
            />
          )}

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Apply to Columns (leave empty for all)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={selectedMissingCols.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...selectedMissingCols, col]
                          : selectedMissingCols.filter((c) => c !== col);
                        onChange('columns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>
        </div>
      );
    }

    case 'statistics':
      return (
        <Select
          value={(config.type as string) || 'descriptive'}
          onValueChange={(v) => onChange('type', v)}
        >
          <SelectTrigger label="Statistics Type">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="descriptive">Descriptive Statistics</SelectItem>
            <SelectItem value="correlation">Correlation Matrix</SelectItem>
            <SelectItem value="covariance">Covariance Matrix</SelectItem>
          </SelectContent>
        </Select>
      );

    case 'regression': {
      const featureCols = (config.features as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.type as string) || 'linear'}
            onValueChange={(v) => onChange('type', v)}
          >
            <SelectTrigger label="Regression Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="linear">Linear Regression</SelectItem>
              <SelectItem value="logistic">Logistic Regression</SelectItem>
            </SelectContent>
          </Select>

          {availableColumns.length > 0 ? (
            <Select
              value={(config.target as string) || ''}
              onValueChange={(v) => onChange('target', v)}
            >
              <SelectTrigger label="Target Column (Y)">
                <SelectValue placeholder="Select target" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Target Column"
              value={(config.target as string) || ''}
              onChange={(e) => onChange('target', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Feature Columns (X)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns
                  .filter((col) => col !== config.target)
                  .map((col) => (
                    <label
                      key={col}
                      className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                    >
                      <input
                        type="checkbox"
                        checked={featureCols.includes(col)}
                        onChange={(e) => {
                          const newCols = e.target.checked
                            ? [...featureCols, col]
                            : featureCols.filter((c) => c !== col);
                          onChange('features', newCols);
                        }}
                        className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                      />
                      <span className="text-small text-text-primary font-mono">{col}</span>
                    </label>
                  ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>
        </div>
      );
    }

    case 'clustering': {
      const clusterFeatures = (config.features as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.algorithm as string) || 'kmeans'}
            onValueChange={(v) => onChange('algorithm', v)}
          >
            <SelectTrigger label="Algorithm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="kmeans">K-Means</SelectItem>
              <SelectItem value="hierarchical">Hierarchical</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Number of Clusters"
            type="number"
            value={(config.nClusters as number) || 3}
            onChange={(e) => onChange('nClusters', parseInt(e.target.value) || 3)}
          />

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Feature Columns
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={clusterFeatures.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...clusterFeatures, col]
                          : clusterFeatures.filter((c) => c !== col);
                        onChange('features', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>
        </div>
      );
    }

    case 'chart':
      return (
        <div className="space-y-4">
          <Select
            value={(config.chartType as string) || 'bar'}
            onValueChange={(v) => onChange('chartType', v)}
          >
            <SelectTrigger label="Chart Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="bar">Bar Chart</SelectItem>
              <SelectItem value="line">Line Chart</SelectItem>
              <SelectItem value="scatter">Scatter Plot</SelectItem>
              <SelectItem value="pie">Pie Chart</SelectItem>
              <SelectItem value="histogram">Histogram</SelectItem>
              <SelectItem value="box">Box Plot</SelectItem>
            </SelectContent>
          </Select>

          {availableColumns.length > 0 ? (
            <Select
              value={(config.x as string) || ''}
              onValueChange={(v) => onChange('x', v)}
            >
              <SelectTrigger label="X Axis Column">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="X Axis Column"
              value={(config.x as string) || ''}
              onChange={(e) => onChange('x', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          {availableColumns.length > 0 ? (
            <Select
              value={(config.y as string) || ''}
              onValueChange={(v) => onChange('y', v)}
            >
              <SelectTrigger label="Y Axis Column">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Y Axis Column"
              value={(config.y as string) || ''}
              onChange={(e) => onChange('y', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          {availableColumns.length > 0 ? (
            <Select
              value={(config.color as string) || '__none__'}
              onValueChange={(v) => onChange('color', v === '__none__' ? '' : v)}
            >
              <SelectTrigger label="Color Column (optional)">
                <SelectValue placeholder="None" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">None</SelectItem>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Color Column (optional)"
              value={(config.color as string) || ''}
              onChange={(e) => onChange('color', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Input
            label="Title"
            value={(config.title as string) || ''}
            onChange={(e) => onChange('title', e.target.value)}
            placeholder="e.g., Sales by Region"
          />
        </div>
      );

    case 'table':
      return (
        <Input
          label="Page Size"
          type="number"
          value={(config.pageSize as number) || 100}
          onChange={(e) => onChange('pageSize', parseInt(e.target.value) || 100)}
        />
      );

    case 'export':
      return (
        <Input
          label="Filename"
          value={(config.filename as string) || 'export'}
          onChange={(e) => onChange('filename', e.target.value)}
          placeholder="e.g., my_data"
        />
      );

    case 'join':
      return (
        <div className="space-y-4">
          <Select
            value={(config.how as string) || 'inner'}
            onValueChange={(v) => onChange('how', v)}
          >
            <SelectTrigger label="Join Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="inner">Inner Join</SelectItem>
              <SelectItem value="left">Left Join</SelectItem>
              <SelectItem value="right">Right Join</SelectItem>
              <SelectItem value="outer">Outer Join</SelectItem>
            </SelectContent>
          </Select>

          {availableColumns.length > 0 ? (
            <Select
              value={(config.leftOn as string) || ''}
              onValueChange={(v) => onChange('leftOn', v)}
            >
              <SelectTrigger label="Left Key Column">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Left Key Column"
              value={(config.leftOn as string) || ''}
              onChange={(e) => onChange('leftOn', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          {availableColumns.length > 0 ? (
            <Select
              value={(config.rightOn as string) || ''}
              onValueChange={(v) => onChange('rightOn', v)}
            >
              <SelectTrigger label="Right Key Column">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Right Key Column"
              value={(config.rightOn as string) || ''}
              onChange={(e) => onChange('rightOn', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <p className="text-small text-text-muted">
            Connect two data sources to this block, then select the key columns to join on.
          </p>
        </div>
      );

    case 'rename-columns':
      return (
        <RenameColumnsConfig
          config={config}
          onChange={onChange}
          availableColumns={availableColumns}
        />
      );

    case 'deduplicate': {
      const dedupColumns = (config.columns as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.keep as string) || 'first'}
            onValueChange={(v) => onChange('keep', v)}
          >
            <SelectTrigger label="Keep">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="first">First Occurrence</SelectItem>
              <SelectItem value="last">Last Occurrence</SelectItem>
              <SelectItem value="none">Drop All Duplicates</SelectItem>
            </SelectContent>
          </Select>

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Compare Columns (empty = all)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={dedupColumns.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...dedupColumns, col]
                          : dedupColumns.filter((c) => c !== col);
                        onChange('columns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>
        </div>
      );
    }

    case 'sample-rows':
      return (
        <div className="space-y-4">
          <Select
            value={(config.sampleType as string) || 'count'}
            onValueChange={(v) => onChange('sampleType', v)}
          >
            <SelectTrigger label="Sample Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="count">Fixed Count</SelectItem>
              <SelectItem value="fraction">Percentage</SelectItem>
            </SelectContent>
          </Select>

          {(config.sampleType as string) !== 'fraction' ? (
            <Input
              label="Number of Rows"
              type="number"
              value={(config.count as number) || 100}
              onChange={(e) => onChange('count', parseInt(e.target.value) || 100)}
            />
          ) : (
            <Input
              label="Fraction (0-1)"
              type="number"
              step="0.1"
              min="0"
              max="1"
              value={(config.fraction as number) || 0.1}
              onChange={(e) => onChange('fraction', parseFloat(e.target.value) || 0.1)}
            />
          )}

          <Input
            label="Random Seed (optional)"
            type="number"
            value={(config.seed as number) || ''}
            onChange={(e) => onChange('seed', e.target.value ? parseInt(e.target.value) : null)}
            placeholder="Leave empty for random"
          />
        </div>
      );

    case 'limit-rows':
      return (
        <div className="space-y-4">
          <Select
            value={(config.position as string) || 'first'}
            onValueChange={(v) => onChange('position', v)}
          >
            <SelectTrigger label="Position">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="first">First N Rows</SelectItem>
              <SelectItem value="last">Last N Rows</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Number of Rows"
            type="number"
            value={(config.count as number) || 10}
            onChange={(e) => onChange('count', parseInt(e.target.value) || 10)}
          />
        </div>
      );

    case 'pivot':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.index as string) || ''}
                onValueChange={(v) => onChange('index', v)}
              >
                <SelectTrigger label="Index Column (Rows)">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.columns as string) || ''}
                onValueChange={(v) => onChange('columns', v)}
              >
                <SelectTrigger label="Columns Column (Creates Headers)">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.values as string) || ''}
                onValueChange={(v) => onChange('values', v)}
              >
                <SelectTrigger label="Values Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
              Run pipeline to see columns
            </p>
          )}

          <Select
            value={(config.aggFunc as string) || 'mean'}
            onValueChange={(v) => onChange('aggFunc', v)}
          >
            <SelectTrigger label="Aggregation Function">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="mean">Mean</SelectItem>
              <SelectItem value="sum">Sum</SelectItem>
              <SelectItem value="count">Count</SelectItem>
              <SelectItem value="min">Min</SelectItem>
              <SelectItem value="max">Max</SelectItem>
              <SelectItem value="first">First</SelectItem>
              <SelectItem value="last">Last</SelectItem>
            </SelectContent>
          </Select>
        </div>
      );

    case 'unpivot': {
      const idCols = (config.idColumns as string[]) || [];
      const valueCols = (config.valueColumns as string[]) || [];
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              ID Columns (keep as-is)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-32 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={idCols.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...idCols, col]
                          : idCols.filter((c) => c !== col);
                        onChange('idColumns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Value Columns (to unpivot)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-32 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.filter(col => !idCols.includes(col)).map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={valueCols.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...valueCols, col]
                          : valueCols.filter((c) => c !== col);
                        onChange('valueColumns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          <Input
            label="Variable Column Name"
            value={(config.varName as string) || 'variable'}
            onChange={(e) => onChange('varName', e.target.value)}
          />

          <Input
            label="Value Column Name"
            value={(config.valueName as string) || 'value'}
            onChange={(e) => onChange('valueName', e.target.value)}
          />
        </div>
      );
    }

    case 'union':
      return (
        <div className="space-y-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.ignoreIndex as boolean) !== false}
              onChange={(e) => onChange('ignoreIndex', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">Reset row index</span>
          </label>
          <p className="text-small text-text-muted">
            Connect two data sources to stack them vertically.
          </p>
        </div>
      );

    case 'split-column':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column to Split">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column to Split"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Input
            label="Delimiter"
            value={(config.delimiter as string) || ','}
            onChange={(e) => onChange('delimiter', e.target.value)}
            placeholder="e.g., , or - or /"
          />

          <Input
            label="New Column Names (comma-separated, optional)"
            value={((config.newColumns as string[]) || []).join(', ')}
            onChange={(e) => onChange('newColumns', e.target.value.split(',').map(s => s.trim()).filter(Boolean))}
            placeholder="e.g., first, second, third"
          />

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.keepOriginal as boolean) || false}
              onChange={(e) => onChange('keepOriginal', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">Keep original column</span>
          </label>
        </div>
      );

    case 'merge-columns': {
      const mergeCols = (config.columns as string[]) || [];
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Columns to Merge
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={mergeCols.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...mergeCols, col]
                          : mergeCols.filter((c) => c !== col);
                        onChange('columns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          <Input
            label="Separator"
            value={(config.separator as string) || ' '}
            onChange={(e) => onChange('separator', e.target.value)}
            placeholder="e.g., space, comma, dash"
          />

          <Input
            label="New Column Name"
            value={(config.newColumn as string) || 'merged'}
            onChange={(e) => onChange('newColumn', e.target.value)}
          />

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.keepOriginal as boolean) !== false}
              onChange={(e) => onChange('keepOriginal', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">Keep original columns</span>
          </label>
        </div>
      );
    }

    case 'conditional-column': {
      const currentCondition = (config.condition as string) || '';
      return (
        <div className="space-y-4">
          <Input
            label="New Column Name"
            value={(config.newColumn as string) || ''}
            onChange={(e) => onChange('newColumn', e.target.value)}
            placeholder="e.g., status"
          />

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Condition
            </label>
            <textarea
              className="w-full min-h-[60px] px-3 py-2 bg-bg-tertiary border border-border-default rounded-lg text-text-primary text-small font-mono focus:outline-none focus:ring-2 focus:ring-electric-indigo/50 focus:border-electric-indigo resize-y"
              value={currentCondition}
              onChange={(e) => onChange('condition', e.target.value)}
              placeholder="e.g., df['age'] > 18"
            />
          </div>

          {availableColumns.length > 0 && (
            <div className="space-y-2">
              <label className="block text-small font-medium text-text-secondary">
                Click to insert column
              </label>
              <div className="flex flex-wrap gap-1.5">
                {availableColumns.map((col) => (
                  <button
                    key={col}
                    type="button"
                    onClick={() => onChange('condition', currentCondition + `df['${col}']`)}
                    className="px-2 py-1 text-small font-mono bg-bg-tertiary hover:bg-electric-indigo/20 border border-border-default rounded transition-colors text-text-primary"
                  >
                    {col}
                  </button>
                ))}
              </div>
            </div>
          )}

          <Input
            label="Value if True"
            value={(config.trueValue as string) || ''}
            onChange={(e) => onChange('trueValue', e.target.value)}
            placeholder="e.g., adult"
          />

          <Input
            label="Value if False"
            value={(config.falseValue as string) || ''}
            onChange={(e) => onChange('falseValue', e.target.value)}
            placeholder="e.g., minor"
          />
        </div>
      );
    }

    case 'datetime-extract': {
      const selectedExtractions = (config.extractions as string[]) || ['year', 'month', 'day'];
      const extractionOptions = [
        { value: 'year', label: 'Year' },
        { value: 'month', label: 'Month (1-12)' },
        { value: 'day', label: 'Day (1-31)' },
        { value: 'weekday', label: 'Weekday Name' },
        { value: 'weekday_num', label: 'Weekday Number (0-6)' },
        { value: 'hour', label: 'Hour (0-23)' },
        { value: 'minute', label: 'Minute (0-59)' },
        { value: 'second', label: 'Second (0-59)' },
        { value: 'quarter', label: 'Quarter (1-4)' },
        { value: 'week', label: 'Week of Year' },
        { value: 'dayofyear', label: 'Day of Year' },
        { value: 'is_weekend', label: 'Is Weekend' },
        { value: 'date', label: 'Date Only' },
      ];
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Date Column">
                <SelectValue placeholder="Select a date column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Date Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Extract Parts
            </label>
            <div className="space-y-2 max-h-60 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
              {extractionOptions.map((opt) => (
                <label
                  key={opt.value}
                  className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                >
                  <input
                    type="checkbox"
                    checked={selectedExtractions.includes(opt.value)}
                    onChange={(e) => {
                      const newExtractions = e.target.checked
                        ? [...selectedExtractions, opt.value]
                        : selectedExtractions.filter((ex) => ex !== opt.value);
                      onChange('extractions', newExtractions);
                    }}
                    className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                  />
                  <span className="text-small text-text-primary">{opt.label}</span>
                </label>
              ))}
            </div>
          </div>

          <Input
            label="Column Prefix (optional)"
            value={(config.prefix as string) || ''}
            onChange={(e) => onChange('prefix', e.target.value)}
            placeholder="Leave empty to use column name"
          />

          <p className="text-small text-text-muted">
            Creates new columns like {String(config.prefix || config.column || 'date')}_year, {String(config.prefix || config.column || 'date')}_month, etc.
          </p>
        </div>
      );
    }

    case 'string-operations': {
      const operation = (config.operation as string) || 'lowercase';
      const showFindReplace = operation === 'find_replace';
      const showRegex = operation === 'regex_replace' || operation === 'regex_extract';

      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select a column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Select
            value={operation}
            onValueChange={(v) => onChange('operation', v)}
          >
            <SelectTrigger label="Operation">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="lowercase">Lowercase</SelectItem>
              <SelectItem value="uppercase">Uppercase</SelectItem>
              <SelectItem value="titlecase">Title Case</SelectItem>
              <SelectItem value="trim">Trim (both sides)</SelectItem>
              <SelectItem value="trim_left">Trim Left</SelectItem>
              <SelectItem value="trim_right">Trim Right</SelectItem>
              <SelectItem value="find_replace">Find & Replace</SelectItem>
              <SelectItem value="regex_replace">Regex Replace</SelectItem>
              <SelectItem value="regex_extract">Regex Extract</SelectItem>
              <SelectItem value="length">Get Length</SelectItem>
              <SelectItem value="remove_digits">Remove Digits</SelectItem>
              <SelectItem value="remove_punctuation">Remove Punctuation</SelectItem>
              <SelectItem value="remove_whitespace">Normalize Whitespace</SelectItem>
            </SelectContent>
          </Select>

          {showFindReplace && (
            <>
              <Input
                label="Find Text"
                value={(config.findText as string) || ''}
                onChange={(e) => onChange('findText', e.target.value)}
                placeholder="Text to find"
              />
              <Input
                label="Replace With"
                value={(config.replaceText as string) || ''}
                onChange={(e) => onChange('replaceText', e.target.value)}
                placeholder="Replacement text"
              />
            </>
          )}

          {showRegex && (
            <>
              <Input
                label="Regex Pattern"
                value={(config.regexPattern as string) || ''}
                onChange={(e) => onChange('regexPattern', e.target.value)}
                placeholder="e.g., [0-9]+ or \w+"
              />
              {operation === 'regex_replace' && (
                <Input
                  label="Replace With"
                  value={(config.replaceText as string) || ''}
                  onChange={(e) => onChange('replaceText', e.target.value)}
                  placeholder="Replacement text"
                />
              )}
            </>
          )}

          <Input
            label="New Column Name (optional)"
            value={(config.newColumn as string) || ''}
            onChange={(e) => onChange('newColumn', e.target.value)}
            placeholder="Leave empty to modify in place"
          />
        </div>
      );
    }

    case 'window-functions': {
      const windowOperation = (config.operation as string) || 'rolling_mean';
      const groupByCols = (config.groupBy as string[]) || [];
      const isRolling = windowOperation.startsWith('rolling_');
      const isShift = windowOperation === 'lag' || windowOperation === 'lead' || windowOperation === 'pct_change' || windowOperation === 'diff';

      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select a numeric column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Select
            value={windowOperation}
            onValueChange={(v) => onChange('operation', v)}
          >
            <SelectTrigger label="Operation">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="rolling_mean">Rolling Mean (Moving Avg)</SelectItem>
              <SelectItem value="rolling_sum">Rolling Sum</SelectItem>
              <SelectItem value="rolling_min">Rolling Min</SelectItem>
              <SelectItem value="rolling_max">Rolling Max</SelectItem>
              <SelectItem value="rolling_std">Rolling Std Dev</SelectItem>
              <SelectItem value="cumsum">Cumulative Sum</SelectItem>
              <SelectItem value="cumprod">Cumulative Product</SelectItem>
              <SelectItem value="cummin">Cumulative Min</SelectItem>
              <SelectItem value="cummax">Cumulative Max</SelectItem>
              <SelectItem value="lag">Lag (Previous Row)</SelectItem>
              <SelectItem value="lead">Lead (Next Row)</SelectItem>
              <SelectItem value="pct_change">Percent Change</SelectItem>
              <SelectItem value="diff">Difference</SelectItem>
            </SelectContent>
          </Select>

          {(isRolling || isShift) && (
            <Input
              label={isRolling ? 'Window Size' : 'Periods'}
              type="number"
              value={(config.windowSize as number) || 3}
              onChange={(e) => onChange('windowSize', parseInt(e.target.value) || 3)}
              min={1}
            />
          )}

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Group By (optional)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-32 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={groupByCols.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...groupByCols, col]
                          : groupByCols.filter((c) => c !== col);
                        onChange('groupBy', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          <Input
            label="New Column Name (optional)"
            value={(config.newColumn as string) || ''}
            onChange={(e) => onChange('newColumn', e.target.value)}
            placeholder="Leave empty for auto-generated name"
          />
        </div>
      );
    }

    case 'bin-bucket': {
      const method = (config.method as string) || 'equal_width';
      const showCustomEdges = method === 'custom';

      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select a numeric column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Select
            value={method}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Binning Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="equal_width">Equal Width (same range per bin)</SelectItem>
              <SelectItem value="equal_frequency">Equal Frequency (same count per bin)</SelectItem>
              <SelectItem value="custom">Custom Edges</SelectItem>
            </SelectContent>
          </Select>

          {!showCustomEdges && (
            <Input
              label="Number of Bins"
              type="number"
              value={(config.numBins as number) || 5}
              onChange={(e) => onChange('numBins', parseInt(e.target.value) || 5)}
              min={2}
              max={100}
            />
          )}

          {showCustomEdges && (
            <Input
              label="Bin Edges (comma-separated)"
              value={(config.customEdges as string) || ''}
              onChange={(e) => onChange('customEdges', e.target.value)}
              placeholder="e.g., 0, 18, 30, 50, 100"
            />
          )}

          <Input
            label="Custom Labels (optional)"
            value={(config.customLabels as string) || ''}
            onChange={(e) => onChange('customLabels', e.target.value)}
            placeholder="e.g., low, medium, high"
          />

          <Input
            label="New Column Name (optional)"
            value={(config.newColumn as string) || ''}
            onChange={(e) => onChange('newColumn', e.target.value)}
            placeholder="Leave empty for auto-generated name"
          />

          <p className="text-small text-text-muted">
            {method === 'equal_width'
              ? 'Divides range into equal-sized intervals'
              : method === 'equal_frequency'
              ? 'Each bin contains approximately the same number of rows'
              : 'Define exact boundaries between bins'}
          </p>
        </div>
      );
    }

    case 'rank': {
      const rankGroupBy = (config.groupBy as string[]) || [];

      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column to Rank">
                <SelectValue placeholder="Select a column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column to Rank"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Select
            value={(config.method as string) || 'average'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Ranking Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="average">Average (ties get average rank)</SelectItem>
              <SelectItem value="min">Min (ties get lowest rank)</SelectItem>
              <SelectItem value="max">Max (ties get highest rank)</SelectItem>
              <SelectItem value="dense">Dense (no gaps in ranks)</SelectItem>
              <SelectItem value="first">First (ties by row order)</SelectItem>
            </SelectContent>
          </Select>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.ascending as boolean) !== false}
              onChange={(e) => onChange('ascending', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">
              Ascending (lowest value = rank 1)
            </span>
          </label>

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Group By (optional)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-32 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={rankGroupBy.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...rankGroupBy, col]
                          : rankGroupBy.filter((c) => c !== col);
                        onChange('groupBy', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
            <p className="text-small text-text-muted">
              Rank within each group (e.g., rank per region)
            </p>
          </div>

          <Input
            label="New Column Name (optional)"
            value={(config.newColumn as string) || ''}
            onChange={(e) => onChange('newColumn', e.target.value)}
            placeholder="Leave empty for auto-generated name"
          />
        </div>
      );
    }

    case 'type-conversion': {
      const targetType = (config.targetType as string) || 'string';
      const showDatetimeFormat = targetType === 'datetime';

      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select a column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Select
            value={targetType}
            onValueChange={(v) => onChange('targetType', v)}
          >
            <SelectTrigger label="Convert To">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="string">String (text)</SelectItem>
              <SelectItem value="integer">Integer (whole numbers)</SelectItem>
              <SelectItem value="float">Float (decimals)</SelectItem>
              <SelectItem value="boolean">Boolean (true/false)</SelectItem>
              <SelectItem value="datetime">DateTime</SelectItem>
              <SelectItem value="category">Category</SelectItem>
            </SelectContent>
          </Select>

          {showDatetimeFormat && (
            <Input
              label="DateTime Format (optional)"
              value={(config.datetimeFormat as string) || ''}
              onChange={(e) => onChange('datetimeFormat', e.target.value)}
              placeholder="e.g., %Y-%m-%d or %d/%m/%Y"
            />
          )}

          <Select
            value={(config.errorHandling as string) || 'coerce'}
            onValueChange={(v) => onChange('errorHandling', v)}
          >
            <SelectTrigger label="On Error">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="coerce">Coerce (set invalid to null)</SelectItem>
              <SelectItem value="raise">Raise Error</SelectItem>
            </SelectContent>
          </Select>

          <p className="text-small text-text-muted">
            {targetType === 'string' && 'Converts any value to text'}
            {targetType === 'integer' && 'Converts to whole numbers (decimals truncated)'}
            {targetType === 'float' && 'Converts to decimal numbers'}
            {targetType === 'boolean' && 'Recognizes: true/false, yes/no, 1/0, t/f, y/n'}
            {targetType === 'datetime' && 'Parses date/time strings. Common formats are auto-detected.'}
            {targetType === 'category' && 'Optimized for columns with limited unique values'}
          </p>
        </div>
      );
    }

    case 'fill-forward-backward': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select a column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Select
            value={(config.method as string) || 'forward'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Fill Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="forward">Forward (use previous value)</SelectItem>
              <SelectItem value="backward">Backward (use next value)</SelectItem>
              <SelectItem value="both">Both (forward then backward)</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Limit (optional)"
            type="number"
            value={(config.limit as string) || ''}
            onChange={(e) => onChange('limit', e.target.value)}
            placeholder="Max consecutive fills"
          />
        </div>
      );
    }

    case 'lag-lead': {
      const lagLeadGroupBy = (config.groupBy as string[]) || [];
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select a column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Select
            value={(config.operation as string) || 'lag'}
            onValueChange={(v) => onChange('operation', v)}
          >
            <SelectTrigger label="Operation">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="lag">Lag (shift backward)</SelectItem>
              <SelectItem value="lead">Lead (shift forward)</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Periods"
            type="number"
            value={(config.periods as number) || 1}
            onChange={(e) => onChange('periods', parseInt(e.target.value) || 1)}
            min={1}
          />

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Group By (optional)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-32 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={lagLeadGroupBy.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...lagLeadGroupBy, col]
                          : lagLeadGroupBy.filter((c) => c !== col);
                        onChange('groupBy', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted">Run pipeline to see columns</p>
            )}
          </div>

          <Input
            label="Output Column (optional)"
            value={(config.outputColumn as string) || ''}
            onChange={(e) => onChange('outputColumn', e.target.value)}
            placeholder="Auto-generated if empty"
          />
        </div>
      );
    }

    case 'row-number': {
      const rowNumGroupBy = (config.groupBy as string[]) || [];
      return (
        <div className="space-y-4">
          <Input
            label="Output Column"
            value={(config.outputColumn as string) || 'row_num'}
            onChange={(e) => onChange('outputColumn', e.target.value)}
            placeholder="row_num"
          />

          <Input
            label="Start From"
            type="number"
            value={(config.startFrom as number) || 1}
            onChange={(e) => onChange('startFrom', parseInt(e.target.value) || 1)}
          />

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Group By (optional)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-32 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={rowNumGroupBy.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...rowNumGroupBy, col]
                          : rowNumGroupBy.filter((c) => c !== col);
                        onChange('groupBy', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted">Run pipeline to see columns</p>
            )}
          </div>
        </div>
      );
    }

    case 'date-difference': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.startColumn as string) || ''}
                onValueChange={(v) => onChange('startColumn', v)}
              >
                <SelectTrigger label="Start Date Column">
                  <SelectValue placeholder="Select start date" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.endColumn as string) || ''}
                onValueChange={(v) => onChange('endColumn', v)}
              >
                <SelectTrigger label="End Date Column">
                  <SelectValue placeholder="Select end date" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Start Date Column"
                value={(config.startColumn as string) || ''}
                onChange={(e) => onChange('startColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="End Date Column"
                value={(config.endColumn as string) || ''}
                onChange={(e) => onChange('endColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            </>
          )}

          <Select
            value={(config.unit as string) || 'days'}
            onValueChange={(v) => onChange('unit', v)}
          >
            <SelectTrigger label="Unit">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="days">Days</SelectItem>
              <SelectItem value="hours">Hours</SelectItem>
              <SelectItem value="minutes">Minutes</SelectItem>
              <SelectItem value="seconds">Seconds</SelectItem>
              <SelectItem value="weeks">Weeks</SelectItem>
              <SelectItem value="months">Months (approx)</SelectItem>
              <SelectItem value="years">Years (approx)</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Output Column (optional)"
            value={(config.outputColumn as string) || ''}
            onChange={(e) => onChange('outputColumn', e.target.value)}
            placeholder="Auto-generated if empty"
          />
        </div>
      );
    }

    case 'transpose': {
      return (
        <div className="space-y-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.useFirstColumnAsHeader as boolean) !== false}
              onChange={(e) => onChange('useFirstColumnAsHeader', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">Use first column as header</span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.useFirstRowAsIndex as boolean) || false}
              onChange={(e) => onChange('useFirstRowAsIndex', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">Use first row as index</span>
          </label>

          <p className="text-small text-text-muted">
            Flips rows and columns. Original column names become row values.
          </p>
        </div>
      );
    }

    case 'string-pad': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select a column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Input
            label="Width"
            type="number"
            value={(config.width as number) || 10}
            onChange={(e) => onChange('width', parseInt(e.target.value) || 10)}
            min={1}
          />

          <Select
            value={(config.side as string) || 'left'}
            onValueChange={(v) => onChange('side', v)}
          >
            <SelectTrigger label="Pad Side">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="left">Left (prefix)</SelectItem>
              <SelectItem value="right">Right (suffix)</SelectItem>
              <SelectItem value="both">Both (center)</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Fill Character"
            value={(config.fillChar as string) || '0'}
            onChange={(e) => onChange('fillChar', e.target.value)}
            maxLength={1}
            placeholder="0"
          />

          <Input
            label="Output Column (optional)"
            value={(config.outputColumn as string) || ''}
            onChange={(e) => onChange('outputColumn', e.target.value)}
            placeholder="Overwrites original if empty"
          />
        </div>
      );
    }

    case 'cumulative-operations': {
      const cumGroupBy = (config.groupBy as string[]) || [];
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select a column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Select
            value={(config.operation as string) || 'sum'}
            onValueChange={(v) => onChange('operation', v)}
          >
            <SelectTrigger label="Operation">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="sum">Cumulative Sum (running total)</SelectItem>
              <SelectItem value="count">Cumulative Count</SelectItem>
              <SelectItem value="mean">Cumulative Mean</SelectItem>
              <SelectItem value="max">Cumulative Max</SelectItem>
              <SelectItem value="min">Cumulative Min</SelectItem>
              <SelectItem value="product">Cumulative Product</SelectItem>
              <SelectItem value="percent">Cumulative Percent</SelectItem>
            </SelectContent>
          </Select>

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Group By (optional)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-32 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={cumGroupBy.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...cumGroupBy, col]
                          : cumGroupBy.filter((c) => c !== col);
                        onChange('groupBy', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted">Run pipeline to see columns</p>
            )}
          </div>

          <Input
            label="Output Column (optional)"
            value={(config.outputColumn as string) || ''}
            onChange={(e) => onChange('outputColumn', e.target.value)}
            placeholder="Auto-generated if empty"
          />
        </div>
      );
    }

    case 'anova': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.valueColumn as string) || ''}
                onValueChange={(v) => onChange('valueColumn', v)}
              >
                <SelectTrigger label="Value Column (numeric)">
                  <SelectValue placeholder="Select value column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.groupColumn as string) || ''}
                onValueChange={(v) => onChange('groupColumn', v)}
              >
                <SelectTrigger label="Group Column (categorical)">
                  <SelectValue placeholder="Select group column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Value Column (numeric)"
                value={(config.valueColumn as string) || ''}
                onChange={(e) => onChange('valueColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Group Column (categorical)"
                value={(config.groupColumn as string) || ''}
                onChange={(e) => onChange('groupColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            </>
          )}

          <Input
            label="Significance Level (alpha)"
            type="number"
            value={(config.alpha as number) || 0.05}
            onChange={(e) => onChange('alpha', parseFloat(e.target.value) || 0.05)}
            step={0.01}
            min={0.01}
            max={0.5}
          />

          <p className="text-small text-text-muted">
            Tests if group means are significantly different. Returns F-statistic and p-value.
          </p>
        </div>
      );
    }

    case 'chi-square-test': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.column1 as string) || ''}
                onValueChange={(v) => onChange('column1', v)}
              >
                <SelectTrigger label="First Column (categorical)">
                  <SelectValue placeholder="Select first column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.column2 as string) || ''}
                onValueChange={(v) => onChange('column2', v)}
              >
                <SelectTrigger label="Second Column (categorical)">
                  <SelectValue placeholder="Select second column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="First Column"
                value={(config.column1 as string) || ''}
                onChange={(e) => onChange('column1', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Second Column"
                value={(config.column2 as string) || ''}
                onChange={(e) => onChange('column2', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            </>
          )}

          <Input
            label="Significance Level (alpha)"
            type="number"
            value={(config.alpha as number) || 0.05}
            onChange={(e) => onChange('alpha', parseFloat(e.target.value) || 0.05)}
            step={0.01}
            min={0.01}
            max={0.5}
          />

          <p className="text-small text-text-muted">
            Tests independence between two categorical variables. Also calculates Cramer's V effect size.
          </p>
        </div>
      );
    }

    case 'correlation-analysis': {
      const corrColumns = (config.columns as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.method as string) || 'pearson'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Correlation Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="pearson">Pearson (linear)</SelectItem>
              <SelectItem value="spearman">Spearman (rank)</SelectItem>
              <SelectItem value="kendall">Kendall (rank)</SelectItem>
            </SelectContent>
          </Select>

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Columns (leave empty for all numeric)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={corrColumns.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...corrColumns, col]
                          : corrColumns.filter((c) => c !== col);
                        onChange('columns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted">Run pipeline to see columns</p>
            )}
          </div>

          <p className="text-small text-text-muted">
            Returns correlation coefficients with p-values and significance.
          </p>
        </div>
      );
    }

    case 'survival-analysis': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.timeColumn as string) || ''}
                onValueChange={(v) => onChange('timeColumn', v)}
              >
                <SelectTrigger label="Time Column">
                  <SelectValue placeholder="Select time column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.eventColumn as string) || ''}
                onValueChange={(v) => onChange('eventColumn', v)}
              >
                <SelectTrigger label="Event Column (0/1)">
                  <SelectValue placeholder="Select event column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.groupColumn as string) || ''}
                onValueChange={(v) => onChange('groupColumn', v)}
              >
                <SelectTrigger label="Group Column (optional)">
                  <SelectValue placeholder="Select group column" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">No grouping</SelectItem>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Time Column"
                value={(config.timeColumn as string) || ''}
                onChange={(e) => onChange('timeColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Event Column (0/1)"
                value={(config.eventColumn as string) || ''}
                onChange={(e) => onChange('eventColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Group Column (optional)"
                value={(config.groupColumn as string) || ''}
                onChange={(e) => onChange('groupColumn', e.target.value)}
                placeholder="For comparing groups"
              />
            </>
          )}

          <p className="text-small text-text-muted">
            Kaplan-Meier analysis for time-to-event data (churn, retention, etc.)
          </p>
        </div>
      );
    }

    case 'association-rules': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.transactionColumn as string) || ''}
                onValueChange={(v) => onChange('transactionColumn', v)}
              >
                <SelectTrigger label="Transaction ID Column">
                  <SelectValue placeholder="Select transaction column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.itemColumn as string) || ''}
                onValueChange={(v) => onChange('itemColumn', v)}
              >
                <SelectTrigger label="Item Column">
                  <SelectValue placeholder="Select item column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Transaction ID Column"
                value={(config.transactionColumn as string) || ''}
                onChange={(e) => onChange('transactionColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Item Column"
                value={(config.itemColumn as string) || ''}
                onChange={(e) => onChange('itemColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            </>
          )}

          <Input
            label="Min Support"
            type="number"
            value={(config.minSupport as number) || 0.01}
            onChange={(e) => onChange('minSupport', parseFloat(e.target.value) || 0.01)}
            step={0.01}
            min={0.001}
            max={1}
          />

          <Input
            label="Min Confidence"
            type="number"
            value={(config.minConfidence as number) || 0.5}
            onChange={(e) => onChange('minConfidence', parseFloat(e.target.value) || 0.5)}
            step={0.1}
            min={0}
            max={1}
          />

          <p className="text-small text-text-muted">
            Market basket analysis. Data should have one row per item per transaction.
          </p>
        </div>
      );
    }

    case 'sentiment-analysis': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.textColumn as string) || ''}
              onValueChange={(v) => onChange('textColumn', v)}
            >
              <SelectTrigger label="Text Column">
                <SelectValue placeholder="Select text column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Text Column"
              value={(config.textColumn as string) || ''}
              onChange={(e) => onChange('textColumn', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Input
            label="Output Column"
            value={(config.outputColumn as string) || 'sentiment'}
            onChange={(e) => onChange('outputColumn', e.target.value)}
            placeholder="sentiment"
          />

          <p className="text-small text-text-muted">
            Classifies text as positive, negative, or neutral. Also adds a sentiment score.
          </p>
        </div>
      );
    }

    case 'moving-average': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select a column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Input
            label="Window Size"
            type="number"
            value={(config.window as number) || 3}
            onChange={(e) => onChange('window', parseInt(e.target.value) || 3)}
            min={2}
          />

          <Select
            value={(config.type as string) || 'simple'}
            onValueChange={(v) => onChange('type', v)}
          >
            <SelectTrigger label="Moving Average Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="simple">Simple (SMA)</SelectItem>
              <SelectItem value="exponential">Exponential (EMA)</SelectItem>
              <SelectItem value="weighted">Weighted (WMA)</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Output Column (optional)"
            value={(config.outputColumn as string) || ''}
            onChange={(e) => onChange('outputColumn', e.target.value)}
            placeholder="Auto-generated if empty"
          />
        </div>
      );
    }

    case 'train-test-split': {
      return (
        <div className="space-y-4">
          <Input
            label="Test Size (0-1)"
            type="number"
            step="0.05"
            min="0.1"
            max="0.9"
            value={(config.testSize as number) || 0.2}
            onChange={(e) => onChange('testSize', parseFloat(e.target.value) || 0.2)}
          />

          <Input
            label="Random State (seed)"
            type="number"
            value={(config.randomState as number) || 42}
            onChange={(e) => onChange('randomState', parseInt(e.target.value) || 42)}
          />

          {availableColumns.length > 0 ? (
            <Select
              value={(config.stratifyColumn as string) || ''}
              onValueChange={(v) => onChange('stratifyColumn', v)}
            >
              <SelectTrigger label="Stratify By (optional)">
                <SelectValue placeholder="No stratification" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">No stratification</SelectItem>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Stratify Column (optional)"
              value={(config.stratifyColumn as string) || ''}
              onChange={(e) => onChange('stratifyColumn', e.target.value)}
              placeholder="For balanced class distribution"
            />
          )}

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.shuffle as boolean) !== false}
              onChange={(e) => onChange('shuffle', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">Shuffle before splitting</span>
          </label>

          <p className="text-small text-text-muted">
            Adds a &apos;_split&apos; column to identify train/test rows.
          </p>
        </div>
      );
    }

    case 'model-evaluation': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.actualColumn as string) || ''}
                onValueChange={(v) => onChange('actualColumn', v)}
              >
                <SelectTrigger label="Actual (True) Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.predictedColumn as string) || ''}
                onValueChange={(v) => onChange('predictedColumn', v)}
              >
                <SelectTrigger label="Predicted Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Actual Column"
                value={(config.actualColumn as string) || ''}
                onChange={(e) => onChange('actualColumn', e.target.value)}
                placeholder="Column with true values"
              />
              <Input
                label="Predicted Column"
                value={(config.predictedColumn as string) || ''}
                onChange={(e) => onChange('predictedColumn', e.target.value)}
                placeholder="Column with predictions"
              />
            </>
          )}

          <Select
            value={(config.taskType as string) || 'classification'}
            onValueChange={(v) => onChange('taskType', v)}
          >
            <SelectTrigger label="Task Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="classification">Classification</SelectItem>
              <SelectItem value="regression">Regression</SelectItem>
            </SelectContent>
          </Select>

          <p className="text-small text-text-muted">
            Classification: Accuracy, Precision, Recall, F1. Regression: MAE, MSE, RMSE, R².
          </p>
        </div>
      );
    }

    case 'knn': {
      const knnFeatures = (config.featureColumns as string[]) || [];
      return (
        <div className="space-y-4">
          <Input
            label="K (Number of Neighbors)"
            type="number"
            min={1}
            max={50}
            value={(config.k as number) || 5}
            onChange={(e) => onChange('k', parseInt(e.target.value) || 5)}
          />

          <Select
            value={(config.taskType as string) || 'classification'}
            onValueChange={(v) => onChange('taskType', v)}
          >
            <SelectTrigger label="Task Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="classification">Classification</SelectItem>
              <SelectItem value="regression">Regression</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={(config.weights as string) || 'uniform'}
            onValueChange={(v) => onChange('weights', v)}
          >
            <SelectTrigger label="Weights">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="uniform">Uniform (equal weight)</SelectItem>
              <SelectItem value="distance">Distance (closer = more weight)</SelectItem>
            </SelectContent>
          </Select>

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Feature Columns
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label key={col} className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded">
                    <input
                      type="checkbox"
                      checked={knnFeatures.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...knnFeatures, col]
                          : knnFeatures.filter((c) => c !== col);
                        onChange('featureColumns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">Run pipeline to see columns</p>
            )}
          </div>

          {availableColumns.length > 0 ? (
            <Select
              value={(config.targetColumn as string) || ''}
              onValueChange={(v) => onChange('targetColumn', v)}
            >
              <SelectTrigger label="Target Column">
                <SelectValue placeholder="Select target" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Target Column"
              value={(config.targetColumn as string) || ''}
              onChange={(e) => onChange('targetColumn', e.target.value)}
            />
          )}

          <Input
            label="Output Column"
            value={(config.outputColumn as string) || 'knn_prediction'}
            onChange={(e) => onChange('outputColumn', e.target.value)}
          />
        </div>
      );
    }

    case 'naive-bayes': {
      const nbFeatures = (config.featureColumns as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.variant as string) || 'gaussian'}
            onValueChange={(v) => onChange('variant', v)}
          >
            <SelectTrigger label="Variant">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="gaussian">Gaussian (continuous features)</SelectItem>
              <SelectItem value="multinomial">Multinomial (count features)</SelectItem>
              <SelectItem value="bernoulli">Bernoulli (binary features)</SelectItem>
            </SelectContent>
          </Select>

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">Feature Columns</label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label key={col} className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded">
                    <input
                      type="checkbox"
                      checked={nbFeatures.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked ? [...nbFeatures, col] : nbFeatures.filter((c) => c !== col);
                        onChange('featureColumns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">Run pipeline to see columns</p>
            )}
          </div>

          {availableColumns.length > 0 ? (
            <Select
              value={(config.targetColumn as string) || ''}
              onValueChange={(v) => onChange('targetColumn', v)}
            >
              <SelectTrigger label="Target Column">
                <SelectValue placeholder="Select target" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Target Column"
              value={(config.targetColumn as string) || ''}
              onChange={(e) => onChange('targetColumn', e.target.value)}
            />
          )}

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.outputProbability as boolean) || false}
              onChange={(e) => onChange('outputProbability', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">Output class probabilities</span>
          </label>

          <Input
            label="Output Column"
            value={(config.outputColumn as string) || 'nb_prediction'}
            onChange={(e) => onChange('outputColumn', e.target.value)}
          />
        </div>
      );
    }

    case 'gradient-boosting': {
      const gbFeatures = (config.featureColumns as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.taskType as string) || 'classification'}
            onValueChange={(v) => onChange('taskType', v)}
          >
            <SelectTrigger label="Task Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="classification">Classification</SelectItem>
              <SelectItem value="regression">Regression</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Number of Estimators"
            type="number"
            min={10}
            max={500}
            value={(config.nEstimators as number) || 100}
            onChange={(e) => onChange('nEstimators', parseInt(e.target.value) || 100)}
          />

          <Input
            label="Learning Rate"
            type="number"
            step="0.01"
            min={0.01}
            max={1}
            value={(config.learningRate as number) || 0.1}
            onChange={(e) => onChange('learningRate', parseFloat(e.target.value) || 0.1)}
          />

          <Input
            label="Max Depth"
            type="number"
            min={1}
            max={20}
            value={(config.maxDepth as number) || 3}
            onChange={(e) => onChange('maxDepth', parseInt(e.target.value) || 3)}
          />

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">Feature Columns</label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label key={col} className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded">
                    <input
                      type="checkbox"
                      checked={gbFeatures.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked ? [...gbFeatures, col] : gbFeatures.filter((c) => c !== col);
                        onChange('featureColumns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">Run pipeline to see columns</p>
            )}
          </div>

          {availableColumns.length > 0 ? (
            <Select
              value={(config.targetColumn as string) || ''}
              onValueChange={(v) => onChange('targetColumn', v)}
            >
              <SelectTrigger label="Target Column">
                <SelectValue placeholder="Select target" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Target Column"
              value={(config.targetColumn as string) || ''}
              onChange={(e) => onChange('targetColumn', e.target.value)}
            />
          )}

          <Input
            label="Output Column"
            value={(config.outputColumn as string) || 'gb_prediction'}
            onChange={(e) => onChange('outputColumn', e.target.value)}
          />
        </div>
      );
    }

    case 'pareto-analysis': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.categoryColumn as string) || ''}
                onValueChange={(v) => onChange('categoryColumn', v)}
              >
                <SelectTrigger label="Category Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.valueColumn as string) || ''}
                onValueChange={(v) => onChange('valueColumn', v)}
              >
                <SelectTrigger label="Value Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Category Column"
                value={(config.categoryColumn as string) || ''}
                onChange={(e) => onChange('categoryColumn', e.target.value)}
              />
              <Input
                label="Value Column"
                value={(config.valueColumn as string) || ''}
                onChange={(e) => onChange('valueColumn', e.target.value)}
              />
            </>
          )}

          <Input
            label="Threshold (%)"
            type="number"
            min={50}
            max={95}
            value={(config.threshold as number) || 80}
            onChange={(e) => onChange('threshold', parseInt(e.target.value) || 80)}
          />

          <p className="text-small text-text-muted">
            Identifies &quot;Vital Few&quot; items contributing to the threshold % of total value.
          </p>
        </div>
      );
    }

    case 'trend-analysis': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.dateColumn as string) || ''}
                onValueChange={(v) => onChange('dateColumn', v)}
              >
                <SelectTrigger label="Date/Time Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.valueColumn as string) || ''}
                onValueChange={(v) => onChange('valueColumn', v)}
              >
                <SelectTrigger label="Value Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Date Column"
                value={(config.dateColumn as string) || ''}
                onChange={(e) => onChange('dateColumn', e.target.value)}
              />
              <Input
                label="Value Column"
                value={(config.valueColumn as string) || ''}
                onChange={(e) => onChange('valueColumn', e.target.value)}
              />
            </>
          )}

          <p className="text-small text-text-muted">
            Detects trend direction (up/down), strength, and calculates trend line values.
          </p>
        </div>
      );
    }

    case 'forecasting': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.dateColumn as string) || ''}
                onValueChange={(v) => onChange('dateColumn', v)}
              >
                <SelectTrigger label="Date/Time Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.valueColumn as string) || ''}
                onValueChange={(v) => onChange('valueColumn', v)}
              >
                <SelectTrigger label="Value Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Date Column"
                value={(config.dateColumn as string) || ''}
                onChange={(e) => onChange('dateColumn', e.target.value)}
              />
              <Input
                label="Value Column"
                value={(config.valueColumn as string) || ''}
                onChange={(e) => onChange('valueColumn', e.target.value)}
              />
            </>
          )}

          <Input
            label="Forecast Periods"
            type="number"
            min={1}
            max={100}
            value={(config.periods as number) || 10}
            onChange={(e) => onChange('periods', parseInt(e.target.value) || 10)}
          />

          <Select
            value={(config.method as string) || 'exponential'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="linear">Linear Extrapolation</SelectItem>
              <SelectItem value="exponential">Exponential Smoothing</SelectItem>
              <SelectItem value="moving_average">Moving Average</SelectItem>
            </SelectContent>
          </Select>

          {(config.method === 'exponential') && (
            <Input
              label="Alpha (smoothing factor)"
              type="number"
              step="0.05"
              min={0.05}
              max={0.95}
              value={(config.alpha as number) || 0.3}
              onChange={(e) => onChange('alpha', parseFloat(e.target.value) || 0.3)}
            />
          )}
        </div>
      );
    }

    case 'percentile-analysis': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
            />
          )}

          <Input
            label="Percentiles (comma-separated)"
            value={(config.percentiles as string) || '10,25,50,75,90,95,99'}
            onChange={(e) => onChange('percentiles', e.target.value)}
            placeholder="e.g., 10,25,50,75,90"
          />

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.addRank as boolean) !== false}
              onChange={(e) => onChange('addRank', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">Add percentile rank column</span>
          </label>

          <p className="text-small text-text-muted">
            Adds percentile rank and quartile classification to data.
          </p>
        </div>
      );
    }

    case 'distribution-fit': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
            />
          )}

          <Input
            label="Distributions to test (comma-separated)"
            value={(config.distributions as string) || 'normal,exponential,uniform'}
            onChange={(e) => onChange('distributions', e.target.value)}
            placeholder="normal,exponential,uniform"
          />

          <p className="text-small text-text-muted">
            Tests: normal, exponential, uniform. Returns fit parameters and KS test results.
          </p>
        </div>
      );
    }

    case 'text-preprocessing': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Text Column">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Text Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
            />
          )}

          <div className="space-y-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={(config.lowercase as boolean) !== false}
                onChange={(e) => onChange('lowercase', e.target.checked)}
                className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
              />
              <span className="text-small text-text-primary">Convert to lowercase</span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={(config.removePunctuation as boolean) !== false}
                onChange={(e) => onChange('removePunctuation', e.target.checked)}
                className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
              />
              <span className="text-small text-text-primary">Remove punctuation</span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={(config.removeNumbers as boolean) || false}
                onChange={(e) => onChange('removeNumbers', e.target.checked)}
                className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
              />
              <span className="text-small text-text-primary">Remove numbers</span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={(config.removeStopwords as boolean) !== false}
                onChange={(e) => onChange('removeStopwords', e.target.checked)}
                className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
              />
              <span className="text-small text-text-primary">Remove stopwords</span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={(config.trimWhitespace as boolean) !== false}
                onChange={(e) => onChange('trimWhitespace', e.target.checked)}
                className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
              />
              <span className="text-small text-text-primary">Trim whitespace</span>
            </label>
          </div>

          <Input
            label="Output Column (optional)"
            value={(config.outputColumn as string) || ''}
            onChange={(e) => onChange('outputColumn', e.target.value)}
            placeholder="Auto-generated if empty"
          />
        </div>
      );
    }

    case 'tfidf-vectorization': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Text Column">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Text Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
            />
          )}

          <Input
            label="Max Features"
            type="number"
            min={10}
            max={5000}
            value={(config.maxFeatures as number) || 100}
            onChange={(e) => onChange('maxFeatures', parseInt(e.target.value) || 100)}
          />

          <div className="grid grid-cols-2 gap-2">
            <Input
              label="N-gram Min"
              type="number"
              min={1}
              max={3}
              value={(config.ngramMin as number) || 1}
              onChange={(e) => onChange('ngramMin', parseInt(e.target.value) || 1)}
            />
            <Input
              label="N-gram Max"
              type="number"
              min={1}
              max={3}
              value={(config.ngramMax as number) || 1}
              onChange={(e) => onChange('ngramMax', parseInt(e.target.value) || 1)}
            />
          </div>

          <Select
            value={(config.outputFormat as string) || 'top_terms'}
            onValueChange={(v) => onChange('outputFormat', v)}
          >
            <SelectTrigger label="Output Format">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="top_terms">Top terms per document</SelectItem>
              <SelectItem value="matrix">Full TF-IDF matrix</SelectItem>
              <SelectItem value="summary">Term importance summary</SelectItem>
            </SelectContent>
          </Select>
        </div>
      );
    }

    case 'topic-modeling': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Text Column">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Text Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
            />
          )}

          <Input
            label="Number of Topics"
            type="number"
            min={2}
            max={20}
            value={(config.numTopics as number) || 5}
            onChange={(e) => onChange('numTopics', parseInt(e.target.value) || 5)}
          />

          <Input
            label="Words per Topic"
            type="number"
            min={3}
            max={20}
            value={(config.numWords as number) || 10}
            onChange={(e) => onChange('numWords', parseInt(e.target.value) || 10)}
          />

          <Input
            label="Max Vocabulary Size"
            type="number"
            min={100}
            max={5000}
            value={(config.maxFeatures as number) || 1000}
            onChange={(e) => onChange('maxFeatures', parseInt(e.target.value) || 1000)}
          />

          <p className="text-small text-text-muted">
            Uses LDA to discover hidden topics. Adds dominant topic and confidence to each row.
          </p>
        </div>
      );
    }

    case 'similarity-analysis': {
      const simColumns = (config.columns as string[]) || [];
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">Feature Columns</label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label key={col} className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded">
                    <input
                      type="checkbox"
                      checked={simColumns.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked ? [...simColumns, col] : simColumns.filter((c) => c !== col);
                        onChange('columns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">Run pipeline to see columns</p>
            )}
          </div>

          <Select
            value={(config.method as string) || 'cosine'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Similarity Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="cosine">Cosine Similarity</SelectItem>
              <SelectItem value="euclidean">Euclidean Distance</SelectItem>
              <SelectItem value="manhattan">Manhattan Distance</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={(config.outputType as string) || 'add_to_data'}
            onValueChange={(v) => onChange('outputType', v)}
          >
            <SelectTrigger label="Output Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="add_to_data">Add similarity scores to data</SelectItem>
              <SelectItem value="top_similar">Top N similar items</SelectItem>
              <SelectItem value="matrix">Full similarity matrix</SelectItem>
            </SelectContent>
          </Select>

          {(config.outputType === 'top_similar') && (
            <Input
              label="Top N"
              type="number"
              min={1}
              max={20}
              value={(config.topN as number) || 5}
              onChange={(e) => onChange('topN', parseInt(e.target.value) || 5)}
            />
          )}
        </div>
      );
    }

    case 'svm': {
      const svmFeatures = (config.featureColumns as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.mode as string) || 'classification'}
            onValueChange={(v) => onChange('mode', v)}
          >
            <SelectTrigger label="Mode">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="classification">Classification</SelectItem>
              <SelectItem value="regression">Regression</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={(config.kernel as string) || 'rbf'}
            onValueChange={(v) => onChange('kernel', v)}
          >
            <SelectTrigger label="Kernel">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="linear">Linear</SelectItem>
              <SelectItem value="rbf">RBF (Radial Basis Function)</SelectItem>
              <SelectItem value="poly">Polynomial</SelectItem>
              <SelectItem value="sigmoid">Sigmoid</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="C (Regularization)"
            type="number"
            step="0.1"
            min={0.01}
            max={100}
            value={(config.c as number) || 1.0}
            onChange={(e) => onChange('c', parseFloat(e.target.value) || 1.0)}
          />

          <Select
            value={(config.gamma as string) || 'scale'}
            onValueChange={(v) => onChange('gamma', v)}
          >
            <SelectTrigger label="Gamma">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="scale">Scale (1 / n_features × variance)</SelectItem>
              <SelectItem value="auto">Auto (1 / n_features)</SelectItem>
            </SelectContent>
          </Select>

          {(config.kernel === 'poly') && (
            <Input
              label="Degree (for polynomial kernel)"
              type="number"
              min={1}
              max={10}
              value={(config.degree as number) || 3}
              onChange={(e) => onChange('degree', parseInt(e.target.value) || 3)}
            />
          )}

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">Feature Columns</label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label key={col} className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded">
                    <input
                      type="checkbox"
                      checked={svmFeatures.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked ? [...svmFeatures, col] : svmFeatures.filter((c) => c !== col);
                        onChange('featureColumns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">Run pipeline to see columns</p>
            )}
          </div>

          {availableColumns.length > 0 ? (
            <Select
              value={(config.targetColumn as string) || ''}
              onValueChange={(v) => onChange('targetColumn', v)}
            >
              <SelectTrigger label="Target Column">
                <SelectValue placeholder="Select target column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Target Column"
              value={(config.targetColumn as string) || ''}
              onChange={(e) => onChange('targetColumn', e.target.value)}
              placeholder="Enter target column name"
            />
          )}

          <Input
            label="Output Column Name"
            value={(config.outputColumn as string) || 'svm_prediction'}
            onChange={(e) => onChange('outputColumn', e.target.value)}
            placeholder="svm_prediction"
          />
        </div>
      );
    }

    case 'xgboost': {
      const xgbFeatures = (config.featureColumns as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.mode as string) || 'classification'}
            onValueChange={(v) => onChange('mode', v)}
          >
            <SelectTrigger label="Mode">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="classification">Classification</SelectItem>
              <SelectItem value="regression">Regression</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Number of Estimators"
            type="number"
            min={10}
            max={1000}
            value={(config.nEstimators as number) || 100}
            onChange={(e) => onChange('nEstimators', parseInt(e.target.value) || 100)}
          />

          <Input
            label="Max Depth"
            type="number"
            min={1}
            max={20}
            value={(config.maxDepth as number) || 6}
            onChange={(e) => onChange('maxDepth', parseInt(e.target.value) || 6)}
          />

          <Input
            label="Learning Rate"
            type="number"
            step="0.01"
            min={0.01}
            max={1}
            value={(config.learningRate as number) || 0.3}
            onChange={(e) => onChange('learningRate', parseFloat(e.target.value) || 0.3)}
          />

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">Feature Columns</label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label key={col} className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded">
                    <input
                      type="checkbox"
                      checked={xgbFeatures.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked ? [...xgbFeatures, col] : xgbFeatures.filter((c) => c !== col);
                        onChange('featureColumns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">Run pipeline to see columns</p>
            )}
          </div>

          {availableColumns.length > 0 ? (
            <Select
              value={(config.targetColumn as string) || ''}
              onValueChange={(v) => onChange('targetColumn', v)}
            >
              <SelectTrigger label="Target Column">
                <SelectValue placeholder="Select target column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Target Column"
              value={(config.targetColumn as string) || ''}
              onChange={(e) => onChange('targetColumn', e.target.value)}
              placeholder="Enter target column name"
            />
          )}

          <Input
            label="Output Column Name"
            value={(config.outputColumn as string) || 'xgb_prediction'}
            onChange={(e) => onChange('outputColumn', e.target.value)}
            placeholder="xgb_prediction"
          />
        </div>
      );
    }

    case 'model-explainability': {
      const shapFeatures = (config.featureColumns as string[]) || [];
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">Feature Columns</label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label key={col} className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded">
                    <input
                      type="checkbox"
                      checked={shapFeatures.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked ? [...shapFeatures, col] : shapFeatures.filter((c) => c !== col);
                        onChange('featureColumns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">Run pipeline to see columns</p>
            )}
          </div>

          {availableColumns.length > 0 ? (
            <Select
              value={(config.targetColumn as string) || ''}
              onValueChange={(v) => onChange('targetColumn', v)}
            >
              <SelectTrigger label="Target Column">
                <SelectValue placeholder="Select target column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Target Column"
              value={(config.targetColumn as string) || ''}
              onChange={(e) => onChange('targetColumn', e.target.value)}
              placeholder="Enter target column name"
            />
          )}

          <Input
            label="Number of Samples"
            type="number"
            min={10}
            max={1000}
            value={(config.nSamples as number) || 100}
            onChange={(e) => onChange('nSamples', parseInt(e.target.value) || 100)}
          />

          <Select
            value={(config.plotType as string) || 'summary'}
            onValueChange={(v) => onChange('plotType', v)}
          >
            <SelectTrigger label="Output Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="summary">Summary (global importance)</SelectItem>
              <SelectItem value="per_sample">Per-sample contributions</SelectItem>
            </SelectContent>
          </Select>
        </div>
      );
    }

    case 'regression-diagnostics': {
      const regDiagFeatures = (config.featureColumns as string[]) || [];
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">Feature Columns</label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label key={col} className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded">
                    <input
                      type="checkbox"
                      checked={regDiagFeatures.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked ? [...regDiagFeatures, col] : regDiagFeatures.filter((c) => c !== col);
                        onChange('featureColumns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">Run pipeline to see columns</p>
            )}
          </div>

          {availableColumns.length > 0 ? (
            <Select
              value={(config.targetColumn as string) || ''}
              onValueChange={(v) => onChange('targetColumn', v)}
            >
              <SelectTrigger label="Target Column">
                <SelectValue placeholder="Select target column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Target Column"
              value={(config.targetColumn as string) || ''}
              onChange={(e) => onChange('targetColumn', e.target.value)}
              placeholder="Enter target column name"
            />
          )}

          <Input
            label="Significance Level (α)"
            type="number"
            step="0.01"
            min={0.01}
            max={0.2}
            value={(config.significanceLevel as number) || 0.05}
            onChange={(e) => onChange('significanceLevel', parseFloat(e.target.value) || 0.05)}
          />

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>Tests performed:</strong><br/>
              • Shapiro-Wilk (normality)<br/>
              • Durbin-Watson (autocorrelation)<br/>
              • Breusch-Pagan (heteroscedasticity)<br/>
              • Cook's distance (influential points)
            </p>
          </div>
        </div>
      );
    }

    case 'vif-analysis': {
      const vifFeatures = (config.featureColumns as string[]) || [];
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">Feature Columns</label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label key={col} className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded">
                    <input
                      type="checkbox"
                      checked={vifFeatures.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked ? [...vifFeatures, col] : vifFeatures.filter((c) => c !== col);
                        onChange('featureColumns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">Run pipeline to see columns</p>
            )}
          </div>

          <Input
            label="VIF Threshold"
            type="number"
            step="0.5"
            min={1}
            max={20}
            value={(config.threshold as number) || 5.0}
            onChange={(e) => onChange('threshold', parseFloat(e.target.value) || 5.0)}
          />

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>VIF interpretation:</strong><br/>
              • VIF = 1: No correlation<br/>
              • VIF 1-5: Moderate correlation<br/>
              • VIF &gt; 5: High multicollinearity<br/>
              • VIF &gt; 10: Severe multicollinearity
            </p>
          </div>
        </div>
      );
    }

    case 'funnel-analysis': {
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.stageColumn as string) || ''}
              onValueChange={(v) => onChange('stageColumn', v)}
            >
              <SelectTrigger label="Stage Column">
                <SelectValue placeholder="Select stage column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Stage Column"
              value={(config.stageColumn as string) || ''}
              onChange={(e) => onChange('stageColumn', e.target.value)}
              placeholder="Enter stage column name"
            />
          )}

          {availableColumns.length > 0 ? (
            <Select
              value={(config.countColumn as string) || ''}
              onValueChange={(v) => onChange('countColumn', v)}
            >
              <SelectTrigger label="Count Column (optional)">
                <SelectValue placeholder="Select count column" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">None (count rows)</SelectItem>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Count Column (optional)"
              value={(config.countColumn as string) || ''}
              onChange={(e) => onChange('countColumn', e.target.value)}
              placeholder="Leave empty to count rows"
            />
          )}

          <Input
            label="Stage Order (comma-separated)"
            value={(config.stageOrder as string) || ''}
            onChange={(e) => onChange('stageOrder', e.target.value)}
            placeholder="e.g., Visit, Signup, Purchase"
          />

          {availableColumns.length > 0 ? (
            <Select
              value={(config.timestampColumn as string) || ''}
              onValueChange={(v) => onChange('timestampColumn', v)}
            >
              <SelectTrigger label="Timestamp Column (optional)">
                <SelectValue placeholder="Select timestamp column" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">None</SelectItem>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Timestamp Column (optional)"
              value={(config.timestampColumn as string) || ''}
              onChange={(e) => onChange('timestampColumn', e.target.value)}
              placeholder="For time-based analysis"
            />
          )}

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>Output includes:</strong><br/>
              • Conversion rate per stage<br/>
              • Drop-off rate per stage<br/>
              • Overall funnel conversion
            </p>
          </div>
        </div>
      );
    }

    case 'pca': {
      const pcaFeatures = (config.features as string[]) || [];
      return (
        <div className="space-y-4">
          <Input
            label="Number of Components"
            type="number"
            value={(config.nComponents as number) || 2}
            onChange={(e) => onChange('nComponents', parseInt(e.target.value) || 2)}
            min={1}
          />

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.scaleData as boolean) !== false}
              onChange={(e) => onChange('scaleData', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">Scale data (recommended)</span>
          </label>

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Feature Columns
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={pcaFeatures.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...pcaFeatures, col]
                          : pcaFeatures.filter((c) => c !== col);
                        onChange('features', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          <p className="text-small text-text-muted">
            PCA reduces dimensionality while preserving variance. Select numeric columns only.
          </p>
        </div>
      );
    }

    case 'outlier-detection': {
      const outlierColumns = (config.columns as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.method as string) || 'iqr'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Detection Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="iqr">IQR (Interquartile Range)</SelectItem>
              <SelectItem value="zscore">Z-Score</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label={(config.method as string) === 'zscore' ? 'Z-Score Threshold' : 'IQR Multiplier'}
            type="number"
            step="0.1"
            value={(config.threshold as number) || ((config.method as string) === 'zscore' ? 3 : 1.5)}
            onChange={(e) => onChange('threshold', parseFloat(e.target.value) || 1.5)}
          />

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Columns to Check (empty = all numeric)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={outlierColumns.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...outlierColumns, col]
                          : outlierColumns.filter((c) => c !== col);
                        onChange('columns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          <p className="text-small text-text-muted">
            Adds an 'is_outlier' column marking rows with outliers in any selected column.
          </p>
        </div>
      );
    }

    case 'classification': {
      const classFeatures = (config.features as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.algorithm as string) || 'decision_tree'}
            onValueChange={(v) => onChange('algorithm', v)}
          >
            <SelectTrigger label="Algorithm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="decision_tree">Decision Tree</SelectItem>
              <SelectItem value="random_forest">Random Forest</SelectItem>
            </SelectContent>
          </Select>

          {availableColumns.length > 0 ? (
            <Select
              value={(config.target as string) || ''}
              onValueChange={(v) => onChange('target', v)}
            >
              <SelectTrigger label="Target Column (Y)">
                <SelectValue placeholder="Select target" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Target Column"
              value={(config.target as string) || ''}
              onChange={(e) => onChange('target', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Feature Columns (X)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns
                  .filter((col) => col !== config.target)
                  .map((col) => (
                    <label
                      key={col}
                      className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                    >
                      <input
                        type="checkbox"
                        checked={classFeatures.includes(col)}
                        onChange={(e) => {
                          const newCols = e.target.checked
                            ? [...classFeatures, col]
                            : classFeatures.filter((c) => c !== col);
                          onChange('features', newCols);
                        }}
                        className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                      />
                      <span className="text-small text-text-primary font-mono">{col}</span>
                    </label>
                  ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          <Input
            label="Test Size (0-1)"
            type="number"
            step="0.1"
            min="0.1"
            max="0.5"
            value={(config.testSize as number) || 0.2}
            onChange={(e) => onChange('testSize', parseFloat(e.target.value) || 0.2)}
          />

          <p className="text-small text-text-muted">
            Trains a classifier and returns predictions with accuracy metrics.
          </p>
        </div>
      );
    }

    case 'normality-test': {
      const normalityColumns = (config.columns as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.method as string) || 'shapiro'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Test Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="shapiro">Shapiro-Wilk</SelectItem>
              <SelectItem value="dagostino">D'Agostino-Pearson</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Significance Level (α)"
            type="number"
            step="0.01"
            min="0.01"
            max="0.1"
            value={(config.alpha as number) || 0.05}
            onChange={(e) => onChange('alpha', parseFloat(e.target.value) || 0.05)}
          />

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Columns to Test
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={normalityColumns.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...normalityColumns, col]
                          : normalityColumns.filter((c) => c !== col);
                        onChange('columns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          <p className="text-small text-text-muted">
            Tests if data follows a normal distribution. p-value &lt; α suggests non-normality.
          </p>
        </div>
      );
    }

    case 'hypothesis-testing':
      return (
        <div className="space-y-4">
          <Select
            value={(config.testType as string) || 'ttest_ind'}
            onValueChange={(v) => onChange('testType', v)}
          >
            <SelectTrigger label="Test Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ttest_ind">Independent T-Test</SelectItem>
              <SelectItem value="ttest_paired">Paired T-Test</SelectItem>
              <SelectItem value="chi2">Chi-Square Test</SelectItem>
              <SelectItem value="anova">One-way ANOVA</SelectItem>
              <SelectItem value="mannwhitney">Mann-Whitney U Test</SelectItem>
            </SelectContent>
          </Select>

          {availableColumns.length > 0 ? (
            <Select
              value={(config.column1 as string) || ''}
              onValueChange={(v) => onChange('column1', v)}
            >
              <SelectTrigger label="Column 1 / Numeric Column">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column 1"
              value={(config.column1 as string) || ''}
              onChange={(e) => onChange('column1', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          {((config.testType as string) === 'ttest_paired' || (config.testType as string) === 'chi2') && (
            availableColumns.length > 0 ? (
              <Select
                value={(config.column2 as string) || ''}
                onValueChange={(v) => onChange('column2', v)}
              >
                <SelectTrigger label="Column 2">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <Input
                label="Column 2"
                value={(config.column2 as string) || ''}
                onChange={(e) => onChange('column2', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            )
          )}

          {((config.testType as string) === 'ttest_ind' || (config.testType as string) === 'anova' || (config.testType as string) === 'mannwhitney') && (
            availableColumns.length > 0 ? (
              <Select
                value={(config.groupColumn as string) || ''}
                onValueChange={(v) => onChange('groupColumn', v)}
              >
                <SelectTrigger label="Group Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <Input
                label="Group Column"
                value={(config.groupColumn as string) || ''}
                onChange={(e) => onChange('groupColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            )
          )}

          <Input
            label="Significance Level (α)"
            type="number"
            step="0.01"
            min="0.01"
            max="0.1"
            value={(config.alpha as number) || 0.05}
            onChange={(e) => onChange('alpha', parseFloat(e.target.value) || 0.05)}
          />
        </div>
      );

    case 'time-series':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.dateColumn as string) || ''}
                onValueChange={(v) => onChange('dateColumn', v)}
              >
                <SelectTrigger label="Date Column">
                  <SelectValue placeholder="Select date column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.valueColumn as string) || ''}
                onValueChange={(v) => onChange('valueColumn', v)}
              >
                <SelectTrigger label="Value Column">
                  <SelectValue placeholder="Select value column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Date Column"
                value={(config.dateColumn as string) || ''}
                onChange={(e) => onChange('dateColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Value Column"
                value={(config.valueColumn as string) || ''}
                onChange={(e) => onChange('valueColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            </>
          )}

          <Select
            value={(config.analysis as string) || 'moving_average'}
            onValueChange={(v) => onChange('analysis', v)}
          >
            <SelectTrigger label="Analysis Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="moving_average">Moving Average</SelectItem>
              <SelectItem value="exponential_smoothing">Exponential Smoothing</SelectItem>
              <SelectItem value="trend">Trend Analysis</SelectItem>
              <SelectItem value="pct_change">Percent Change</SelectItem>
              <SelectItem value="lag_features">Lag Features</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Window Size"
            type="number"
            min="2"
            value={(config.windowSize as number) || 7}
            onChange={(e) => onChange('windowSize', parseInt(e.target.value) || 7)}
          />
        </div>
      );

    case 'feature-importance': {
      const fiFeatures = (config.features as string[]) || [];
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.target as string) || ''}
              onValueChange={(v) => onChange('target', v)}
            >
              <SelectTrigger label="Target Column">
                <SelectValue placeholder="Select target" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Target Column"
              value={(config.target as string) || ''}
              onChange={(e) => onChange('target', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Feature Columns
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.filter((col) => col !== config.target).map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={fiFeatures.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...fiFeatures, col]
                          : fiFeatures.filter((c) => c !== col);
                        onChange('features', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          <Select
            value={(config.taskType as string) || 'auto'}
            onValueChange={(v) => onChange('taskType', v)}
          >
            <SelectTrigger label="Task Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto-detect</SelectItem>
              <SelectItem value="classification">Classification</SelectItem>
              <SelectItem value="regression">Regression</SelectItem>
            </SelectContent>
          </Select>
        </div>
      );
    }

    case 'cross-validation': {
      const cvFeatures = (config.features as string[]) || [];
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.target as string) || ''}
              onValueChange={(v) => onChange('target', v)}
            >
              <SelectTrigger label="Target Column">
                <SelectValue placeholder="Select target" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Target Column"
              value={(config.target as string) || ''}
              onChange={(e) => onChange('target', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Feature Columns
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.filter((col) => col !== config.target).map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={cvFeatures.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...cvFeatures, col]
                          : cvFeatures.filter((c) => c !== col);
                        onChange('features', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          <Select
            value={(config.modelType as string) || 'random_forest'}
            onValueChange={(v) => onChange('modelType', v)}
          >
            <SelectTrigger label="Model Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="random_forest">Random Forest</SelectItem>
              <SelectItem value="logistic">Logistic Regression</SelectItem>
              <SelectItem value="linear">Linear Regression</SelectItem>
              <SelectItem value="decision_tree">Decision Tree</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Number of Folds"
            type="number"
            min="2"
            max="10"
            value={(config.nFolds as number) || 5}
            onChange={(e) => onChange('nFolds', parseInt(e.target.value) || 5)}
          />
        </div>
      );
    }

    case 'data-profiling':
      return (
        <p className="text-small text-text-muted">
          This block automatically analyzes all columns in your data. No configuration needed.
        </p>
      );

    case 'value-counts':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.normalize as boolean) || false}
              onChange={(e) => onChange('normalize', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">Show proportions instead of counts</span>
          </label>

          <Select
            value={(config.sortBy as string) || 'count'}
            onValueChange={(v) => onChange('sortBy', v)}
          >
            <SelectTrigger label="Sort By">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="count">Count (descending)</SelectItem>
              <SelectItem value="value">Value (alphabetical)</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Top N (0 for all)"
            type="number"
            min="0"
            value={(config.topN as number) || 0}
            onChange={(e) => onChange('topN', parseInt(e.target.value) || 0)}
          />
        </div>
      );

    case 'cross-tabulation':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.rowColumn as string) || ''}
                onValueChange={(v) => onChange('rowColumn', v)}
              >
                <SelectTrigger label="Row Variable">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.colColumn as string) || ''}
                onValueChange={(v) => onChange('colColumn', v)}
              >
                <SelectTrigger label="Column Variable">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Row Variable"
                value={(config.rowColumn as string) || ''}
                onChange={(e) => onChange('rowColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Column Variable"
                value={(config.colColumn as string) || ''}
                onChange={(e) => onChange('colColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            </>
          )}

          <Select
            value={(config.normalize as string) || 'none'}
            onValueChange={(v) => onChange('normalize', v)}
          >
            <SelectTrigger label="Normalize">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">No normalization (counts)</SelectItem>
              <SelectItem value="row">By row (row percentages)</SelectItem>
              <SelectItem value="column">By column (column percentages)</SelectItem>
              <SelectItem value="all">By total (overall percentages)</SelectItem>
            </SelectContent>
          </Select>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.showTotals as boolean) !== false}
              onChange={(e) => onChange('showTotals', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">Show row/column totals</span>
          </label>
        </div>
      );

    case 'scaling': {
      const scalingColumns = (config.columns as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.method as string) || 'standard'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Scaling Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="standard">Standard (Z-score)</SelectItem>
              <SelectItem value="minmax">Min-Max (0-1)</SelectItem>
              <SelectItem value="robust">Robust (median/IQR)</SelectItem>
              <SelectItem value="log">Log Transform</SelectItem>
            </SelectContent>
          </Select>

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Columns (empty = all numeric)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={scalingColumns.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...scalingColumns, col]
                          : scalingColumns.filter((c) => c !== col);
                        onChange('columns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.keepOriginal as boolean) || false}
              onChange={(e) => onChange('keepOriginal', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">Keep original columns</span>
          </label>
        </div>
      );
    }

    case 'encoding': {
      const encodingColumns = (config.columns as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.method as string) || 'onehot'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Encoding Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="onehot">One-Hot Encoding</SelectItem>
              <SelectItem value="label">Label Encoding</SelectItem>
              <SelectItem value="ordinal">Ordinal Encoding</SelectItem>
              <SelectItem value="frequency">Frequency Encoding</SelectItem>
            </SelectContent>
          </Select>

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Columns (empty = all categorical)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={encodingColumns.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...encodingColumns, col]
                          : encodingColumns.filter((c) => c !== col);
                        onChange('columns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          {(config.method as string) === 'onehot' && (
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={(config.dropFirst as boolean) || false}
                onChange={(e) => onChange('dropFirst', e.target.checked)}
                className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
              />
              <span className="text-small text-text-primary">Drop first category (avoid multicollinearity)</span>
            </label>
          )}
        </div>
      );
    }

    case 'ab-test':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.groupColumn as string) || ''}
                onValueChange={(v) => onChange('groupColumn', v)}
              >
                <SelectTrigger label="Group Column (A/B)">
                  <SelectValue placeholder="Select group column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.metricColumn as string) || ''}
                onValueChange={(v) => onChange('metricColumn', v)}
              >
                <SelectTrigger label="Metric Column">
                  <SelectValue placeholder="Select metric column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Group Column"
                value={(config.groupColumn as string) || ''}
                onChange={(e) => onChange('groupColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Metric Column"
                value={(config.metricColumn as string) || ''}
                onChange={(e) => onChange('metricColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            </>
          )}

          <Input
            label="Control Group Value"
            value={(config.controlValue as string) || ''}
            onChange={(e) => onChange('controlValue', e.target.value)}
            placeholder="e.g., control, A, 0"
          />

          <Select
            value={(config.testType as string) || 'continuous'}
            onValueChange={(v) => onChange('testType', v)}
          >
            <SelectTrigger label="Test Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="continuous">Continuous (t-test)</SelectItem>
              <SelectItem value="binary">Binary/Conversion (z-test)</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={String((config.confidenceLevel as number) || 0.95)}
            onValueChange={(v) => onChange('confidenceLevel', parseFloat(v))}
          >
            <SelectTrigger label="Confidence Level">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0.90">90%</SelectItem>
              <SelectItem value="0.95">95%</SelectItem>
              <SelectItem value="0.99">99%</SelectItem>
            </SelectContent>
          </Select>
        </div>
      );

    case 'cohort-analysis':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.userColumn as string) || ''}
                onValueChange={(v) => onChange('userColumn', v)}
              >
                <SelectTrigger label="User ID Column">
                  <SelectValue placeholder="Select user ID column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.dateColumn as string) || ''}
                onValueChange={(v) => onChange('dateColumn', v)}
              >
                <SelectTrigger label="Date Column">
                  <SelectValue placeholder="Select date column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {(config.metricType as string) !== 'retention' && (
                <Select
                  value={(config.metricColumn as string) || ''}
                  onValueChange={(v) => onChange('metricColumn', v)}
                >
                  <SelectTrigger label="Metric Column">
                    <SelectValue placeholder="Select metric column" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableColumns.map((col) => (
                      <SelectItem key={col} value={col}>{col}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </>
          ) : (
            <>
              <Input
                label="User ID Column"
                value={(config.userColumn as string) || ''}
                onChange={(e) => onChange('userColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Date Column"
                value={(config.dateColumn as string) || ''}
                onChange={(e) => onChange('dateColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              {(config.metricType as string) !== 'retention' && (
                <Input
                  label="Metric Column"
                  value={(config.metricColumn as string) || ''}
                  onChange={(e) => onChange('metricColumn', e.target.value)}
                  placeholder="Run pipeline to see columns"
                />
              )}
            </>
          )}

          <Select
            value={(config.cohortPeriod as string) || 'month'}
            onValueChange={(v) => onChange('cohortPeriod', v)}
          >
            <SelectTrigger label="Cohort Period">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="week">Weekly</SelectItem>
              <SelectItem value="month">Monthly</SelectItem>
              <SelectItem value="quarter">Quarterly</SelectItem>
              <SelectItem value="year">Yearly</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={(config.metricType as string) || 'retention'}
            onValueChange={(v) => onChange('metricType', v)}
          >
            <SelectTrigger label="Analysis Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="retention">Retention Rate</SelectItem>
              <SelectItem value="sum">Metric Sum</SelectItem>
              <SelectItem value="average">Metric Average</SelectItem>
            </SelectContent>
          </Select>
        </div>
      );

    case 'rfm-analysis':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.customerColumn as string) || ''}
                onValueChange={(v) => onChange('customerColumn', v)}
              >
                <SelectTrigger label="Customer ID Column">
                  <SelectValue placeholder="Select customer ID column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.dateColumn as string) || ''}
                onValueChange={(v) => onChange('dateColumn', v)}
              >
                <SelectTrigger label="Transaction Date Column">
                  <SelectValue placeholder="Select date column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.revenueColumn as string) || ''}
                onValueChange={(v) => onChange('revenueColumn', v)}
              >
                <SelectTrigger label="Revenue/Amount Column">
                  <SelectValue placeholder="Select revenue column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Customer ID Column"
                value={(config.customerColumn as string) || ''}
                onChange={(e) => onChange('customerColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Transaction Date Column"
                value={(config.dateColumn as string) || ''}
                onChange={(e) => onChange('dateColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Revenue/Amount Column"
                value={(config.revenueColumn as string) || ''}
                onChange={(e) => onChange('revenueColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            </>
          )}

          <Select
            value={String((config.segments as number) || 5)}
            onValueChange={(v) => onChange('segments', parseInt(v))}
          >
            <SelectTrigger label="Number of Segments">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="3">3 (Low/Medium/High)</SelectItem>
              <SelectItem value="4">4 Quartiles</SelectItem>
              <SelectItem value="5">5 Quintiles</SelectItem>
            </SelectContent>
          </Select>
        </div>
      );

    case 'correlation-matrix': {
      const corrColumns = (config.columns as string[]) || [];
      return (
        <div className="space-y-4">
          <Select
            value={(config.method as string) || 'pearson'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Correlation Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="pearson">Pearson</SelectItem>
              <SelectItem value="spearman">Spearman</SelectItem>
              <SelectItem value="kendall">Kendall</SelectItem>
            </SelectContent>
          </Select>

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Columns (empty = all numeric)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={corrColumns.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...corrColumns, col]
                          : corrColumns.filter((c) => c !== col);
                        onChange('columns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>
        </div>
      );
    }

    case 'violin-plot':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.column as string) || ''}
                onValueChange={(v) => onChange('column', v)}
              >
                <SelectTrigger label="Numeric Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.groupColumn as string) || '__none__'}
                onValueChange={(v) => onChange('groupColumn', v === '__none__' ? '' : v)}
              >
                <SelectTrigger label="Group By (optional)">
                  <SelectValue placeholder="None" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">None</SelectItem>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Numeric Column"
                value={(config.column as string) || ''}
                onChange={(e) => onChange('column', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Group By (optional)"
                value={(config.groupColumn as string) || ''}
                onChange={(e) => onChange('groupColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            </>
          )}

          <Input
            label="Title"
            value={(config.title as string) || ''}
            onChange={(e) => onChange('title', e.target.value)}
            placeholder="Distribution chart title"
          />
        </div>
      );

    case 'pair-plot': {
      const pairColumns = (config.columns as string[]) || [];
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Columns (empty = first 5 numeric)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={pairColumns.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...pairColumns, col]
                          : pairColumns.filter((c) => c !== col);
                        onChange('columns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          {availableColumns.length > 0 && (
            <Select
              value={(config.colorColumn as string) || '__none__'}
              onValueChange={(v) => onChange('colorColumn', v === '__none__' ? '' : v)}
            >
              <SelectTrigger label="Color By (optional)">
                <SelectValue placeholder="None" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">None</SelectItem>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
      );
    }

    case 'area-chart':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.x as string) || ''}
                onValueChange={(v) => onChange('x', v)}
              >
                <SelectTrigger label="X Axis Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.y as string) || ''}
                onValueChange={(v) => onChange('y', v)}
              >
                <SelectTrigger label="Y Axis Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.color as string) || '__none__'}
                onValueChange={(v) => onChange('color', v === '__none__' ? '' : v)}
              >
                <SelectTrigger label="Group By (optional)">
                  <SelectValue placeholder="None" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">None</SelectItem>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="X Axis"
                value={(config.x as string) || ''}
                onChange={(e) => onChange('x', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Y Axis"
                value={(config.y as string) || ''}
                onChange={(e) => onChange('y', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            </>
          )}

          <Input
            label="Title"
            value={(config.title as string) || ''}
            onChange={(e) => onChange('title', e.target.value)}
            placeholder="Chart title"
          />
        </div>
      );

    case 'stacked-chart': {
      const stackedColumns = (config.yColumns as string[]) || [];
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.x as string) || ''}
              onValueChange={(v) => onChange('x', v)}
            >
              <SelectTrigger label="X Axis Column">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="X Axis"
              value={(config.x as string) || ''}
              onChange={(e) => onChange('x', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">
              Y Columns (to stack)
            </label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.filter((col) => col !== config.x).map((col) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded"
                  >
                    <input
                      type="checkbox"
                      checked={stackedColumns.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...stackedColumns, col]
                          : stackedColumns.filter((c) => c !== col);
                        onChange('yColumns', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">
                Run pipeline to see columns
              </p>
            )}
          </div>

          <Select
            value={(config.chartType as string) || 'bar'}
            onValueChange={(v) => onChange('chartType', v)}
          >
            <SelectTrigger label="Chart Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="bar">Stacked Bar</SelectItem>
              <SelectItem value="area">Stacked Area</SelectItem>
            </SelectContent>
          </Select>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.normalize as boolean) || false}
              onChange={(e) => onChange('normalize', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">Normalize to 100%</span>
          </label>

          <Input
            label="Title"
            value={(config.title as string) || ''}
            onChange={(e) => onChange('title', e.target.value)}
            placeholder="Chart title"
          />
        </div>
      );
    }

    case 'bubble-chart':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.x as string) || ''}
                onValueChange={(v) => onChange('x', v)}
              >
                <SelectTrigger label="X Axis Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.y as string) || ''}
                onValueChange={(v) => onChange('y', v)}
              >
                <SelectTrigger label="Y Axis Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.size as string) || ''}
                onValueChange={(v) => onChange('size', v)}
              >
                <SelectTrigger label="Size Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.color as string) || '__none__'}
                onValueChange={(v) => onChange('color', v === '__none__' ? '' : v)}
              >
                <SelectTrigger label="Color By (optional)">
                  <SelectValue placeholder="None" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">None</SelectItem>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="X Axis"
                value={(config.x as string) || ''}
                onChange={(e) => onChange('x', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Y Axis"
                value={(config.y as string) || ''}
                onChange={(e) => onChange('y', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Size"
                value={(config.size as string) || ''}
                onChange={(e) => onChange('size', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            </>
          )}

          <Input
            label="Title"
            value={(config.title as string) || ''}
            onChange={(e) => onChange('title', e.target.value)}
            placeholder="Chart title"
          />
        </div>
      );

    case 'qq-plot':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Run pipeline to see columns"
            />
          )}

          <Input
            label="Title"
            value={(config.title as string) || ''}
            onChange={(e) => onChange('title', e.target.value)}
            placeholder="Chart title"
          />

          <p className="text-small text-text-muted">
            Q-Q plot compares sample quantiles to theoretical normal distribution. Points on the diagonal line indicate normality.
          </p>
        </div>
      );

    case 'confusion-matrix':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.actualColumn as string) || ''}
                onValueChange={(v) => onChange('actualColumn', v)}
              >
                <SelectTrigger label="Actual Labels Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.predictedColumn as string) || ''}
                onValueChange={(v) => onChange('predictedColumn', v)}
              >
                <SelectTrigger label="Predicted Labels Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Actual Labels"
                value={(config.actualColumn as string) || ''}
                onChange={(e) => onChange('actualColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Predicted Labels"
                value={(config.predictedColumn as string) || ''}
                onChange={(e) => onChange('predictedColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            </>
          )}

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.normalize as boolean) || false}
              onChange={(e) => onChange('normalize', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <span className="text-small text-text-primary">Normalize values</span>
          </label>

          <Input
            label="Title"
            value={(config.title as string) || ''}
            onChange={(e) => onChange('title', e.target.value)}
            placeholder="Chart title"
          />
        </div>
      );

    case 'roc-curve':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.actualColumn as string) || ''}
                onValueChange={(v) => onChange('actualColumn', v)}
              >
                <SelectTrigger label="Actual Labels Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.probabilityColumn as string) || ''}
                onValueChange={(v) => onChange('probabilityColumn', v)}
              >
                <SelectTrigger label="Probability/Score Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Actual Labels"
                value={(config.actualColumn as string) || ''}
                onChange={(e) => onChange('actualColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
              <Input
                label="Probability/Score"
                value={(config.probabilityColumn as string) || ''}
                onChange={(e) => onChange('probabilityColumn', e.target.value)}
                placeholder="Run pipeline to see columns"
              />
            </>
          )}

          <Input
            label="Positive Class (optional)"
            value={(config.positiveClass as string) || ''}
            onChange={(e) => onChange('positiveClass', e.target.value)}
            placeholder="Leave empty for auto-detect"
          />

          <Input
            label="Title"
            value={(config.title as string) || ''}
            onChange={(e) => onChange('title', e.target.value)}
            placeholder="Chart title"
          />

          <p className="text-small text-text-muted">
            ROC curve shows the trade-off between true positive rate and false positive rate. AUC closer to 1 is better.
          </p>
        </div>
      );

    case 'customer-ltv':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.customerIdColumn as string) || ''}
                onValueChange={(v) => onChange('customerIdColumn', v)}
              >
                <SelectTrigger label="Customer ID Column">
                  <SelectValue placeholder="Select customer ID column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.transactionDateColumn as string) || ''}
                onValueChange={(v) => onChange('transactionDateColumn', v)}
              >
                <SelectTrigger label="Transaction Date Column">
                  <SelectValue placeholder="Select date column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.monetaryColumn as string) || ''}
                onValueChange={(v) => onChange('monetaryColumn', v)}
              >
                <SelectTrigger label="Monetary/Revenue Column">
                  <SelectValue placeholder="Select monetary column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Customer ID Column"
                value={(config.customerIdColumn as string) || ''}
                onChange={(e) => onChange('customerIdColumn', e.target.value)}
                placeholder="Enter customer ID column name"
              />
              <Input
                label="Transaction Date Column"
                value={(config.transactionDateColumn as string) || ''}
                onChange={(e) => onChange('transactionDateColumn', e.target.value)}
                placeholder="Enter date column name"
              />
              <Input
                label="Monetary/Revenue Column"
                value={(config.monetaryColumn as string) || ''}
                onChange={(e) => onChange('monetaryColumn', e.target.value)}
                placeholder="Enter monetary column name"
              />
            </>
          )}

          <Input
            label="Projection Periods (months)"
            type="number"
            min={1}
            max={60}
            value={(config.projectionPeriods as number) || 12}
            onChange={(e) => onChange('projectionPeriods', parseInt(e.target.value) || 12)}
          />

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>Outputs:</strong> CLV per customer, predicted purchases, customer segments (Low/Medium/High/Premium), and churn risk assessment.
            </p>
          </div>
        </div>
      );

    case 'churn-analysis': {
      const churnFeatures = (config.features as string[]) || [];
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="block text-small font-medium text-text-secondary">Feature Columns</label>
            {availableColumns.length > 0 ? (
              <div className="space-y-2 max-h-40 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                {availableColumns.map((col) => (
                  <label key={col} className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded">
                    <input
                      type="checkbox"
                      checked={churnFeatures.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked ? [...churnFeatures, col] : churnFeatures.filter((c) => c !== col);
                        onChange('features', newCols);
                      }}
                      className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
                    />
                    <span className="text-small text-text-primary font-mono">{col}</span>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-small text-text-muted bg-bg-tertiary p-3 rounded-lg">Run pipeline to see columns</p>
            )}
          </div>

          {availableColumns.length > 0 ? (
            <Select
              value={(config.targetColumn as string) || ''}
              onValueChange={(v) => onChange('targetColumn', v)}
            >
              <SelectTrigger label="Target Column (Churn)">
                <SelectValue placeholder="Select churn indicator column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Target Column (Churn)"
              value={(config.targetColumn as string) || ''}
              onChange={(e) => onChange('targetColumn', e.target.value)}
              placeholder="Enter target column name"
            />
          )}

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="handleImbalance"
              checked={(config.handleImbalance as boolean) !== false}
              onChange={(e) => onChange('handleImbalance', e.target.checked)}
              className="w-4 h-4 rounded border-border-default text-electric-indigo focus:ring-electric-indigo"
            />
            <label htmlFor="handleImbalance" className="text-small text-text-primary">
              Handle class imbalance
            </label>
          </div>

          <Input
            label="Classification Threshold"
            type="number"
            step="0.1"
            min={0}
            max={1}
            value={(config.threshold as number) || 0.5}
            onChange={(e) => onChange('threshold', parseFloat(e.target.value) || 0.5)}
          />

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>Outputs:</strong> Churn probability, prediction, risk segments, and top 5 churn drivers with importance scores.
            </p>
          </div>
        </div>
      );
    }

    case 'growth-metrics':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.dateColumn as string) || ''}
                onValueChange={(v) => onChange('dateColumn', v)}
              >
                <SelectTrigger label="Date Column">
                  <SelectValue placeholder="Select date column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.valueColumn as string) || ''}
                onValueChange={(v) => onChange('valueColumn', v)}
              >
                <SelectTrigger label="Value Column">
                  <SelectValue placeholder="Select value column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Date Column"
                value={(config.dateColumn as string) || ''}
                onChange={(e) => onChange('dateColumn', e.target.value)}
                placeholder="Enter date column name"
              />
              <Input
                label="Value Column"
                value={(config.valueColumn as string) || ''}
                onChange={(e) => onChange('valueColumn', e.target.value)}
                placeholder="Enter value column name"
              />
            </>
          )}

          <Select
            value={(config.granularity as string) || 'month'}
            onValueChange={(v) => onChange('granularity', v)}
          >
            <SelectTrigger label="Time Granularity">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="day">Daily</SelectItem>
              <SelectItem value="week">Weekly</SelectItem>
              <SelectItem value="month">Monthly</SelectItem>
              <SelectItem value="quarter">Quarterly</SelectItem>
              <SelectItem value="year">Yearly</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Rolling Window (periods)"
            type="number"
            min={2}
            max={24}
            value={(config.rollingWindow as number) || 3}
            onChange={(e) => onChange('rollingWindow', parseInt(e.target.value) || 3)}
          />

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>Outputs:</strong> Period-over-period growth, YoY growth, rolling average, CAGR, growth acceleration, and trend classification.
            </p>
          </div>
        </div>
      );

    case 'attribution-modeling':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.userIdColumn as string) || ''}
                onValueChange={(v) => onChange('userIdColumn', v)}
              >
                <SelectTrigger label="User ID Column">
                  <SelectValue placeholder="Select user ID column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.channelColumn as string) || ''}
                onValueChange={(v) => onChange('channelColumn', v)}
              >
                <SelectTrigger label="Channel Column">
                  <SelectValue placeholder="Select channel/source column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.conversionColumn as string) || ''}
                onValueChange={(v) => onChange('conversionColumn', v)}
              >
                <SelectTrigger label="Conversion Column">
                  <SelectValue placeholder="Select conversion value column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.timestampColumn as string) || ''}
                onValueChange={(v) => onChange('timestampColumn', v)}
              >
                <SelectTrigger label="Timestamp Column (optional)">
                  <SelectValue placeholder="Select timestamp column" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">None</SelectItem>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="User ID Column"
                value={(config.userIdColumn as string) || ''}
                onChange={(e) => onChange('userIdColumn', e.target.value)}
                placeholder="Enter user ID column name"
              />
              <Input
                label="Channel Column"
                value={(config.channelColumn as string) || ''}
                onChange={(e) => onChange('channelColumn', e.target.value)}
                placeholder="Enter channel column name"
              />
              <Input
                label="Conversion Column"
                value={(config.conversionColumn as string) || ''}
                onChange={(e) => onChange('conversionColumn', e.target.value)}
                placeholder="Enter conversion column name"
              />
              <Input
                label="Timestamp Column (optional)"
                value={(config.timestampColumn as string) || ''}
                onChange={(e) => onChange('timestampColumn', e.target.value)}
                placeholder="Enter timestamp column name"
              />
            </>
          )}

          <Select
            value={(config.model as string) || 'linear'}
            onValueChange={(v) => onChange('model', v)}
          >
            <SelectTrigger label="Attribution Model">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="first-touch">First Touch</SelectItem>
              <SelectItem value="last-touch">Last Touch</SelectItem>
              <SelectItem value="linear">Linear</SelectItem>
              <SelectItem value="time-decay">Time Decay</SelectItem>
              <SelectItem value="position-based">Position Based (40/20/40)</SelectItem>
            </SelectContent>
          </Select>

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>Outputs:</strong> Attribution credit per channel, efficiency scores, and channel rankings.
            </p>
          </div>
        </div>
      );

    case 'breakeven-analysis':
      return (
        <div className="space-y-4">
          <Input
            label="Fixed Costs ($)"
            type="number"
            min={0}
            value={(config.fixedCosts as number) || 10000}
            onChange={(e) => onChange('fixedCosts', parseFloat(e.target.value) || 0)}
          />

          <Input
            label="Variable Cost per Unit ($)"
            type="number"
            min={0}
            step="0.01"
            value={(config.variableCostPerUnit as number) || 5}
            onChange={(e) => onChange('variableCostPerUnit', parseFloat(e.target.value) || 0)}
          />

          <Input
            label="Price per Unit ($)"
            type="number"
            min={0}
            step="0.01"
            value={(config.pricePerUnit as number) || 15}
            onChange={(e) => onChange('pricePerUnit', parseFloat(e.target.value) || 0)}
          />

          <Input
            label="Current Sales (units)"
            type="number"
            min={0}
            value={(config.currentSalesUnits as number) || 0}
            onChange={(e) => onChange('currentSalesUnits', parseInt(e.target.value) || 0)}
          />

          <Input
            label="Scenario Range (+/- units)"
            type="number"
            min={10}
            max={1000}
            value={(config.scenarioRange as number) || 50}
            onChange={(e) => onChange('scenarioRange', parseInt(e.target.value) || 50)}
          />

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>Outputs:</strong> Break-even units/revenue, contribution margin, margin of safety, and scenario analysis.
            </p>
          </div>
        </div>
      );

    case 'confidence-intervals':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.column as string) || ''}
                onValueChange={(v) => onChange('column', v)}
              >
                <SelectTrigger label="Primary Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.column2 as string) || ''}
                onValueChange={(v) => onChange('column2', v)}
              >
                <SelectTrigger label="Second Column (for two-sample/paired)">
                  <SelectValue placeholder="Select second column" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">None</SelectItem>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Primary Column"
                value={(config.column as string) || ''}
                onChange={(e) => onChange('column', e.target.value)}
                placeholder="Enter column name"
              />
              <Input
                label="Second Column (for two-sample/paired)"
                value={(config.column2 as string) || ''}
                onChange={(e) => onChange('column2', e.target.value)}
                placeholder="Enter second column name"
              />
            </>
          )}

          <Select
            value={(config.analysisType as string) || 'one-sample-mean'}
            onValueChange={(v) => onChange('analysisType', v)}
          >
            <SelectTrigger label="Analysis Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="one-sample-mean">One-Sample Mean</SelectItem>
              <SelectItem value="one-sample-proportion">One-Sample Proportion</SelectItem>
              <SelectItem value="two-sample-mean">Two-Sample Mean Difference</SelectItem>
              <SelectItem value="paired">Paired Samples</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={String((config.confidenceLevel as number) || 0.95)}
            onValueChange={(v) => onChange('confidenceLevel', parseFloat(v))}
          >
            <SelectTrigger label="Confidence Level">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0.90">90%</SelectItem>
              <SelectItem value="0.95">95%</SelectItem>
              <SelectItem value="0.99">99%</SelectItem>
            </SelectContent>
          </Select>

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>Outputs:</strong> Point estimate, CI bounds, margin of error, and interpretation.
            </p>
          </div>
        </div>
      );

    case 'bootstrap-analysis':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.column as string) || ''}
                onValueChange={(v) => onChange('column', v)}
              >
                <SelectTrigger label="Column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.column2 as string) || ''}
                onValueChange={(v) => onChange('column2', v)}
              >
                <SelectTrigger label="Second Column (for correlation)">
                  <SelectValue placeholder="Select second column" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">None</SelectItem>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Column"
                value={(config.column as string) || ''}
                onChange={(e) => onChange('column', e.target.value)}
                placeholder="Enter column name"
              />
              <Input
                label="Second Column (for correlation)"
                value={(config.column2 as string) || ''}
                onChange={(e) => onChange('column2', e.target.value)}
                placeholder="Enter second column name"
              />
            </>
          )}

          <Select
            value={(config.statistic as string) || 'mean'}
            onValueChange={(v) => onChange('statistic', v)}
          >
            <SelectTrigger label="Statistic">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="mean">Mean</SelectItem>
              <SelectItem value="median">Median</SelectItem>
              <SelectItem value="std">Standard Deviation</SelectItem>
              <SelectItem value="correlation">Correlation</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Number of Iterations"
            type="number"
            min={100}
            max={10000}
            value={(config.nIterations as number) || 1000}
            onChange={(e) => onChange('nIterations', parseInt(e.target.value) || 1000)}
          />

          <Select
            value={String((config.confidenceLevel as number) || 0.95)}
            onValueChange={(v) => onChange('confidenceLevel', parseFloat(v))}
          >
            <SelectTrigger label="Confidence Level">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0.90">90%</SelectItem>
              <SelectItem value="0.95">95%</SelectItem>
              <SelectItem value="0.99">99%</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={(config.method as string) || 'percentile'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="CI Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="percentile">Percentile</SelectItem>
              <SelectItem value="bca">BCa (Bias-Corrected)</SelectItem>
            </SelectContent>
          </Select>

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>Outputs:</strong> Bootstrap CI, standard error, bias estimate, and bias-corrected estimate.
            </p>
          </div>
        </div>
      );

    case 'posthoc-tests':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.valueColumn as string) || ''}
                onValueChange={(v) => onChange('valueColumn', v)}
              >
                <SelectTrigger label="Value Column">
                  <SelectValue placeholder="Select value column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.groupColumn as string) || ''}
                onValueChange={(v) => onChange('groupColumn', v)}
              >
                <SelectTrigger label="Group Column">
                  <SelectValue placeholder="Select group column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Value Column"
                value={(config.valueColumn as string) || ''}
                onChange={(e) => onChange('valueColumn', e.target.value)}
                placeholder="Enter value column name"
              />
              <Input
                label="Group Column"
                value={(config.groupColumn as string) || ''}
                onChange={(e) => onChange('groupColumn', e.target.value)}
                placeholder="Enter group column name"
              />
            </>
          )}

          <Select
            value={(config.method as string) || 'tukey'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Correction Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="tukey">Tukey HSD</SelectItem>
              <SelectItem value="bonferroni">Bonferroni</SelectItem>
              <SelectItem value="holm">Holm</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={String((config.alpha as number) || 0.05)}
            onValueChange={(v) => onChange('alpha', parseFloat(v))}
          >
            <SelectTrigger label="Significance Level (α)">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0.01">0.01</SelectItem>
              <SelectItem value="0.05">0.05</SelectItem>
              <SelectItem value="0.10">0.10</SelectItem>
            </SelectContent>
          </Select>

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>Outputs:</strong> Pairwise comparisons, adjusted p-values, significance flags, and Cohen's d effect sizes.
            </p>
          </div>
        </div>
      );

    case 'power-analysis':
      return (
        <div className="space-y-4">
          <Select
            value={(config.testType as string) || 't-test'}
            onValueChange={(v) => onChange('testType', v)}
          >
            <SelectTrigger label="Test Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="t-test">T-Test</SelectItem>
              <SelectItem value="anova">ANOVA</SelectItem>
              <SelectItem value="chi-square">Chi-Square</SelectItem>
              <SelectItem value="correlation">Correlation</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={(config.calculateFor as string) || 'sample_size'}
            onValueChange={(v) => onChange('calculateFor', v)}
          >
            <SelectTrigger label="Calculate For">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="sample_size">Required Sample Size</SelectItem>
              <SelectItem value="power">Achieved Power</SelectItem>
              <SelectItem value="effect_size">Detectable Effect Size</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Effect Size (Cohen's d/f/w/r)"
            type="number"
            min={0.01}
            max={3}
            step="0.1"
            value={(config.effectSize as number) || 0.5}
            onChange={(e) => onChange('effectSize', parseFloat(e.target.value) || 0.5)}
          />

          <Select
            value={String((config.alpha as number) || 0.05)}
            onValueChange={(v) => onChange('alpha', parseFloat(v))}
          >
            <SelectTrigger label="Alpha (α)">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0.01">0.01</SelectItem>
              <SelectItem value="0.05">0.05</SelectItem>
              <SelectItem value="0.10">0.10</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Target Power"
            type="number"
            min={0.5}
            max={0.99}
            step="0.05"
            value={(config.power as number) || 0.8}
            onChange={(e) => onChange('power', parseFloat(e.target.value) || 0.8)}
          />

          <Input
            label="Sample Size (if calculating power/effect)"
            type="number"
            min={5}
            max={10000}
            value={(config.sampleSize as number) || 30}
            onChange={(e) => onChange('sampleSize', parseInt(e.target.value) || 30)}
          />

          <Input
            label="Number of Groups (for ANOVA)"
            type="number"
            min={2}
            max={10}
            value={(config.groups as number) || 2}
            onChange={(e) => onChange('groups', parseInt(e.target.value) || 2)}
          />

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>Outputs:</strong> Required sample size or achieved power, plus power curve data.
            </p>
          </div>
        </div>
      );

    case 'bayesian-inference':
      return (
        <div className="space-y-4">
          <Select
            value={(config.analysisType as string) || 'proportion'}
            onValueChange={(v) => onChange('analysisType', v)}
          >
            <SelectTrigger label="Analysis Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="proportion">Proportion (Beta-Binomial)</SelectItem>
              <SelectItem value="mean">Mean (Normal)</SelectItem>
              <SelectItem value="ab-test">A/B Test</SelectItem>
            </SelectContent>
          </Select>

          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.column as string) || ''}
                onValueChange={(v) => onChange('column', v)}
              >
                <SelectTrigger label="Column (or Group A)">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.column2 as string) || ''}
                onValueChange={(v) => onChange('column2', v)}
              >
                <SelectTrigger label="Column 2 / Group B (for A/B test)">
                  <SelectValue placeholder="Select second column" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">None</SelectItem>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Column (or Group A)"
                value={(config.column as string) || ''}
                onChange={(e) => onChange('column', e.target.value)}
                placeholder="Enter column name"
              />
              <Input
                label="Column 2 / Group B (for A/B test)"
                value={(config.column2 as string) || ''}
                onChange={(e) => onChange('column2', e.target.value)}
                placeholder="Enter second column name"
              />
            </>
          )}

          <Input
            label="Prior Alpha"
            type="number"
            min={0.01}
            step="0.1"
            value={(config.priorAlpha as number) || 1}
            onChange={(e) => onChange('priorAlpha', parseFloat(e.target.value) || 1)}
          />

          <Input
            label="Prior Beta"
            type="number"
            min={0.01}
            step="0.1"
            value={(config.priorBeta as number) || 1}
            onChange={(e) => onChange('priorBeta', parseFloat(e.target.value) || 1)}
          />

          <Select
            value={String((config.credibleLevel as number) || 0.95)}
            onValueChange={(v) => onChange('credibleLevel', parseFloat(v))}
          >
            <SelectTrigger label="Credible Level">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0.90">90%</SelectItem>
              <SelectItem value="0.95">95%</SelectItem>
              <SelectItem value="0.99">99%</SelectItem>
            </SelectContent>
          </Select>

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>Outputs:</strong> Posterior distributions, credible intervals, P(B&gt;A) for A/B tests.
            </p>
          </div>
        </div>
      );

    case 'data-quality-score':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <div className="space-y-2">
              <label className="text-small font-medium text-text-primary">
                Columns to Analyze (leave empty for all)
              </label>
              <div className="max-h-32 overflow-y-auto border border-border-primary rounded-lg p-2">
                {availableColumns.map((col) => (
                  <label key={col} className="flex items-center gap-2 py-1">
                    <input
                      type="checkbox"
                      checked={((config.columns as string[]) || []).includes(col)}
                      onChange={(e) => {
                        const current = (config.columns as string[]) || [];
                        if (e.target.checked) {
                          onChange('columns', [...current, col]);
                        } else {
                          onChange('columns', current.filter((c) => c !== col));
                        }
                      }}
                      className="rounded border-border-primary"
                    />
                    <span className="text-small">{col}</span>
                  </label>
                ))}
              </div>
            </div>
          ) : (
            <Input
              label="Columns (comma-separated, empty for all)"
              value={((config.columns as string[]) || []).join(', ')}
              onChange={(e) => onChange('columns', e.target.value.split(',').map((s) => s.trim()).filter(Boolean))}
              placeholder="col1, col2, col3"
            />
          )}

          <Input
            label="Completeness Weight"
            type="number"
            min={0}
            max={1}
            step="0.1"
            value={(config.completenessWeight as number) || 0.3}
            onChange={(e) => onChange('completenessWeight', parseFloat(e.target.value) || 0.3)}
          />

          <Input
            label="Validity Weight"
            type="number"
            min={0}
            max={1}
            step="0.1"
            value={(config.validityWeight as number) || 0.3}
            onChange={(e) => onChange('validityWeight', parseFloat(e.target.value) || 0.3)}
          />

          <Input
            label="Uniqueness Weight"
            type="number"
            min={0}
            max={1}
            step="0.1"
            value={(config.uniquenessWeight as number) || 0.2}
            onChange={(e) => onChange('uniquenessWeight', parseFloat(e.target.value) || 0.2)}
          />

          <Input
            label="Consistency Weight"
            type="number"
            min={0}
            max={1}
            step="0.1"
            value={(config.consistencyWeight as number) || 0.2}
            onChange={(e) => onChange('consistencyWeight', parseFloat(e.target.value) || 0.2)}
          />

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>Outputs:</strong> Overall quality score (0-100), dimension scores, specific issues with row/column references.
            </p>
          </div>
        </div>
      );

    case 'changepoint-detection':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.valueColumn as string) || ''}
                onValueChange={(v) => onChange('valueColumn', v)}
              >
                <SelectTrigger label="Value Column">
                  <SelectValue placeholder="Select value column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.dateColumn as string) || ''}
                onValueChange={(v) => onChange('dateColumn', v)}
              >
                <SelectTrigger label="Date Column (optional)">
                  <SelectValue placeholder="Select date column" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">None</SelectItem>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Value Column"
                value={(config.valueColumn as string) || ''}
                onChange={(e) => onChange('valueColumn', e.target.value)}
                placeholder="Enter value column name"
              />
              <Input
                label="Date Column (optional)"
                value={(config.dateColumn as string) || ''}
                onChange={(e) => onChange('dateColumn', e.target.value)}
                placeholder="Enter date column name"
              />
            </>
          )}

          <Select
            value={(config.method as string) || 'cusum'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Detection Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="cusum">CUSUM</SelectItem>
              <SelectItem value="pelt">PELT</SelectItem>
              <SelectItem value="binary">Binary Segmentation</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Minimum Segment Size"
            type="number"
            min={5}
            max={100}
            value={(config.minSegmentSize as number) || 10}
            onChange={(e) => onChange('minSegmentSize', parseInt(e.target.value) || 10)}
          />

          <Input
            label="Maximum Change Points"
            type="number"
            min={1}
            max={20}
            value={(config.maxChangePoints as number) || 5}
            onChange={(e) => onChange('maxChangePoints', parseInt(e.target.value) || 5)}
          />

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              <strong>Outputs:</strong> Change point locations, segment statistics before/after, confidence scores.
            </p>
          </div>
        </div>
      );

    case 'log-transform':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Enter column name"
            />
          )}

          <Select
            value={(config.operation as string) || 'log'}
            onValueChange={(v) => onChange('operation', v)}
          >
            <SelectTrigger label="Operation">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="log">Natural Log (ln)</SelectItem>
              <SelectItem value="log10">Log Base 10</SelectItem>
              <SelectItem value="log2">Log Base 2</SelectItem>
              <SelectItem value="log1p">Log(1+x)</SelectItem>
              <SelectItem value="exp">Exponential (e^x)</SelectItem>
              <SelectItem value="sqrt">Square Root</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={(config.handleZero as string) || 'add_one'}
            onValueChange={(v) => onChange('handleZero', v)}
          >
            <SelectTrigger label="Handle Zero/Negative">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="add_one">Add 1 before transform</SelectItem>
              <SelectItem value="replace_min">Replace with min positive/2</SelectItem>
              <SelectItem value="skip">Skip (result in NaN)</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="New Column Name (optional)"
            value={(config.newColumn as string) || ''}
            onChange={(e) => onChange('newColumn', e.target.value)}
            placeholder="Leave empty for auto-naming"
          />
        </div>
      );

    case 'interpolate-missing':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.column as string) || ''}
                onValueChange={(v) => onChange('column', v)}
              >
                <SelectTrigger label="Column to Interpolate">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.orderColumn as string) || ''}
                onValueChange={(v) => onChange('orderColumn', v)}
              >
                <SelectTrigger label="Order By Column (optional)">
                  <SelectValue placeholder="Select order column" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">None</SelectItem>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Column to Interpolate"
                value={(config.column as string) || ''}
                onChange={(e) => onChange('column', e.target.value)}
                placeholder="Enter column name"
              />
              <Input
                label="Order By Column (optional)"
                value={(config.orderColumn as string) || ''}
                onChange={(e) => onChange('orderColumn', e.target.value)}
                placeholder="Enter order column name"
              />
            </>
          )}

          <Select
            value={(config.method as string) || 'linear'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Interpolation Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="linear">Linear</SelectItem>
              <SelectItem value="polynomial">Polynomial (order 2)</SelectItem>
              <SelectItem value="spline">Spline (cubic)</SelectItem>
              <SelectItem value="nearest">Nearest</SelectItem>
              <SelectItem value="time">Time-based</SelectItem>
              <SelectItem value="ffill">Forward Fill</SelectItem>
              <SelectItem value="bfill">Backward Fill</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Limit (0 = no limit)"
            type="number"
            min={0}
            value={(config.limit as number) || 0}
            onChange={(e) => onChange('limit', parseInt(e.target.value) || 0)}
          />
        </div>
      );

    case 'date-truncate':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Date Column">
                <SelectValue placeholder="Select date column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Date Column"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Enter date column name"
            />
          )}

          <Select
            value={(config.unit as string) || 'day'}
            onValueChange={(v) => onChange('unit', v)}
          >
            <SelectTrigger label="Truncate To">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="minute">Minute Start</SelectItem>
              <SelectItem value="hour">Hour Start</SelectItem>
              <SelectItem value="day">Day Start</SelectItem>
              <SelectItem value="week">Week Start</SelectItem>
              <SelectItem value="month">Month Start</SelectItem>
              <SelectItem value="quarter">Quarter Start</SelectItem>
              <SelectItem value="year">Year Start</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="New Column Name (optional)"
            value={(config.newColumn as string) || ''}
            onChange={(e) => onChange('newColumn', e.target.value)}
            placeholder="Leave empty for auto-naming"
          />
        </div>
      );

    case 'period-over-period':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.dateColumn as string) || ''}
                onValueChange={(v) => onChange('dateColumn', v)}
              >
                <SelectTrigger label="Date Column">
                  <SelectValue placeholder="Select date column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.valueColumn as string) || ''}
                onValueChange={(v) => onChange('valueColumn', v)}
              >
                <SelectTrigger label="Value Column">
                  <SelectValue placeholder="Select value column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Date Column"
                value={(config.dateColumn as string) || ''}
                onChange={(e) => onChange('dateColumn', e.target.value)}
                placeholder="Enter date column name"
              />
              <Input
                label="Value Column"
                value={(config.valueColumn as string) || ''}
                onChange={(e) => onChange('valueColumn', e.target.value)}
                placeholder="Enter value column name"
              />
            </>
          )}

          <Select
            value={(config.period as string) || 'mom'}
            onValueChange={(v) => onChange('period', v)}
          >
            <SelectTrigger label="Period Comparison">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="dod">Day over Day (DoD)</SelectItem>
              <SelectItem value="wow">Week over Week (WoW)</SelectItem>
              <SelectItem value="mom">Month over Month (MoM)</SelectItem>
              <SelectItem value="qoq">Quarter over Quarter (QoQ)</SelectItem>
              <SelectItem value="yoy">Year over Year (YoY)</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={(config.changeType as string) || 'percent'}
            onValueChange={(v) => onChange('changeType', v)}
          >
            <SelectTrigger label="Change Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="percent">Percentage Change</SelectItem>
              <SelectItem value="absolute">Absolute Change</SelectItem>
              <SelectItem value="both">Both</SelectItem>
            </SelectContent>
          </Select>
        </div>
      );

    case 'hash-column':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.column as string) || ''}
              onValueChange={(v) => onChange('column', v)}
            >
              <SelectTrigger label="Column to Hash">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Column to Hash"
              value={(config.column as string) || ''}
              onChange={(e) => onChange('column', e.target.value)}
              placeholder="Enter column name"
            />
          )}

          <Select
            value={(config.algorithm as string) || 'sha256'}
            onValueChange={(v) => onChange('algorithm', v)}
          >
            <SelectTrigger label="Hash Algorithm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="sha256">SHA-256</SelectItem>
              <SelectItem value="sha512">SHA-512</SelectItem>
              <SelectItem value="sha1">SHA-1</SelectItem>
              <SelectItem value="md5">MD5</SelectItem>
              <SelectItem value="blake2">BLAKE2</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Truncate Length (0 = full)"
            type="number"
            min={0}
            max={128}
            value={(config.truncateLength as number) || 0}
            onChange={(e) => onChange('truncateLength', parseInt(e.target.value) || 0)}
          />

          <Input
            label="New Column Name (optional)"
            value={(config.newColumn as string) || ''}
            onChange={(e) => onChange('newColumn', e.target.value)}
            placeholder="Leave empty for auto-naming"
          />
        </div>
      );

    case 'expand-date-range':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <Select
              value={(config.dateColumn as string) || ''}
              onValueChange={(v) => onChange('dateColumn', v)}
            >
              <SelectTrigger label="Date Column">
                <SelectValue placeholder="Select date column" />
              </SelectTrigger>
              <SelectContent>
                {availableColumns.map((col) => (
                  <SelectItem key={col} value={col}>{col}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              label="Date Column"
              value={(config.dateColumn as string) || ''}
              onChange={(e) => onChange('dateColumn', e.target.value)}
              placeholder="Enter date column name"
            />
          )}

          <Select
            value={(config.freq as string) || 'D'}
            onValueChange={(v) => onChange('freq', v)}
          >
            <SelectTrigger label="Frequency">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="H">Hourly</SelectItem>
              <SelectItem value="D">Daily</SelectItem>
              <SelectItem value="W">Weekly</SelectItem>
              <SelectItem value="M">Monthly</SelectItem>
              <SelectItem value="Q">Quarterly</SelectItem>
              <SelectItem value="Y">Yearly</SelectItem>
            </SelectContent>
          </Select>

          <Select
            value={(config.fillMethod as string) || 'ffill'}
            onValueChange={(v) => onChange('fillMethod', v)}
          >
            <SelectTrigger label="Fill Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ffill">Forward Fill</SelectItem>
              <SelectItem value="bfill">Backward Fill</SelectItem>
              <SelectItem value="zero">Fill with Zero</SelectItem>
              <SelectItem value="interpolate">Linear Interpolation</SelectItem>
            </SelectContent>
          </Select>
        </div>
      );

    case 'string-similarity':
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <Select
                value={(config.column1 as string) || ''}
                onValueChange={(v) => onChange('column1', v)}
              >
                <SelectTrigger label="First Column">
                  <SelectValue placeholder="Select first column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={(config.column2 as string) || ''}
                onValueChange={(v) => onChange('column2', v)}
              >
                <SelectTrigger label="Second Column">
                  <SelectValue placeholder="Select second column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="First Column"
                value={(config.column1 as string) || ''}
                onChange={(e) => onChange('column1', e.target.value)}
                placeholder="Enter first column name"
              />
              <Input
                label="Second Column"
                value={(config.column2 as string) || ''}
                onChange={(e) => onChange('column2', e.target.value)}
                placeholder="Enter second column name"
              />
            </>
          )}

          <Select
            value={(config.method as string) || 'levenshtein'}
            onValueChange={(v) => onChange('method', v)}
          >
            <SelectTrigger label="Similarity Method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="levenshtein">Levenshtein</SelectItem>
              <SelectItem value="jaro">Jaro</SelectItem>
              <SelectItem value="jaro_winkler">Jaro-Winkler</SelectItem>
              <SelectItem value="exact">Exact Match</SelectItem>
              <SelectItem value="contains">Contains</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Match Threshold (0-1)"
            type="number"
            min={0}
            max={1}
            step={0.1}
            value={(config.threshold as number) || 0.8}
            onChange={(e) => onChange('threshold', parseFloat(e.target.value) || 0.8)}
          />

          <Input
            label="Output Column Name"
            value={(config.newColumn as string) || 'similarity'}
            onChange={(e) => onChange('newColumn', e.target.value)}
          />
        </div>
      );

    case 'generate-sequence':
      return (
        <div className="space-y-4">
          <Select
            value={(config.type as string) || 'number'}
            onValueChange={(v) => onChange('type', v)}
          >
            <SelectTrigger label="Sequence Type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="number">Number Range</SelectItem>
              <SelectItem value="date">Date Range</SelectItem>
              <SelectItem value="repeat">Repeat Value</SelectItem>
            </SelectContent>
          </Select>

          <Input
            label="Column Name"
            value={(config.columnName as string) || 'sequence'}
            onChange={(e) => onChange('columnName', e.target.value)}
          />

          {(config.type as string) === 'number' && (
            <>
              <Input
                label="Start"
                type="number"
                value={(config.start as number) || 1}
                onChange={(e) => onChange('start', parseFloat(e.target.value) || 1)}
              />
              <Input
                label="End"
                type="number"
                value={(config.end as number) || 10}
                onChange={(e) => onChange('end', parseFloat(e.target.value) || 10)}
              />
              <Input
                label="Step"
                type="number"
                value={(config.step as number) || 1}
                onChange={(e) => onChange('step', parseFloat(e.target.value) || 1)}
              />
            </>
          )}

          {(config.type as string) === 'date' && (
            <>
              <Input
                label="Start Date"
                type="date"
                value={(config.dateStart as string) || ''}
                onChange={(e) => onChange('dateStart', e.target.value)}
              />
              <Input
                label="End Date"
                type="date"
                value={(config.dateEnd as string) || ''}
                onChange={(e) => onChange('dateEnd', e.target.value)}
              />
              <Select
                value={(config.dateFreq as string) || 'D'}
                onValueChange={(v) => onChange('dateFreq', v)}
              >
                <SelectTrigger label="Frequency">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="H">Hourly</SelectItem>
                  <SelectItem value="D">Daily</SelectItem>
                  <SelectItem value="W">Weekly</SelectItem>
                  <SelectItem value="M">Monthly</SelectItem>
                </SelectContent>
              </Select>
            </>
          )}

          {(config.type as string) === 'repeat' && (
            <>
              <Input
                label="Value to Repeat"
                value={(config.repeatValue as string) || ''}
                onChange={(e) => onChange('repeatValue', e.target.value)}
                placeholder="Enter value"
              />
              <Input
                label="Repeat Count"
                type="number"
                min={1}
                value={(config.repeatCount as number) || 10}
                onChange={(e) => onChange('repeatCount', parseInt(e.target.value) || 10)}
              />
            </>
          )}

          <div className="p-3 bg-bg-tertiary rounded-lg">
            <p className="text-small text-text-muted">
              This block generates a new dataset with no input required.
            </p>
          </div>
        </div>
      );

    case 'top-n-per-group': {
      const topNGroupCols = (config.groupColumns as string[]) || [];
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <div className="space-y-2">
                <label className="block text-small font-medium text-text-secondary">
                  Group Columns (optional)
                </label>
                <div className="space-y-2 max-h-32 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                  {availableColumns.map((col) => (
                    <label key={col} className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded">
                      <input
                        type="checkbox"
                        checked={topNGroupCols.includes(col)}
                        onChange={(e) => {
                          const newCols = e.target.checked
                            ? [...topNGroupCols, col]
                            : topNGroupCols.filter((c) => c !== col);
                          onChange('groupColumns', newCols);
                        }}
                        className="rounded border-border-default"
                      />
                      <span className="text-small text-text-primary">{col}</span>
                    </label>
                  ))}
                </div>
              </div>

              <Select
                value={(config.orderColumn as string) || ''}
                onValueChange={(v) => onChange('orderColumn', v)}
              >
                <SelectTrigger label="Order By Column">
                  <SelectValue placeholder="Select column to order by" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Group Columns (comma-separated)"
                value={topNGroupCols.join(', ')}
                onChange={(e) => onChange('groupColumns', e.target.value.split(',').map(s => s.trim()).filter(Boolean))}
                placeholder="e.g., category, region"
              />
              <Input
                label="Order By Column"
                value={(config.orderColumn as string) || ''}
                onChange={(e) => onChange('orderColumn', e.target.value)}
                placeholder="Enter column name"
              />
            </>
          )}

          <Input
            label="Number of Rows (N)"
            type="number"
            min={1}
            value={(config.n as number) || 5}
            onChange={(e) => onChange('n', parseInt(e.target.value) || 5)}
          />

          <Select
            value={String(config.ascending || false)}
            onValueChange={(v) => onChange('ascending', v === 'true')}
          >
            <SelectTrigger label="Order Direction">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="false">Top N (Descending)</SelectItem>
              <SelectItem value="true">Bottom N (Ascending)</SelectItem>
            </SelectContent>
          </Select>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={(config.includeRank as boolean) !== false}
              onChange={(e) => onChange('includeRank', e.target.checked)}
              className="rounded border-border-default"
            />
            <span className="text-small text-text-primary">Include rank column</span>
          </label>
        </div>
      );
    }

    case 'first-last-per-group': {
      const flGroupCols = (config.groupColumns as string[]) || [];
      return (
        <div className="space-y-4">
          {availableColumns.length > 0 ? (
            <>
              <div className="space-y-2">
                <label className="block text-small font-medium text-text-secondary">
                  Group Columns
                </label>
                <div className="space-y-2 max-h-32 overflow-y-auto bg-bg-tertiary rounded-lg p-3">
                  {availableColumns.map((col) => (
                    <label key={col} className="flex items-center gap-2 cursor-pointer hover:bg-bg-secondary p-1.5 rounded">
                      <input
                        type="checkbox"
                        checked={flGroupCols.includes(col)}
                        onChange={(e) => {
                          const newCols = e.target.checked
                            ? [...flGroupCols, col]
                            : flGroupCols.filter((c) => c !== col);
                          onChange('groupColumns', newCols);
                        }}
                        className="rounded border-border-default"
                      />
                      <span className="text-small text-text-primary">{col}</span>
                    </label>
                  ))}
                </div>
              </div>

              <Select
                value={(config.orderColumn as string) || ''}
                onValueChange={(v) => onChange('orderColumn', v)}
              >
                <SelectTrigger label="Order By Column (optional)">
                  <SelectValue placeholder="Select column to order by" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">None</SelectItem>
                  {availableColumns.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          ) : (
            <>
              <Input
                label="Group Columns (comma-separated)"
                value={flGroupCols.join(', ')}
                onChange={(e) => onChange('groupColumns', e.target.value.split(',').map(s => s.trim()).filter(Boolean))}
                placeholder="e.g., customer_id, product"
              />
              <Input
                label="Order By Column (optional)"
                value={(config.orderColumn as string) || ''}
                onChange={(e) => onChange('orderColumn', e.target.value)}
                placeholder="Enter column name"
              />
            </>
          )}

          <Select
            value={(config.position as string) || 'first'}
            onValueChange={(v) => onChange('position', v)}
          >
            <SelectTrigger label="Position">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="first">First Row</SelectItem>
              <SelectItem value="last">Last Row</SelectItem>
              <SelectItem value="both">Both First and Last</SelectItem>
            </SelectContent>
          </Select>
        </div>
      );
    }

    default:
      return (
        <p className="text-small text-text-muted">
          No configuration options for this block type.
        </p>
      );
  }
}

interface VisualizationPanelProps {
  block: PipelineBlock | null;
  result: { success: boolean; data?: unknown; error?: string } | null;
}

function VisualizationPanel({ block, result }: VisualizationPanelProps) {
  if (!block) {
    return (
      <div className="text-center text-text-muted py-8">
        <BarChart3 size={48} className="mx-auto mb-4 opacity-50" />
        <p>Select a block to see visualization</p>
      </div>
    );
  }

  if (!result?.success || !result.data) {
    return (
      <div className="text-center text-text-muted py-8">
        <p>Run the pipeline to see visualization</p>
      </div>
    );
  }

  const { type } = block.data;

  // Chart block visualization
  if (type === 'chart') {
    const chartData = result.data as {
      chartType: 'bar' | 'line' | 'scatter' | 'pie' | 'histogram' | 'box' | 'heatmap';
      data: Record<string, unknown>[];
      x: string;
      y: string;
      color?: string;
      title?: string;
    };

    return (
      <div className="h-[400px]">
        <ChartVisualization chartData={chartData} width={280} height={350} />
      </div>
    );
  }

  // Table block visualization
  if (type === 'table') {
    const tableData = result.data as {
      data: Record<string, unknown>[];
      columns: string[];
      dtypes: Record<string, string>;
      rowCount: number;
    };

    return (
      <div className="h-[400px]">
        <TableVisualization tableData={tableData} pageSize={25} maxHeight={380} />
      </div>
    );
  }

  // For other blocks that output DataFrames, show as a simple table
  if (Array.isArray(result.data) && result.data.length > 0) {
    const data = result.data as Record<string, unknown>[];
    const columns = Object.keys(data[0]);
    const dtypes: Record<string, string> = {};
    columns.forEach((col) => {
      const val = data[0][col];
      if (typeof val === 'number') {
        dtypes[col] = Number.isInteger(val) ? 'int64' : 'float64';
      } else if (typeof val === 'boolean') {
        dtypes[col] = 'bool';
      } else {
        dtypes[col] = 'object';
      }
    });

    return (
      <div className="h-[400px]">
        <TableVisualization
          tableData={{
            data,
            columns,
            dtypes,
            rowCount: data.length,
          }}
          pageSize={25}
          maxHeight={380}
        />
      </div>
    );
  }

  // For other data types, show JSON
  return (
    <div className="text-center text-text-muted py-8">
      <Table2 size={48} className="mx-auto mb-4 opacity-50" />
      <p>No visualization available for this block type</p>
      <pre className="mt-4 text-left text-small bg-bg-tertiary p-3 rounded-lg overflow-auto max-h-60">
        {JSON.stringify(result.data, null, 2)}
      </pre>
    </div>
  );
}

// Helper function to check if data is an export file result
function isExportData(data: unknown): data is { content: string; filename: string; mimeType: string } {
  return (
    typeof data === 'object' &&
    data !== null &&
    'content' in data &&
    'filename' in data &&
    'mimeType' in data &&
    typeof (data as Record<string, unknown>).content === 'string' &&
    typeof (data as Record<string, unknown>).filename === 'string' &&
    typeof (data as Record<string, unknown>).mimeType === 'string'
  );
}

// Helper function to download export data
function downloadExportFile(data: { content: string; filename: string; mimeType: string }) {
  try {
    // Decode base64 content
    const binaryString = atob(data.content);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    // Create blob and trigger download
    const blob = new Blob([bytes], { type: data.mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = data.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Failed to download file:', error);
  }
}

function DataPreview({ data }: { data: unknown }) {
  if (!data) {
    return <p className="text-text-muted">No data available</p>;
  }

  // Handle export file data (CSV export block output)
  if (isExportData(data)) {
    return (
      <div className="text-center py-8">
        <Download size={48} className="mx-auto mb-4 text-accent-teal opacity-70" />
        <p className="text-text-primary font-medium mb-2">File Ready for Download</p>
        <p className="text-small text-text-muted mb-4">{data.filename}</p>
        <Button
          variant="primary"
          onClick={() => downloadExportFile(data)}
          className="inline-flex items-center gap-2"
        >
          <Download size={16} />
          Download CSV
        </Button>
      </div>
    );
  }

  if (Array.isArray(data) && data.length > 0 && typeof data[0] === 'object') {
    const columns = Object.keys(data[0] as Record<string, unknown>);
    const rows = data.slice(0, 50);

    return (
      <div className="overflow-auto">
        <table className="w-full text-small">
          <thead>
            <tr className="border-b border-border-default">
              {columns.map((col) => (
                <th
                  key={col}
                  className="px-2 py-1 text-left text-text-secondary font-medium"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-border-default/50">
                {columns.map((col) => (
                  <td key={col} className="px-2 py-1 text-text-primary">
                    {String((row as Record<string, unknown>)[col] ?? '')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {data.length > 50 && (
          <p className="text-small text-text-muted mt-2 text-center">
            Showing 50 of {data.length} rows
          </p>
        )}
      </div>
    );
  }

  return (
    <pre className="text-small text-text-primary bg-bg-tertiary p-3 rounded-lg overflow-auto">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}
