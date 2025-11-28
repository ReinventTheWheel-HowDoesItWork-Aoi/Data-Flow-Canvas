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
            ? blockDefinitions[selectedBlock.data.type].label
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

  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-small font-medium text-text-primary mb-1">
          {definition.label}
        </h4>
        <p className="text-small text-text-muted">{definition.description}</p>
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
