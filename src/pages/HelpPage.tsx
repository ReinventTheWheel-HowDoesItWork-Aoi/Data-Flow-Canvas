/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React from 'react';
import { Link } from 'react-router-dom';
import {
  ChevronLeft,
  FileUp,
  Database,
  PenLine,
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
  Play,
  MousePointer,
  Link2,
  Lightbulb,
  Workflow,
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
import { cn } from '@/lib/utils/cn';

interface BlockInfo {
  icon: React.ElementType;
  name: string;
  description: string;
  category: string;
}

const blocks: BlockInfo[] = [
  // Data Input (3 blocks)
  { icon: FileUp, name: 'Load CSV', description: 'Import data from CSV files', category: 'Data Input' },
  { icon: Database, name: 'Sample Data', description: 'Use built-in sample datasets (Iris, Wine, Diabetes)', category: 'Data Input' },
  { icon: PenLine, name: 'Create Dataset', description: 'Manually enter data in CSV format', category: 'Data Input' },

  // Transform (17 blocks)
  { icon: Filter, name: 'Filter Rows', description: 'Filter data based on conditions (equals, greater than, contains, etc.)', category: 'Transform' },
  { icon: Columns, name: 'Select Columns', description: 'Choose, reorder, or rename specific columns', category: 'Transform' },
  { icon: ArrowUpDown, name: 'Sort', description: 'Sort data by one or more columns', category: 'Transform' },
  { icon: Group, name: 'Group & Aggregate', description: 'Group by columns and apply sum, mean, count, etc.', category: 'Transform' },
  { icon: GitMerge, name: 'Join', description: 'Merge two datasets on common columns', category: 'Transform' },
  { icon: Plus, name: 'Derive Column', description: 'Create new columns using expressions', category: 'Transform' },
  { icon: Eraser, name: 'Handle Missing', description: 'Drop rows or fill missing values', category: 'Transform' },
  { icon: TextCursorInput, name: 'Rename Columns', description: 'Rename one or more columns', category: 'Transform' },
  { icon: Copy, name: 'Deduplicate', description: 'Remove duplicate rows', category: 'Transform' },
  { icon: Shuffle, name: 'Sample Rows', description: 'Randomly sample rows from data', category: 'Transform' },
  { icon: ListFilter, name: 'Limit Rows', description: 'Get first or last N rows', category: 'Transform' },
  { icon: RotateCcw, name: 'Pivot', description: 'Reshape data from long to wide format', category: 'Transform' },
  { icon: RotateCw, name: 'Unpivot', description: 'Reshape data from wide to long format', category: 'Transform' },
  { icon: Layers, name: 'Union', description: 'Stack datasets vertically (append rows)', category: 'Transform' },
  { icon: Scissors, name: 'Split Column', description: 'Split a column by delimiter into multiple columns', category: 'Transform' },
  { icon: Combine, name: 'Merge Columns', description: 'Combine multiple columns into one', category: 'Transform' },
  { icon: GitBranch, name: 'Conditional Column', description: 'Create column based on if/else logic', category: 'Transform' },

  // Analysis (16 blocks)
  { icon: BarChart3, name: 'Statistics', description: 'Calculate descriptive statistics and correlations', category: 'Analysis' },
  { icon: TrendingUp, name: 'Regression', description: 'Perform linear or logistic regression', category: 'Analysis' },
  { icon: Network, name: 'Clustering', description: 'K-means or hierarchical clustering', category: 'Analysis' },
  { icon: Minimize2, name: 'PCA', description: 'Principal Component Analysis for dimensionality reduction', category: 'Analysis' },
  { icon: AlertTriangle, name: 'Outlier Detection', description: 'Detect outliers using IQR or Z-score methods', category: 'Analysis' },
  { icon: GitFork, name: 'Classification', description: 'Train a decision tree or random forest classifier', category: 'Analysis' },
  { icon: Activity, name: 'Normality Test', description: 'Test if data follows a normal distribution', category: 'Analysis' },
  { icon: FlaskConical, name: 'Hypothesis Testing', description: 'Perform t-test, chi-square, ANOVA, Mann-Whitney', category: 'Analysis' },
  { icon: Clock, name: 'Time Series Analysis', description: 'Analyze trends, seasonality, and moving averages', category: 'Analysis' },
  { icon: Award, name: 'Feature Importance', description: 'Calculate feature importance using Random Forest', category: 'Analysis' },
  { icon: Repeat, name: 'Cross-Validation', description: 'Evaluate model performance with k-fold CV', category: 'Analysis' },
  { icon: FileSearch, name: 'Data Profiling', description: 'Generate comprehensive data quality profile', category: 'Analysis' },
  { icon: Hash, name: 'Value Counts', description: 'Count occurrences of each unique value', category: 'Analysis' },
  { icon: Grid3x3, name: 'Cross-Tabulation', description: 'Create frequency table between categorical columns', category: 'Analysis' },
  { icon: Scale, name: 'Scaling / Normalization', description: 'Scale features using standard, minmax, or robust methods', category: 'Analysis' },
  { icon: Binary, name: 'Encoding', description: 'Encode categorical variables (one-hot, label, ordinal)', category: 'Analysis' },
  { icon: FlaskConical, name: 'A/B Test Analysis', description: 'Statistical analysis for A/B experiments with lift and significance', category: 'Analysis' },
  { icon: Users, name: 'Cohort Analysis', description: 'Analyze user retention and behavior by cohort over time', category: 'Analysis' },
  { icon: Target, name: 'RFM Analysis', description: 'Segment customers by Recency, Frequency, Monetary value', category: 'Analysis' },

  // Visualization (11 blocks)
  { icon: PieChart, name: 'Chart', description: 'Create bar, line, scatter, pie, and histogram charts', category: 'Visualization' },
  { icon: Table, name: 'Table', description: 'Display data in an interactive table', category: 'Visualization' },
  { icon: Grid2x2, name: 'Correlation Matrix', description: 'Visualize correlations as a heatmap', category: 'Visualization' },
  { icon: Music, name: 'Violin Plot', description: 'Show distribution of data with violin plots', category: 'Visualization' },
  { icon: LayoutGrid, name: 'Pair Plot', description: 'Create scatter matrix showing pairwise relationships', category: 'Visualization' },
  { icon: AreaChart, name: 'Area Chart', description: 'Create filled area charts for time series', category: 'Visualization' },
  { icon: Layers, name: 'Stacked Bar/Area', description: 'Create stacked charts for composition', category: 'Visualization' },
  { icon: Circle, name: 'Bubble Chart', description: 'Scatter plot with size dimension', category: 'Visualization' },
  { icon: ScatterChart, name: 'Q-Q Plot', description: 'Quantile-Quantile plot to check normality', category: 'Visualization' },
  { icon: Grid3x3, name: 'Confusion Matrix', description: 'Visualize classification results', category: 'Visualization' },
  { icon: TrendingUp, name: 'ROC Curve', description: 'ROC curve for binary classification evaluation', category: 'Visualization' },

  // Output (1 block)
  { icon: Download, name: 'Export CSV', description: 'Export data to CSV format and download', category: 'Output' },
];

export default function HelpPage() {
  return (
    <div className="min-h-screen bg-bg-primary">
      {/* Header */}
      <header className="bg-bg-secondary border-b border-border-default">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center gap-4">
          <Link
            to="/editor"
            className="flex items-center gap-2 text-text-muted hover:text-text-primary transition-colors"
          >
            <ChevronLeft size={20} />
            <span>Back to Editor</span>
          </Link>
          <div className="flex-1" />
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-electric-indigo to-soft-violet flex items-center justify-center">
              <Workflow size={18} className="text-white" />
            </div>
            <span className="font-semibold text-text-primary">Data Flow Canvas</span>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-5xl mx-auto px-6 py-8">
        <h1 className="text-h1 text-text-primary mb-2">Getting Started</h1>
        <p className="text-text-secondary mb-8">
          Learn how to use Data Flow Canvas to build visual data pipelines.
        </p>

        {/* Quick Start */}
        <section className="mb-12">
          <h2 className="text-h2 text-text-primary mb-4 flex items-center gap-2">
            <Lightbulb className="text-golden-amber" />
            Quick Start Guide
          </h2>
          <div className="bg-bg-secondary rounded-xl border border-border-default p-6">
            <ol className="space-y-4">
              <li className="flex gap-4">
                <span className="flex-shrink-0 w-8 h-8 rounded-full bg-electric-indigo text-white flex items-center justify-center font-semibold">1</span>
                <div>
                  <h3 className="font-semibold text-text-primary">Add Data Input</h3>
                  <p className="text-text-secondary text-small">
                    Drag a <strong>Load Data</strong>, <strong>Sample Data</strong>, or <strong>Create Dataset</strong> block from the sidebar onto the canvas.
                  </p>
                </div>
              </li>
              <li className="flex gap-4">
                <span className="flex-shrink-0 w-8 h-8 rounded-full bg-electric-indigo text-white flex items-center justify-center font-semibold">2</span>
                <div>
                  <h3 className="font-semibold text-text-primary">Configure Your Block</h3>
                  <p className="text-text-secondary text-small">
                    Click on the block to select it. The right panel will show configuration options. Upload a file, select a sample dataset, or enter data manually.
                  </p>
                </div>
              </li>
              <li className="flex gap-4">
                <span className="flex-shrink-0 w-8 h-8 rounded-full bg-electric-indigo text-white flex items-center justify-center font-semibold">3</span>
                <div>
                  <h3 className="font-semibold text-text-primary">Add Transform Blocks</h3>
                  <p className="text-text-secondary text-small">
                    Drag transform blocks (Filter, Sort, etc.) and connect them by dragging from the output handle (right) of one block to the input handle (left) of another.
                  </p>
                </div>
              </li>
              <li className="flex gap-4">
                <span className="flex-shrink-0 w-8 h-8 rounded-full bg-electric-indigo text-white flex items-center justify-center font-semibold">4</span>
                <div>
                  <h3 className="font-semibold text-text-primary">Run Your Pipeline</h3>
                  <p className="text-text-secondary text-small">
                    Click the <strong>Run</strong> button in the top bar or press <kbd className="px-1.5 py-0.5 bg-bg-tertiary rounded text-small">Ctrl+Enter</kbd>. The first run will take ~30 seconds to load Python.
                  </p>
                </div>
              </li>
              <li className="flex gap-4">
                <span className="flex-shrink-0 w-8 h-8 rounded-full bg-electric-indigo text-white flex items-center justify-center font-semibold">5</span>
                <div>
                  <h3 className="font-semibold text-text-primary">View Results</h3>
                  <p className="text-text-secondary text-small">
                    Select a block and check the <strong>Preview</strong> or <strong>Viz</strong> tab in the right panel to see the output data.
                  </p>
                </div>
              </li>
            </ol>
          </div>
        </section>

        {/* Keyboard Shortcuts */}
        <section className="mb-12">
          <h2 className="text-h2 text-text-primary mb-4">Keyboard Shortcuts</h2>
          <div className="bg-bg-secondary rounded-xl border border-border-default overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border-default">
                  <th className="text-left px-4 py-3 text-small font-semibold text-text-secondary">Shortcut</th>
                  <th className="text-left px-4 py-3 text-small font-semibold text-text-secondary">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-default">
                <tr>
                  <td className="px-4 py-3"><kbd className="px-2 py-1 bg-bg-tertiary rounded text-small">Ctrl+Enter</kbd></td>
                  <td className="px-4 py-3 text-text-secondary">Run pipeline</td>
                </tr>
                <tr>
                  <td className="px-4 py-3"><kbd className="px-2 py-1 bg-bg-tertiary rounded text-small">Ctrl+Z</kbd></td>
                  <td className="px-4 py-3 text-text-secondary">Undo</td>
                </tr>
                <tr>
                  <td className="px-4 py-3"><kbd className="px-2 py-1 bg-bg-tertiary rounded text-small">Ctrl+Shift+Z</kbd> or <kbd className="px-2 py-1 bg-bg-tertiary rounded text-small">Ctrl+Y</kbd></td>
                  <td className="px-4 py-3 text-text-secondary">Redo</td>
                </tr>
                <tr>
                  <td className="px-4 py-3"><kbd className="px-2 py-1 bg-bg-tertiary rounded text-small">Delete</kbd> or <kbd className="px-2 py-1 bg-bg-tertiary rounded text-small">Backspace</kbd></td>
                  <td className="px-4 py-3 text-text-secondary">Delete selected blocks</td>
                </tr>
                <tr>
                  <td className="px-4 py-3"><kbd className="px-2 py-1 bg-bg-tertiary rounded text-small">Escape</kbd></td>
                  <td className="px-4 py-3 text-text-secondary">Clear selection</td>
                </tr>
                <tr>
                  <td className="px-4 py-3"><kbd className="px-2 py-1 bg-bg-tertiary rounded text-small">Ctrl+J</kbd></td>
                  <td className="px-4 py-3 text-text-secondary">Toggle bottom panel (logs)</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Block Reference */}
        <section className="mb-12">
          <h2 className="text-h2 text-text-primary mb-4">Block Reference</h2>
          {['Data Input', 'Transform', 'Analysis', 'Visualization', 'Output'].map((category) => (
            <div key={category} className="mb-6">
              <h3 className="text-h3 text-text-primary mb-3">{category}</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {blocks
                  .filter((b) => b.category === category)
                  .map((block) => (
                    <div
                      key={block.name}
                      className="bg-bg-secondary rounded-lg border border-border-default p-4 flex items-start gap-3"
                    >
                      <div className={cn(
                        'p-2 rounded-lg',
                        category === 'Data Input' && 'bg-electric-indigo/10 text-electric-indigo',
                        category === 'Transform' && 'bg-soft-violet/10 text-soft-violet',
                        category === 'Analysis' && 'bg-fresh-teal/10 text-fresh-teal',
                        category === 'Visualization' && 'bg-golden-amber/10 text-golden-amber',
                        category === 'Output' && 'bg-warm-coral/10 text-warm-coral',
                      )}>
                        <block.icon size={20} />
                      </div>
                      <div>
                        <h4 className="font-semibold text-text-primary">{block.name}</h4>
                        <p className="text-small text-text-secondary">{block.description}</p>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          ))}
        </section>

        {/* Tips */}
        <section className="mb-12">
          <h2 className="text-h2 text-text-primary mb-4">Tips</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-bg-secondary rounded-xl border border-border-default p-5">
              <div className="flex items-center gap-2 mb-2">
                <MousePointer size={18} className="text-electric-indigo" />
                <h3 className="font-semibold text-text-primary">Selecting Blocks</h3>
              </div>
              <p className="text-small text-text-secondary">
                Click on a block to select it and view its configuration. Hold Shift to select multiple blocks.
              </p>
            </div>
            <div className="bg-bg-secondary rounded-xl border border-border-default p-5">
              <div className="flex items-center gap-2 mb-2">
                <Link2 size={18} className="text-fresh-teal" />
                <h3 className="font-semibold text-text-primary">Connecting Blocks</h3>
              </div>
              <p className="text-small text-text-secondary">
                Drag from the output handle (right side) of a block to the input handle (left side) of another to connect them.
              </p>
            </div>
            <div className="bg-bg-secondary rounded-xl border border-border-default p-5">
              <div className="flex items-center gap-2 mb-2">
                <Play size={18} className="text-golden-amber" />
                <h3 className="font-semibold text-text-primary">First Run</h3>
              </div>
              <p className="text-small text-text-secondary">
                The first time you run a pipeline, it takes about 30 seconds to load Python (Pyodide). Subsequent runs are much faster.
              </p>
            </div>
            <div className="bg-bg-secondary rounded-xl border border-border-default p-5">
              <div className="flex items-center gap-2 mb-2">
                <FileUp size={18} className="text-soft-violet" />
                <h3 className="font-semibold text-text-primary">Supported Files</h3>
              </div>
              <p className="text-small text-text-secondary">
                Upload CSV files to get started. You can also use the built-in sample datasets to explore.
              </p>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center py-8 border-t border-border-default">
          <p className="text-text-muted text-small">
            Data Flow Canvas &copy; 2025 Lavelle Hatcher Jr. All rights reserved.
          </p>
          <p className="text-text-muted text-small mt-1">
            Licensed under AGPL-3.0. For commercial licensing, contact the author.
          </p>
        </footer>
      </main>
    </div>
  );
}
