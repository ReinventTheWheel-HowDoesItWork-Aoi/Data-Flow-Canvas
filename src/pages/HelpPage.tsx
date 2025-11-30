/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { LanguageSelector } from '@/components/ui/LanguageSelector';
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
  Zap,
  BookOpen,
  Keyboard,
  Box,
  ChevronRight,
  Sparkles,
  Calendar,
  Type,
  Waypoints,
  BarChart,
  Medal,
  RefreshCw,
  ArrowLeftRight,
  MoveHorizontal,
  CalendarRange,
  FlipHorizontal,
  AlignJustify,
  Link as LinkIcon,
  MessageSquare,
  DollarSign,
  UserMinus,
  GitCompareArrows,
  Gauge,
  BarChart2,
  CheckCircle,
  GitCommitHorizontal,
  Replace,
  PlusSquare,
  Calculator,
  CalendarCheck,
  SplitSquareVertical,
  Expand,
  Trash2,
  Merge,
  Search,
  Code,
  Regex,
  GitCompare,
  ListOrdered,
  Trophy,
  Filter as FunnelIcon,
  GitPullRequestArrow,
  LayoutGrid as TreemapIcon,
  Sun,
  Gauge as GaugeIcon,
  Radar,
  BarChartHorizontal,
  CandlestickChart,
  Map,
  Cloud,
  BarChart4,
  SeparatorHorizontal,
  Network as DendrogramIcon,
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

  // Transform (64 blocks)
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
  { icon: Calendar, name: 'Date/Time Extract', description: 'Extract year, month, day, weekday, hour from date columns', category: 'Transform' },
  { icon: Type, name: 'String Operations', description: 'Clean text with case conversion, trim, find/replace, regex', category: 'Transform' },
  { icon: Waypoints, name: 'Window Functions', description: 'Rolling averages, cumulative sums, lag/lead values', category: 'Transform' },
  { icon: BarChart, name: 'Bin/Bucket', description: 'Group numbers into ranges (equal width, frequency, custom)', category: 'Transform' },
  { icon: Medal, name: 'Rank', description: 'Assign rank positions with optional grouping', category: 'Transform' },
  { icon: RefreshCw, name: 'Type Conversion', description: 'Convert column types (string, int, float, datetime)', category: 'Transform' },
  { icon: ArrowLeftRight, name: 'Fill Forward/Backward', description: 'Fill missing values with previous or next values', category: 'Transform' },
  { icon: MoveHorizontal, name: 'Lag/Lead', description: 'Create columns with shifted values (lag or lead)', category: 'Transform' },
  { icon: Hash, name: 'Row Number', description: 'Add a unique row number column', category: 'Transform' },
  { icon: CalendarRange, name: 'Date Difference', description: 'Calculate difference between two date columns', category: 'Transform' },
  { icon: FlipHorizontal, name: 'Transpose', description: 'Flip rows and columns', category: 'Transform' },
  { icon: AlignJustify, name: 'String Pad', description: 'Pad strings to a fixed length with specified character', category: 'Transform' },
  { icon: TrendingUp, name: 'Cumulative Operations', description: 'Calculate running totals, cumulative counts, and percentages', category: 'Transform' },
  { icon: Replace, name: 'Replace Values', description: 'Map and replace specific values with new values', category: 'Transform' },
  { icon: TrendingUp, name: 'Percent Change', description: 'Calculate percentage change between consecutive rows', category: 'Transform' },
  { icon: Hash, name: 'Round Numbers', description: 'Round, floor, or ceiling numeric values', category: 'Transform' },
  { icon: PieChart, name: 'Percent of Total', description: 'Calculate what percentage each row represents of the total', category: 'Transform' },
  { icon: PlusSquare, name: 'Absolute Value', description: 'Convert negative values to positive', category: 'Transform' },
  { icon: Calculator, name: 'Column Math', description: 'Perform arithmetic operations between two columns', category: 'Transform' },
  { icon: Scissors, name: 'Extract Substring', description: 'Extract portion of text from a string column', category: 'Transform' },
  { icon: CalendarCheck, name: 'Parse Date', description: 'Convert text strings to proper date format', category: 'Transform' },
  { icon: SplitSquareVertical, name: 'Split to Rows', description: 'Expand delimited values in a cell into separate rows', category: 'Transform' },
  { icon: Minimize2, name: 'Clip Values', description: 'Cap values at minimum and/or maximum thresholds', category: 'Transform' },
  { icon: Type, name: 'Standardize Text', description: 'Comprehensive text cleaning and normalization', category: 'Transform' },
  { icon: GitBranch, name: 'Case When', description: 'Create column based on multiple if-then conditions', category: 'Transform' },
  { icon: Expand, name: 'Explode Column', description: 'Expand list/array values in a column into separate rows', category: 'Transform' },
  { icon: PlusSquare, name: 'Add Constant Column', description: 'Add a new column with a fixed value for all rows', category: 'Transform' },
  { icon: Trash2, name: 'Drop Columns', description: 'Remove specific columns from the dataset', category: 'Transform' },
  { icon: Layers, name: 'Flatten JSON', description: 'Expand nested dict/JSON columns into separate columns', category: 'Transform' },
  { icon: Merge, name: 'Coalesce Columns', description: 'Get the first non-null value from multiple columns', category: 'Transform' },
  { icon: ArrowUpDown, name: 'Reorder Columns', description: 'Rearrange column order in the dataset', category: 'Transform' },
  { icon: Eraser, name: 'Trim & Clean Text', description: 'Remove whitespace and clean text formatting', category: 'Transform' },
  { icon: Search, name: 'Lookup (VLOOKUP)', description: 'Match and retrieve values from another dataset like Excel VLOOKUP', category: 'Transform' },
  { icon: Grid3x3, name: 'Cross Join', description: 'Create Cartesian product of two datasets (all combinations)', category: 'Transform' },
  { icon: Code, name: 'Filter by Expression', description: 'Filter rows using a Python expression for advanced conditions', category: 'Transform' },
  { icon: Hash, name: 'Number Format', description: 'Format numbers with thousands separators, decimals, currency, or percentage', category: 'Transform' },
  { icon: Regex, name: 'Extract Pattern', description: 'Extract text matching a regex pattern into a new column', category: 'Transform' },
  { icon: TrendingUp, name: 'Log Transform', description: 'Apply log, log10, log2, log1p, exp, or sqrt with zero handling', category: 'Transform' },
  { icon: TrendingUp, name: 'Interpolate Missing', description: 'Fill gaps using linear, polynomial, spline, nearest, or time-based interpolation', category: 'Transform' },
  { icon: Calendar, name: 'Date Truncate', description: 'Round dates to day/week/month/quarter/year/hour/minute start', category: 'Transform' },
  { icon: GitCompare, name: 'Period over Period', description: 'Calculate YoY, MoM, WoW, QoQ, DoD changes with absolute or percent values', category: 'Transform' },
  { icon: Hash, name: 'Hash Column', description: 'Hash column values using SHA-256, SHA-512, MD5, SHA-1, or BLAKE2', category: 'Transform' },
  { icon: CalendarRange, name: 'Expand Date Range', description: 'Fill missing dates in time series with configurable frequency and fill methods', category: 'Transform' },
  { icon: GitCompare, name: 'String Similarity', description: 'Calculate string similarity using Levenshtein, Jaro, Jaro-Winkler for fuzzy matching', category: 'Transform' },
  { icon: ListOrdered, name: 'Generate Sequence', description: 'Create helper tables with number ranges, date ranges, or repeated patterns', category: 'Transform' },
  { icon: Trophy, name: 'Top N per Group', description: 'Get top or bottom N rows per group based on a ranking column', category: 'Transform' },
  { icon: ListFilter, name: 'First/Last per Group', description: 'Get the first, last, or both rows per group based on sort order', category: 'Transform' },

  // Analysis (58 blocks)
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
  { icon: FlaskConical, name: 'ANOVA', description: 'Analysis of variance to compare means across groups', category: 'Analysis' },
  { icon: Grid3x3, name: 'Chi-Square Test', description: 'Test independence between categorical variables', category: 'Analysis' },
  { icon: TrendingUp, name: 'Correlation Analysis', description: 'Calculate correlation coefficients with statistical significance', category: 'Analysis' },
  { icon: Activity, name: 'Survival Analysis', description: 'Kaplan-Meier survival analysis for time-to-event data', category: 'Analysis' },
  { icon: LinkIcon, name: 'Association Rules', description: 'Find item associations using Apriori algorithm', category: 'Analysis' },
  { icon: MessageSquare, name: 'Sentiment Analysis', description: 'Analyze text sentiment (positive, negative, neutral)', category: 'Analysis' },
  { icon: TrendingUp, name: 'Moving Average', description: 'Calculate simple, exponential, or weighted moving averages', category: 'Analysis' },
  { icon: Scissors, name: 'Train/Test Split', description: 'Split data into training and testing sets for ML', category: 'Analysis' },
  { icon: Award, name: 'Model Evaluation', description: 'Calculate model performance metrics (accuracy, F1, RMSE)', category: 'Analysis' },
  { icon: Users, name: 'K-Nearest Neighbors', description: 'KNN classifier or regressor for prediction', category: 'Analysis' },
  { icon: Network, name: 'Naive Bayes', description: 'Naive Bayes classifier for text and categorical data', category: 'Analysis' },
  { icon: TrendingUp, name: 'Gradient Boosting', description: 'High-performance gradient boosting ML model', category: 'Analysis' },
  { icon: BarChart3, name: 'Pareto Analysis', description: 'Identify vital few vs trivial many using 80/20 rule', category: 'Analysis' },
  { icon: TrendingUp, name: 'Trend Analysis', description: 'Detect and quantify trends in time series data', category: 'Analysis' },
  { icon: Calendar, name: 'Forecasting', description: 'Predict future values using time series methods', category: 'Analysis' },
  { icon: Hash, name: 'Percentile Analysis', description: 'Calculate percentiles and quantile ranks', category: 'Analysis' },
  { icon: Activity, name: 'Distribution Fit', description: 'Fit data to statistical distributions', category: 'Analysis' },
  { icon: FileSearch, name: 'Text Preprocessing', description: 'Clean and prepare text data for analysis', category: 'Analysis' },
  { icon: Hash, name: 'TF-IDF Vectorization', description: 'Convert text to TF-IDF numerical features', category: 'Analysis' },
  { icon: Layers, name: 'Topic Modeling', description: 'Discover hidden topics in text using LDA', category: 'Analysis' },
  { icon: GitMerge, name: 'Similarity Analysis', description: 'Calculate similarity between rows or find similar items', category: 'Analysis' },
  { icon: Target, name: 'SVM', description: 'Support Vector Machine for classification and regression', category: 'Analysis' },
  { icon: TrendingUp, name: 'XGBoost', description: 'Extreme Gradient Boosting for high-performance ML', category: 'Analysis' },
  { icon: Lightbulb, name: 'Model Explainability', description: 'Explain model predictions with SHAP and feature importance', category: 'Analysis' },
  { icon: FileSearch, name: 'Regression Diagnostics', description: 'Analyze regression model assumptions and residuals', category: 'Analysis' },
  { icon: Scale, name: 'VIF Analysis', description: 'Detect multicollinearity with Variance Inflation Factor', category: 'Analysis' },
  { icon: Filter, name: 'Funnel Analysis', description: 'Analyze conversion funnels and drop-off rates', category: 'Analysis' },
  { icon: DollarSign, name: 'Customer Lifetime Value', description: 'Calculate CLV using RFM-based approach', category: 'Analysis' },
  { icon: UserMinus, name: 'Churn Prediction', description: 'Binary classification optimized for churn prediction', category: 'Analysis' },
  { icon: TrendingUp, name: 'Growth Metrics', description: 'Calculate business growth rate metrics', category: 'Analysis' },
  { icon: GitBranch, name: 'Attribution Modeling', description: 'Multi-touch marketing attribution analysis', category: 'Analysis' },
  { icon: Scale, name: 'Break-even Analysis', description: 'Calculate financial break-even point', category: 'Analysis' },
  { icon: ArrowLeftRight, name: 'Confidence Intervals', description: 'Calculate confidence intervals for means and proportions', category: 'Analysis' },
  { icon: RefreshCw, name: 'Bootstrap Analysis', description: 'Non-parametric bootstrap resampling for confidence intervals', category: 'Analysis' },
  { icon: GitCompareArrows, name: 'Post-hoc Tests', description: 'Multiple comparison tests after ANOVA', category: 'Analysis' },
  { icon: Gauge, name: 'Power Analysis', description: 'Sample size and power calculations', category: 'Analysis' },
  { icon: BarChart2, name: 'Bayesian Inference', description: 'Bayesian estimation for common scenarios', category: 'Analysis' },
  { icon: CheckCircle, name: 'Data Quality Score', description: 'Comprehensive data quality assessment', category: 'Analysis' },
  { icon: GitCommitHorizontal, name: 'Change Point Detection', description: 'Detect structural breaks in time series', category: 'Analysis' },

  // Visualization (24 blocks)
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
  { icon: FunnelIcon, name: 'Funnel Chart', description: 'Visualize sequential stages with progressive reduction', category: 'Visualization' },
  { icon: GitPullRequestArrow, name: 'Sankey Diagram', description: 'Show flow between nodes with proportional width bands', category: 'Visualization' },
  { icon: TreemapIcon, name: 'Treemap', description: 'Display hierarchical data as nested rectangles', category: 'Visualization' },
  { icon: Sun, name: 'Sunburst Chart', description: 'Radial hierarchical visualization with concentric rings', category: 'Visualization' },
  { icon: GaugeIcon, name: 'Gauge Chart', description: 'Speedometer-style display for single metrics', category: 'Visualization' },
  { icon: Radar, name: 'Radar Chart', description: 'Multi-axis radial chart for comparing variables', category: 'Visualization' },
  { icon: BarChartHorizontal, name: 'Waterfall Chart', description: 'Show cumulative effect of sequential values', category: 'Visualization' },
  { icon: CandlestickChart, name: 'Candlestick Chart', description: 'Financial OHLC visualization for price data', category: 'Visualization' },
  { icon: Map, name: 'Choropleth Map', description: 'Geographic heatmap coloring regions by values', category: 'Visualization' },
  { icon: Cloud, name: 'Word Cloud', description: 'Text visualization where word size represents frequency', category: 'Visualization' },
  { icon: BarChart4, name: 'Pareto Chart', description: 'Combined bar and cumulative line for 80/20 analysis', category: 'Visualization' },
  { icon: SeparatorHorizontal, name: 'Parallel Coordinates', description: 'High-dimensional data with connected lines', category: 'Visualization' },
  { icon: DendrogramIcon, name: 'Dendrogram', description: 'Tree diagram showing hierarchical clustering', category: 'Visualization' },

  // Output (1 block)
  { icon: Download, name: 'Export CSV', description: 'Export data to CSV format and download', category: 'Output' },
];

// Category data for navigation
const categories = ['Data Input', 'Transform', 'Analysis', 'Visualization', 'Output'] as const;
type Category = typeof categories[number];

const categoryColors: Record<Category, { bg: string; text: string; border: string; gradient: string }> = {
  'Data Input': {
    bg: 'bg-electric-indigo/10',
    text: 'text-electric-indigo',
    border: 'border-electric-indigo/20',
    gradient: 'from-electric-indigo to-soft-violet'
  },
  'Transform': {
    bg: 'bg-soft-violet/10',
    text: 'text-soft-violet',
    border: 'border-soft-violet/20',
    gradient: 'from-soft-violet to-electric-indigo'
  },
  'Analysis': {
    bg: 'bg-fresh-teal/10',
    text: 'text-fresh-teal',
    border: 'border-fresh-teal/20',
    gradient: 'from-fresh-teal to-electric-indigo'
  },
  'Visualization': {
    bg: 'bg-golden-amber/10',
    text: 'text-golden-amber',
    border: 'border-golden-amber/20',
    gradient: 'from-golden-amber to-warm-coral'
  },
  'Output': {
    bg: 'bg-warm-coral/10',
    text: 'text-warm-coral',
    border: 'border-warm-coral/20',
    gradient: 'from-warm-coral to-golden-amber'
  },
};

// Navigation sections
const navSections = [
  { id: 'quick-start', labelKey: 'help.nav.quickStart', icon: Zap },
  { id: 'shortcuts', labelKey: 'help.nav.shortcuts', icon: Keyboard },
  { id: 'blocks', labelKey: 'help.nav.blockReference', icon: Box },
  { id: 'tips', labelKey: 'help.nav.tips', icon: Lightbulb },
];

// Map category names to translation keys
const categoryTranslationKeys: Record<Category, string> = {
  'Data Input': 'categories.dataInput',
  'Transform': 'categories.transform',
  'Analysis': 'categories.analysis',
  'Visualization': 'categories.visualization',
  'Output': 'categories.output',
};

// Map block names to translation keys
const blockTranslationKeys: Record<string, { name: string; description: string }> = {
  'Load CSV': { name: 'blocks.loadData', description: 'blockDescriptions.loadData' },
  'Sample Data': { name: 'blocks.sampleData', description: 'blockDescriptions.sampleData' },
  'Create Dataset': { name: 'blocks.createDataset', description: 'blockDescriptions.createDataset' },
  'Filter Rows': { name: 'blocks.filterRows', description: 'blockDescriptions.filterRows' },
  'Select Columns': { name: 'blocks.selectColumns', description: 'blockDescriptions.selectColumns' },
  'Sort': { name: 'blocks.sort', description: 'blockDescriptions.sort' },
  'Group & Aggregate': { name: 'blocks.groupAggregate', description: 'blockDescriptions.groupAggregate' },
  'Join': { name: 'blocks.join', description: 'blockDescriptions.join' },
  'Derive Column': { name: 'blocks.deriveColumn', description: 'blockDescriptions.deriveColumn' },
  'Handle Missing': { name: 'blocks.handleMissing', description: 'blockDescriptions.handleMissing' },
  'Rename Columns': { name: 'blocks.renameColumns', description: 'blockDescriptions.renameColumns' },
  'Deduplicate': { name: 'blocks.deduplicate', description: 'blockDescriptions.deduplicate' },
  'Sample Rows': { name: 'blocks.sampleRows', description: 'blockDescriptions.sampleRows' },
  'Limit Rows': { name: 'blocks.limitRows', description: 'blockDescriptions.limitRows' },
  'Pivot': { name: 'blocks.pivot', description: 'blockDescriptions.pivot' },
  'Unpivot': { name: 'blocks.unpivot', description: 'blockDescriptions.unpivot' },
  'Union': { name: 'blocks.union', description: 'blockDescriptions.union' },
  'Split Column': { name: 'blocks.splitColumn', description: 'blockDescriptions.splitColumn' },
  'Merge Columns': { name: 'blocks.mergeColumns', description: 'blockDescriptions.mergeColumns' },
  'Conditional Column': { name: 'blocks.conditionalColumn', description: 'blockDescriptions.conditionalColumn' },
  'Date/Time Extract': { name: 'blocks.datetimeExtract', description: 'blockDescriptions.datetimeExtract' },
  'String Operations': { name: 'blocks.stringOperations', description: 'blockDescriptions.stringOperations' },
  'Window Functions': { name: 'blocks.windowFunctions', description: 'blockDescriptions.windowFunctions' },
  'Bin/Bucket': { name: 'blocks.binBucket', description: 'blockDescriptions.binBucket' },
  'Rank': { name: 'blocks.rank', description: 'blockDescriptions.rank' },
  'Type Conversion': { name: 'blocks.typeConversion', description: 'blockDescriptions.typeConversion' },
  'Fill Forward/Backward': { name: 'blocks.fillForwardBackward', description: 'blockDescriptions.fillForwardBackward' },
  'Lag/Lead': { name: 'blocks.lagLead', description: 'blockDescriptions.lagLead' },
  'Row Number': { name: 'blocks.rowNumber', description: 'blockDescriptions.rowNumber' },
  'Date Difference': { name: 'blocks.dateDifference', description: 'blockDescriptions.dateDifference' },
  'Transpose': { name: 'blocks.transpose', description: 'blockDescriptions.transpose' },
  'String Pad': { name: 'blocks.stringPad', description: 'blockDescriptions.stringPad' },
  'Cumulative Operations': { name: 'blocks.cumulativeOperations', description: 'blockDescriptions.cumulativeOperations' },
  'Replace Values': { name: 'blocks.replaceValues', description: 'blockDescriptions.replaceValues' },
  'Percent Change': { name: 'blocks.percentChange', description: 'blockDescriptions.percentChange' },
  'Round Numbers': { name: 'blocks.roundNumbers', description: 'blockDescriptions.roundNumbers' },
  'Percent of Total': { name: 'blocks.percentOfTotal', description: 'blockDescriptions.percentOfTotal' },
  'Absolute Value': { name: 'blocks.absoluteValue', description: 'blockDescriptions.absoluteValue' },
  'Column Math': { name: 'blocks.columnMath', description: 'blockDescriptions.columnMath' },
  'Extract Substring': { name: 'blocks.extractSubstring', description: 'blockDescriptions.extractSubstring' },
  'Parse Date': { name: 'blocks.parseDate', description: 'blockDescriptions.parseDate' },
  'Split to Rows': { name: 'blocks.splitToRows', description: 'blockDescriptions.splitToRows' },
  'Clip Values': { name: 'blocks.clipValues', description: 'blockDescriptions.clipValues' },
  'Standardize Text': { name: 'blocks.standardizeText', description: 'blockDescriptions.standardizeText' },
  'Case When': { name: 'blocks.caseWhen', description: 'blockDescriptions.caseWhen' },
  'Explode Column': { name: 'blocks.explodeColumn', description: 'blockDescriptions.explodeColumn' },
  'Add Constant Column': { name: 'blocks.addConstantColumn', description: 'blockDescriptions.addConstantColumn' },
  'Drop Columns': { name: 'blocks.dropColumns', description: 'blockDescriptions.dropColumns' },
  'Flatten JSON': { name: 'blocks.flattenJson', description: 'blockDescriptions.flattenJson' },
  'Coalesce Columns': { name: 'blocks.coalesceColumns', description: 'blockDescriptions.coalesceColumns' },
  'Reorder Columns': { name: 'blocks.reorderColumns', description: 'blockDescriptions.reorderColumns' },
  'Trim & Clean Text': { name: 'blocks.trimText', description: 'blockDescriptions.trimText' },
  'Lookup (VLOOKUP)': { name: 'blocks.lookupVlookup', description: 'blockDescriptions.lookupVlookup' },
  'Cross Join': { name: 'blocks.crossJoin', description: 'blockDescriptions.crossJoin' },
  'Filter by Expression': { name: 'blocks.filterExpression', description: 'blockDescriptions.filterExpression' },
  'Number Format': { name: 'blocks.numberFormat', description: 'blockDescriptions.numberFormat' },
  'Extract Pattern': { name: 'blocks.extractPattern', description: 'blockDescriptions.extractPattern' },
  'Log Transform': { name: 'blocks.logTransform', description: 'blockDescriptions.logTransform' },
  'Interpolate Missing': { name: 'blocks.interpolateMissing', description: 'blockDescriptions.interpolateMissing' },
  'Date Truncate': { name: 'blocks.dateTruncate', description: 'blockDescriptions.dateTruncate' },
  'Period over Period': { name: 'blocks.periodOverPeriod', description: 'blockDescriptions.periodOverPeriod' },
  'Hash Column': { name: 'blocks.hashColumn', description: 'blockDescriptions.hashColumn' },
  'Expand Date Range': { name: 'blocks.expandDateRange', description: 'blockDescriptions.expandDateRange' },
  'String Similarity': { name: 'blocks.stringSimilarity', description: 'blockDescriptions.stringSimilarity' },
  'Generate Sequence': { name: 'blocks.generateSequence', description: 'blockDescriptions.generateSequence' },
  'Top N per Group': { name: 'blocks.topNPerGroup', description: 'blockDescriptions.topNPerGroup' },
  'First/Last per Group': { name: 'blocks.firstLastPerGroup', description: 'blockDescriptions.firstLastPerGroup' },
  'Statistics': { name: 'blocks.statistics', description: 'blockDescriptions.statistics' },
  'Regression': { name: 'blocks.regression', description: 'blockDescriptions.regression' },
  'Clustering': { name: 'blocks.clustering', description: 'blockDescriptions.clustering' },
  'PCA': { name: 'blocks.pca', description: 'blockDescriptions.pca' },
  'Outlier Detection': { name: 'blocks.outlierDetection', description: 'blockDescriptions.outlierDetection' },
  'Classification': { name: 'blocks.classification', description: 'blockDescriptions.classification' },
  'Normality Test': { name: 'blocks.normalityTest', description: 'blockDescriptions.normalityTest' },
  'Hypothesis Testing': { name: 'blocks.hypothesisTesting', description: 'blockDescriptions.hypothesisTesting' },
  'Time Series Analysis': { name: 'blocks.timeSeries', description: 'blockDescriptions.timeSeries' },
  'Feature Importance': { name: 'blocks.featureImportance', description: 'blockDescriptions.featureImportance' },
  'Cross-Validation': { name: 'blocks.crossValidation', description: 'blockDescriptions.crossValidation' },
  'Data Profiling': { name: 'blocks.dataProfiling', description: 'blockDescriptions.dataProfiling' },
  'Value Counts': { name: 'blocks.valueCounts', description: 'blockDescriptions.valueCounts' },
  'Cross-Tabulation': { name: 'blocks.crossTabulation', description: 'blockDescriptions.crossTabulation' },
  'Scaling / Normalization': { name: 'blocks.scaling', description: 'blockDescriptions.scaling' },
  'Encoding': { name: 'blocks.encoding', description: 'blockDescriptions.encoding' },
  'A/B Test Analysis': { name: 'blocks.abTest', description: 'blockDescriptions.abTest' },
  'Cohort Analysis': { name: 'blocks.cohortAnalysis', description: 'blockDescriptions.cohortAnalysis' },
  'RFM Analysis': { name: 'blocks.rfmAnalysis', description: 'blockDescriptions.rfmAnalysis' },
  'ANOVA': { name: 'blocks.anova', description: 'blockDescriptions.anova' },
  'Chi-Square Test': { name: 'blocks.chiSquareTest', description: 'blockDescriptions.chiSquareTest' },
  'Correlation Analysis': { name: 'blocks.correlationAnalysis', description: 'blockDescriptions.correlationAnalysis' },
  'Survival Analysis': { name: 'blocks.survivalAnalysis', description: 'blockDescriptions.survivalAnalysis' },
  'Association Rules': { name: 'blocks.associationRules', description: 'blockDescriptions.associationRules' },
  'Sentiment Analysis': { name: 'blocks.sentimentAnalysis', description: 'blockDescriptions.sentimentAnalysis' },
  'Moving Average': { name: 'blocks.movingAverage', description: 'blockDescriptions.movingAverage' },
  'Train/Test Split': { name: 'blocks.trainTestSplit', description: 'blockDescriptions.trainTestSplit' },
  'Model Evaluation': { name: 'blocks.modelEvaluation', description: 'blockDescriptions.modelEvaluation' },
  'K-Nearest Neighbors': { name: 'blocks.knn', description: 'blockDescriptions.knn' },
  'Naive Bayes': { name: 'blocks.naiveBayes', description: 'blockDescriptions.naiveBayes' },
  'Gradient Boosting': { name: 'blocks.gradientBoosting', description: 'blockDescriptions.gradientBoosting' },
  'Pareto Analysis': { name: 'blocks.paretoAnalysis', description: 'blockDescriptions.paretoAnalysis' },
  'Trend Analysis': { name: 'blocks.trendAnalysis', description: 'blockDescriptions.trendAnalysis' },
  'Forecasting': { name: 'blocks.forecasting', description: 'blockDescriptions.forecasting' },
  'Percentile Analysis': { name: 'blocks.percentileAnalysis', description: 'blockDescriptions.percentileAnalysis' },
  'Distribution Fit': { name: 'blocks.distributionFit', description: 'blockDescriptions.distributionFit' },
  'Text Preprocessing': { name: 'blocks.textPreprocessing', description: 'blockDescriptions.textPreprocessing' },
  'TF-IDF Vectorization': { name: 'blocks.tfidfVectorization', description: 'blockDescriptions.tfidfVectorization' },
  'Topic Modeling': { name: 'blocks.topicModeling', description: 'blockDescriptions.topicModeling' },
  'Similarity Analysis': { name: 'blocks.similarityAnalysis', description: 'blockDescriptions.similarityAnalysis' },
  'SVM': { name: 'blocks.svm', description: 'blockDescriptions.svm' },
  'XGBoost': { name: 'blocks.xgboost', description: 'blockDescriptions.xgboost' },
  'Model Explainability': { name: 'blocks.modelExplainability', description: 'blockDescriptions.modelExplainability' },
  'Regression Diagnostics': { name: 'blocks.regressionDiagnostics', description: 'blockDescriptions.regressionDiagnostics' },
  'VIF Analysis': { name: 'blocks.vifAnalysis', description: 'blockDescriptions.vifAnalysis' },
  'Funnel Analysis': { name: 'blocks.funnelAnalysis', description: 'blockDescriptions.funnelAnalysis' },
  'Customer Lifetime Value': { name: 'blocks.customerLtv', description: 'blockDescriptions.customerLtv' },
  'Churn Prediction': { name: 'blocks.churnAnalysis', description: 'blockDescriptions.churnAnalysis' },
  'Growth Metrics': { name: 'blocks.growthMetrics', description: 'blockDescriptions.growthMetrics' },
  'Attribution Modeling': { name: 'blocks.attributionModeling', description: 'blockDescriptions.attributionModeling' },
  'Break-even Analysis': { name: 'blocks.breakevenAnalysis', description: 'blockDescriptions.breakevenAnalysis' },
  'Confidence Intervals': { name: 'blocks.confidenceIntervals', description: 'blockDescriptions.confidenceIntervals' },
  'Bootstrap Analysis': { name: 'blocks.bootstrapAnalysis', description: 'blockDescriptions.bootstrapAnalysis' },
  'Post-hoc Tests': { name: 'blocks.posthocTests', description: 'blockDescriptions.posthocTests' },
  'Power Analysis': { name: 'blocks.powerAnalysis', description: 'blockDescriptions.powerAnalysis' },
  'Bayesian Inference': { name: 'blocks.bayesianInference', description: 'blockDescriptions.bayesianInference' },
  'Data Quality Score': { name: 'blocks.dataQualityScore', description: 'blockDescriptions.dataQualityScore' },
  'Change Point Detection': { name: 'blocks.changepointDetection', description: 'blockDescriptions.changepointDetection' },
  'Chart': { name: 'blocks.chart', description: 'blockDescriptions.chart' },
  'Table': { name: 'blocks.table', description: 'blockDescriptions.table' },
  'Correlation Matrix': { name: 'blocks.correlationMatrix', description: 'blockDescriptions.correlationMatrix' },
  'Violin Plot': { name: 'blocks.violinPlot', description: 'blockDescriptions.violinPlot' },
  'Pair Plot': { name: 'blocks.pairPlot', description: 'blockDescriptions.pairPlot' },
  'Area Chart': { name: 'blocks.areaChart', description: 'blockDescriptions.areaChart' },
  'Stacked Bar/Area': { name: 'blocks.stackedChart', description: 'blockDescriptions.stackedChart' },
  'Bubble Chart': { name: 'blocks.bubbleChart', description: 'blockDescriptions.bubbleChart' },
  'Q-Q Plot': { name: 'blocks.qqPlot', description: 'blockDescriptions.qqPlot' },
  'Confusion Matrix': { name: 'blocks.confusionMatrix', description: 'blockDescriptions.confusionMatrix' },
  'ROC Curve': { name: 'blocks.rocCurve', description: 'blockDescriptions.rocCurve' },
  'Funnel Chart': { name: 'blocks.funnelChart', description: 'blockDescriptions.funnelChart' },
  'Sankey Diagram': { name: 'blocks.sankeyDiagram', description: 'blockDescriptions.sankeyDiagram' },
  'Treemap': { name: 'blocks.treemap', description: 'blockDescriptions.treemap' },
  'Sunburst Chart': { name: 'blocks.sunburstChart', description: 'blockDescriptions.sunburstChart' },
  'Gauge Chart': { name: 'blocks.gaugeChart', description: 'blockDescriptions.gaugeChart' },
  'Radar Chart': { name: 'blocks.radarChart', description: 'blockDescriptions.radarChart' },
  'Waterfall Chart': { name: 'blocks.waterfallChart', description: 'blockDescriptions.waterfallChart' },
  'Candlestick Chart': { name: 'blocks.candlestickChart', description: 'blockDescriptions.candlestickChart' },
  'Choropleth Map': { name: 'blocks.choroplethMap', description: 'blockDescriptions.choroplethMap' },
  'Word Cloud': { name: 'blocks.wordCloud', description: 'blockDescriptions.wordCloud' },
  'Pareto Chart': { name: 'blocks.paretoChart', description: 'blockDescriptions.paretoChart' },
  'Parallel Coordinates': { name: 'blocks.parallelCoordinates', description: 'blockDescriptions.parallelCoordinates' },
  'Dendrogram': { name: 'blocks.dendrogram', description: 'blockDescriptions.dendrogram' },
  'Export CSV': { name: 'blocks.export', description: 'blockDescriptions.export' },
};

export default function HelpPage() {
  const { t } = useTranslation();
  const [activeCategory, setActiveCategory] = useState<Category>('Data Input');
  const [activeSection, setActiveSection] = useState('quick-start');

  // Track active section on scroll
  useEffect(() => {
    const handleScroll = () => {
      const sections = navSections.map(s => document.getElementById(s.id));
      const scrollPosition = window.scrollY + 200;

      for (let i = sections.length - 1; i >= 0; i--) {
        const section = sections[i];
        if (section && section.offsetTop <= scrollPosition) {
          setActiveSection(navSections[i].id);
          break;
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      const offset = 100;
      const elementPosition = element.getBoundingClientRect().top + window.pageYOffset;
      window.scrollTo({ top: elementPosition - offset, behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-bg-primary">
      {/* Header with gradient accent */}
      <header className="sticky top-0 z-50 bg-bg-secondary/80 backdrop-blur-xl border-b border-border-default">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center gap-4">
          <Link
            to="/editor"
            className="group flex items-center gap-2 text-text-muted hover:text-text-primary transition-all duration-200"
          >
            <div className="p-1.5 rounded-lg bg-bg-tertiary group-hover:bg-electric-indigo/10 transition-colors">
              <ChevronLeft size={18} className="group-hover:text-electric-indigo transition-colors" />
            </div>
            <span className="text-small font-medium">{t('help.backToEditor')}</span>
          </Link>
          <div className="flex-1" />

          {/* Section Navigation Pills */}
          <nav className="hidden md:flex items-center gap-1 bg-bg-tertiary rounded-full p-1">
            {navSections.map((section) => (
              <button
                key={section.id}
                onClick={() => scrollToSection(section.id)}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-full text-small font-medium transition-all duration-200',
                  activeSection === section.id
                    ? 'bg-electric-indigo text-white shadow-md'
                    : 'text-text-muted hover:text-text-primary hover:bg-bg-secondary'
                )}
              >
                <section.icon size={14} />
                <span>{t(section.labelKey)}</span>
              </button>
            ))}
          </nav>

          <LanguageSelector variant="full" />
          <div className="flex-1 md:flex-none" />
          <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
            <img src="/logo.svg" alt="Data Flow Canvas" className="w-8 h-8 rounded-lg shadow-glow" />
            <span className="hidden sm:block font-semibold text-text-primary">Data Flow Canvas</span>
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative overflow-hidden border-b border-border-default">
        {/* Background gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-electric-indigo/5 via-transparent to-soft-violet/5" />
        <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-br from-electric-indigo/10 to-transparent rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />

        <div className="relative max-w-6xl mx-auto px-6 py-16 md:py-20">
          <div className="flex items-center gap-2 mb-4">
            <div className="flex items-center gap-1.5 px-3 py-1 bg-electric-indigo/10 rounded-full">
              <BookOpen size={14} className="text-electric-indigo" />
              <span className="text-small font-medium text-electric-indigo">{t('help.hero.documentation')}</span>
            </div>
          </div>
          <h1 className="text-display md:text-[56px] font-bold text-text-primary mb-4 tracking-tight">
            {t('help.hero.title')}
          </h1>
          <p className="text-h3 text-text-secondary font-normal max-w-2xl leading-relaxed">
            {t('help.hero.description')}
          </p>

          {/* Quick stats */}
          <div className="flex flex-wrap gap-6 mt-10">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-electric-indigo/10 flex items-center justify-center">
                <Box size={22} className="text-electric-indigo" />
              </div>
              <div>
                <p className="text-h2 font-bold text-text-primary">150</p>
                <p className="text-small text-text-muted">{t('help.hero.availableBlocks')}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-fresh-teal/10 flex items-center justify-center">
                <Zap size={22} className="text-fresh-teal" />
              </div>
              <div>
                <p className="text-h2 font-bold text-text-primary">5</p>
                <p className="text-small text-text-muted">{t('help.hero.stepsToStart')}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-golden-amber/10 flex items-center justify-center">
                <Sparkles size={22} className="text-golden-amber" />
              </div>
              <div>
                <p className="text-h2 font-bold text-text-primary">{t('help.hero.noCode')}</p>
                <p className="text-small text-text-muted">{t('help.hero.required')}</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Content */}
      <main className="max-w-6xl mx-auto px-6 py-12">
        {/* Quick Start */}
        <section id="quick-start" className="mb-16 scroll-mt-28">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-golden-amber to-warm-coral shadow-md">
              <Lightbulb size={22} className="text-white" />
            </div>
            <div>
              <h2 className="text-h2 text-text-primary">{t('help.quickStart.title')}</h2>
              <p className="text-small text-text-muted">{t('help.quickStart.subtitle')}</p>
            </div>
          </div>

          <div className="relative">
            {/* Connecting line */}
            <div className="absolute left-[23px] top-8 bottom-8 w-0.5 bg-gradient-to-b from-electric-indigo via-soft-violet to-fresh-teal hidden md:block" />

            <div className="space-y-4">
              {[
                {
                  step: 1,
                  titleKey: 'help.quickStart.step1.title',
                  descriptionKey: 'help.quickStart.step1.description',
                  icon: Database,
                  color: 'electric-indigo',
                },
                {
                  step: 2,
                  titleKey: 'help.quickStart.step2.title',
                  descriptionKey: 'help.quickStart.step2.description',
                  icon: PenLine,
                  color: 'soft-violet',
                },
                {
                  step: 3,
                  titleKey: 'help.quickStart.step3.title',
                  descriptionKey: 'help.quickStart.step3.description',
                  icon: GitMerge,
                  color: 'electric-indigo',
                },
                {
                  step: 4,
                  titleKey: 'help.quickStart.step4.title',
                  descriptionKey: 'help.quickStart.step4.description',
                  icon: Play,
                  color: 'fresh-teal',
                  kbd: 'Ctrl+Enter',
                },
                {
                  step: 5,
                  titleKey: 'help.quickStart.step5.title',
                  descriptionKey: 'help.quickStart.step5.description',
                  icon: BarChart3,
                  color: 'golden-amber',
                },
              ].map((item, index) => (
                <div
                  key={item.step}
                  className={cn(
                    'group relative bg-bg-secondary rounded-2xl border border-border-default p-6 transition-all duration-300',
                    'hover:border-electric-indigo/30 hover:shadow-lg hover:shadow-electric-indigo/5'
                  )}
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="flex gap-5">
                    <div className="relative z-10 flex-shrink-0">
                      <div className={cn(
                        'w-12 h-12 rounded-xl flex items-center justify-center font-bold text-white shadow-lg transition-transform duration-300 group-hover:scale-110',
                        `bg-${item.color}`
                      )}
                      style={{
                        background: item.color === 'electric-indigo' ? '#6366f1' :
                                  item.color === 'soft-violet' ? '#8b5cf6' :
                                  item.color === 'fresh-teal' ? '#14b8a6' :
                                  item.color === 'golden-amber' ? '#f59e0b' : '#6366f1'
                      }}
                      >
                        {item.step}
                      </div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="text-h3 text-text-primary">{t(item.titleKey)}</h3>
                        <item.icon size={18} className="text-text-muted" />
                      </div>
                      <p className="text-text-secondary leading-relaxed">
                        {t(item.descriptionKey)}
                        {item.kbd && (
                          <kbd className="ml-2 px-2 py-0.5 bg-bg-tertiary rounded-md text-small font-mono text-text-muted border border-border-default">
                            {item.kbd}
                          </kbd>
                        )}
                      </p>
                    </div>
                    <ChevronRight size={20} className="text-text-muted opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 mt-1" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Keyboard Shortcuts */}
        <section id="shortcuts" className="mb-16 scroll-mt-28">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-soft-violet to-electric-indigo shadow-md">
              <Keyboard size={22} className="text-white" />
            </div>
            <div>
              <h2 className="text-h2 text-text-primary">{t('help.shortcuts.title')}</h2>
              <p className="text-small text-text-muted">{t('help.shortcuts.subtitle')}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {[
              { keys: ['Ctrl', 'Enter'], actionKey: 'help.shortcuts.runPipeline', icon: Play, color: 'fresh-teal' },
              { keys: ['Ctrl', 'Z'], actionKey: 'help.shortcuts.undo', icon: RotateCcw, color: 'electric-indigo' },
              { keys: ['Ctrl', 'Shift', 'Z'], actionKey: 'help.shortcuts.redo', icon: RotateCw, color: 'electric-indigo' },
              { keys: ['Delete'], actionKey: 'help.shortcuts.deleteBlocks', icon: Eraser, color: 'warm-coral' },
              { keys: ['Escape'], actionKey: 'help.shortcuts.clearSelection', icon: MousePointer, color: 'soft-violet' },
              { keys: ['Ctrl', 'J'], actionKey: 'help.shortcuts.toggleLogs', icon: Layers, color: 'golden-amber' },
            ].map((shortcut, index) => (
              <div
                key={index}
                className="group bg-bg-secondary rounded-xl border border-border-default p-4 hover:border-electric-indigo/30 transition-all duration-200"
              >
                <div className="flex items-start gap-3">
                  <div className={cn(
                    'p-2 rounded-lg transition-colors',
                    `bg-${shortcut.color}/10`
                  )}
                  style={{
                    backgroundColor: shortcut.color === 'fresh-teal' ? 'rgba(20,184,166,0.1)' :
                                    shortcut.color === 'electric-indigo' ? 'rgba(99,102,241,0.1)' :
                                    shortcut.color === 'warm-coral' ? 'rgba(244,63,94,0.1)' :
                                    shortcut.color === 'soft-violet' ? 'rgba(139,92,246,0.1)' :
                                    shortcut.color === 'golden-amber' ? 'rgba(245,158,11,0.1)' : 'rgba(99,102,241,0.1)'
                  }}
                  >
                    <shortcut.icon size={16} style={{
                      color: shortcut.color === 'fresh-teal' ? '#14b8a6' :
                            shortcut.color === 'electric-indigo' ? '#6366f1' :
                            shortcut.color === 'warm-coral' ? '#f43f5e' :
                            shortcut.color === 'soft-violet' ? '#8b5cf6' :
                            shortcut.color === 'golden-amber' ? '#f59e0b' : '#6366f1'
                    }} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex flex-wrap gap-1 mb-1.5">
                      {shortcut.keys.map((key, i) => (
                        <React.Fragment key={i}>
                          <kbd className="px-2 py-1 bg-bg-tertiary rounded-md text-small font-mono text-text-primary border border-border-default shadow-sm">
                            {key}
                          </kbd>
                          {i < shortcut.keys.length - 1 && (
                            <span className="text-text-muted text-small self-center">+</span>
                          )}
                        </React.Fragment>
                      ))}
                    </div>
                    <p className="text-small text-text-secondary">{t(shortcut.actionKey)}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Block Reference */}
        <section id="blocks" className="mb-16 scroll-mt-28">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-fresh-teal to-electric-indigo shadow-md">
              <Box size={22} className="text-white" />
            </div>
            <div>
              <h2 className="text-h2 text-text-primary">{t('help.blockReference.title')}</h2>
              <p className="text-small text-text-muted">{t('help.blockReference.subtitle')}</p>
            </div>
          </div>

          {/* Category Tabs */}
          <div className="flex flex-wrap gap-2 mb-6 p-1 bg-bg-secondary rounded-xl border border-border-default">
            {categories.map((category) => {
              const colors = categoryColors[category];
              const count = blocks.filter(b => b.category === category).length;
              return (
                <button
                  key={category}
                  onClick={() => setActiveCategory(category)}
                  className={cn(
                    'flex items-center gap-2 px-4 py-2.5 rounded-lg text-small font-medium transition-all duration-200',
                    activeCategory === category
                      ? `${colors.bg} ${colors.text} shadow-sm`
                      : 'text-text-muted hover:text-text-primary hover:bg-bg-tertiary'
                  )}
                >
                  <span>{t(categoryTranslationKeys[category])}</span>
                  <span className={cn(
                    'px-1.5 py-0.5 rounded-md text-[10px] font-semibold',
                    activeCategory === category
                      ? 'bg-white/20'
                      : 'bg-bg-tertiary'
                  )}>
                    {count}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Block Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {blocks
              .filter((b) => b.category === activeCategory)
              .map((block, index) => {
                const colors = categoryColors[activeCategory];
                return (
                  <div
                    key={block.name}
                    className={cn(
                      'group bg-bg-secondary rounded-xl border border-border-default p-4 transition-all duration-200',
                      'hover:shadow-lg hover:shadow-black/5 hover:-translate-y-0.5',
                      `hover:${colors.border}`
                    )}
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <div className="flex items-start gap-3">
                      <div className={cn(
                        'p-2.5 rounded-xl transition-all duration-200 group-hover:scale-110',
                        colors.bg, colors.text
                      )}>
                        <block.icon size={20} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h4 className="font-semibold text-text-primary mb-0.5">
                          {blockTranslationKeys[block.name] ? t(blockTranslationKeys[block.name].name) : block.name}
                        </h4>
                        <p className="text-small text-text-secondary leading-relaxed">
                          {blockTranslationKeys[block.name] ? t(blockTranslationKeys[block.name].description) : block.description}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>
        </section>

        {/* Tips */}
        <section id="tips" className="mb-16 scroll-mt-28">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-electric-indigo to-soft-violet shadow-md">
              <Lightbulb size={22} className="text-white" />
            </div>
            <div>
              <h2 className="text-h2 text-text-primary">{t('help.tips.title')}</h2>
              <p className="text-small text-text-muted">{t('help.tips.subtitle')}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              {
                icon: MousePointer,
                titleKey: 'help.tips.selectingBlocks.title',
                descriptionKey: 'help.tips.selectingBlocks.description',
                color: 'electric-indigo',
                gradient: 'from-electric-indigo to-soft-violet',
              },
              {
                icon: Link2,
                titleKey: 'help.tips.connectingBlocks.title',
                descriptionKey: 'help.tips.connectingBlocks.description',
                color: 'fresh-teal',
                gradient: 'from-fresh-teal to-electric-indigo',
              },
              {
                icon: Play,
                titleKey: 'help.tips.firstRun.title',
                descriptionKey: 'help.tips.firstRun.description',
                color: 'golden-amber',
                gradient: 'from-golden-amber to-warm-coral',
              },
              {
                icon: FileUp,
                titleKey: 'help.tips.supportedFiles.title',
                descriptionKey: 'help.tips.supportedFiles.description',
                color: 'soft-violet',
                gradient: 'from-soft-violet to-electric-indigo',
              },
            ].map((tip, index) => (
              <div
                key={tip.titleKey}
                className="group relative bg-bg-secondary rounded-2xl border border-border-default p-6 overflow-hidden transition-all duration-300 hover:shadow-xl hover:shadow-black/5 hover:-translate-y-1"
              >
                {/* Decorative gradient */}
                <div className={cn(
                  'absolute top-0 right-0 w-32 h-32 opacity-0 group-hover:opacity-100 transition-opacity duration-500',
                  `bg-gradient-to-br ${tip.gradient}`
                )}
                style={{
                  filter: 'blur(60px)',
                  transform: 'translate(30%, -30%)'
                }} />

                <div className="relative">
                  <div className="flex items-center gap-3 mb-3">
                    <div className={cn(
                      'p-2.5 rounded-xl transition-transform duration-300 group-hover:scale-110'
                    )}
                    style={{
                      backgroundColor: tip.color === 'electric-indigo' ? 'rgba(99,102,241,0.1)' :
                                      tip.color === 'fresh-teal' ? 'rgba(20,184,166,0.1)' :
                                      tip.color === 'golden-amber' ? 'rgba(245,158,11,0.1)' :
                                      tip.color === 'soft-violet' ? 'rgba(139,92,246,0.1)' : 'rgba(99,102,241,0.1)'
                    }}
                    >
                      <tip.icon size={20} style={{
                        color: tip.color === 'electric-indigo' ? '#6366f1' :
                              tip.color === 'fresh-teal' ? '#14b8a6' :
                              tip.color === 'golden-amber' ? '#f59e0b' :
                              tip.color === 'soft-violet' ? '#8b5cf6' : '#6366f1'
                      }} />
                    </div>
                    <h3 className="font-semibold text-text-primary text-h3">{t(tip.titleKey)}</h3>
                  </div>
                  <p className="text-text-secondary leading-relaxed">
                    {t(tip.descriptionKey)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center py-12 border-t border-border-default">
          <div className="flex items-center justify-center gap-2 mb-4">
            <img src="/logo.svg" alt="Data Flow Canvas" className="w-10 h-10 rounded-xl shadow-lg" />
          </div>
          <p className="text-text-secondary font-medium mb-1">
            Data Flow Canvas
          </p>
          <p className="text-text-muted text-small">
            {t('help.footer.copyright')}
          </p>
          <p className="text-text-muted text-small mt-1">
            {t('help.footer.license')}
          </p>
          <div className="flex items-center justify-center gap-4 mt-4 text-small">
            <Link to="/terms" className="text-text-muted hover:text-text-primary transition-colors">
              {t('help.footer.terms')}
            </Link>
            <span className="text-text-muted">|</span>
            <Link to="/privacy" className="text-text-muted hover:text-text-primary transition-colors">
              {t('help.footer.privacy')}
            </Link>
          </div>
        </footer>
      </main>
    </div>
  );
}
