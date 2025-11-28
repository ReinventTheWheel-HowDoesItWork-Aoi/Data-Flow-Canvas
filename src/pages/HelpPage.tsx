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
                <p className="text-h2 font-bold text-text-primary">50+</p>
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
