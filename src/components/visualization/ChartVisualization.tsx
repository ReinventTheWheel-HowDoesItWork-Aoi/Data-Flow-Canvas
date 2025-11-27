/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { useUIStore } from '@/stores/uiStore';

interface ChartData {
  chartType: 'bar' | 'line' | 'scatter' | 'pie' | 'histogram' | 'box' | 'heatmap' | 'correlation_matrix' | 'violin' | 'pair_plot' | 'area' | 'stacked' | 'bubble' | 'qq_plot' | 'confusion_matrix' | 'roc_curve';
  data: Record<string, unknown>[];
  x: string;
  y: string;
  color?: string;
  title?: string;
  // Extended properties for new chart types
  columns?: string[];
  matrix?: number[][];
  column?: string;
  groupColumn?: string;
  colorColumn?: string;
  yColumns?: string[];
  stackType?: string;
  normalize?: boolean;
  size?: string;
  labels?: string[];
  auc?: number;
}

interface ChartVisualizationProps {
  chartData: ChartData;
  width?: number;
  height?: number;
}

export function ChartVisualization({ chartData, width = 400, height = 300 }: ChartVisualizationProps) {
  const { isDarkMode } = useUIStore();

  const { data, layout } = useMemo(() => {
    const { chartType, data: rawData, x, y, color, title } = chartData;

    if (!rawData || rawData.length === 0 || !x) {
      return { data: [], layout: {} };
    }

    const xValues = rawData.map((row) => row[x]);
    const yValues = y ? rawData.map((row) => row[y]) : [];

    // Color palette
    const colors = [
      '#6366f1', // electric-indigo
      '#8b5cf6', // soft-violet
      '#14b8a6', // fresh-teal
      '#f97316', // warm-coral
      '#eab308', // sunny-yellow
      '#ec4899', // pink
      '#06b6d4', // cyan
      '#84cc16', // lime
    ];

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let plotData: any[] = [];

    switch (chartType) {
      case 'bar':
        if (color && color !== x && color !== y) {
          // Grouped bar chart
          const groups = [...new Set(rawData.map((row) => row[color]))];
          plotData = groups.map((group, idx) => ({
            type: 'bar' as const,
            name: String(group),
            x: rawData.filter((row) => row[color] === group).map((row) => row[x]),
            y: rawData.filter((row) => row[color] === group).map((row) => row[y]),
            marker: { color: colors[idx % colors.length] },
          }));
        } else {
          plotData = [{
            type: 'bar' as const,
            x: xValues,
            y: yValues,
            marker: { color: colors[0] },
          }];
        }
        break;

      case 'line':
        if (color && color !== x && color !== y) {
          const groups = [...new Set(rawData.map((row) => row[color]))];
          plotData = groups.map((group, idx) => ({
            type: 'scatter' as const,
            mode: 'lines+markers' as const,
            name: String(group),
            x: rawData.filter((row) => row[color] === group).map((row) => row[x]),
            y: rawData.filter((row) => row[color] === group).map((row) => row[y]),
            line: { color: colors[idx % colors.length] },
          }));
        } else {
          plotData = [{
            type: 'scatter' as const,
            mode: 'lines+markers' as const,
            x: xValues,
            y: yValues,
            line: { color: colors[0] },
          }];
        }
        break;

      case 'scatter':
        if (color && color !== x && color !== y) {
          const groups = [...new Set(rawData.map((row) => row[color]))];
          plotData = groups.map((group, idx) => ({
            type: 'scatter' as const,
            mode: 'markers' as const,
            name: String(group),
            x: rawData.filter((row) => row[color] === group).map((row) => row[x]),
            y: rawData.filter((row) => row[color] === group).map((row) => row[y]),
            marker: { color: colors[idx % colors.length], size: 8 },
          }));
        } else {
          plotData = [{
            type: 'scatter' as const,
            mode: 'markers' as const,
            x: xValues,
            y: yValues,
            marker: { color: colors[0], size: 8 },
          }];
        }
        break;

      case 'pie':
        plotData = [{
          type: 'pie' as const,
          labels: xValues as string[],
          values: yValues as number[],
          marker: { colors },
        }];
        break;

      case 'histogram':
        plotData = [{
          type: 'histogram' as const,
          x: xValues,
          marker: { color: colors[0] },
        }];
        break;

      case 'box':
        if (color && color !== x) {
          const groups = [...new Set(rawData.map((row) => row[color]))];
          plotData = groups.map((group, idx) => ({
            type: 'box' as const,
            name: String(group),
            y: rawData.filter((row) => row[color] === group).map((row) => row[x]),
            marker: { color: colors[idx % colors.length] },
          }));
        } else {
          plotData = [{
            type: 'box' as const,
            y: xValues,
            marker: { color: colors[0] },
          }];
        }
        break;

      case 'heatmap':
        // For heatmap, we need a matrix format
        const xLabels = [...new Set(rawData.map((row) => row[x]))];
        const yLabels = y ? [...new Set(rawData.map((row) => row[y]))] : [];
        const zMatrix: number[][] = [];

        if (y && color) {
          yLabels.forEach((yLabel) => {
            const row: number[] = [];
            xLabels.forEach((xLabel) => {
              const match = rawData.find((r) => r[x] === xLabel && r[y] === yLabel);
              row.push(match ? Number(match[color]) || 0 : 0);
            });
            zMatrix.push(row);
          });

          plotData = [{
            type: 'heatmap' as const,
            x: xLabels as string[],
            y: yLabels as string[],
            z: zMatrix,
            colorscale: 'Viridis',
          }];
        }
        break;

      case 'correlation_matrix': {
        const corrColumns = chartData.columns || [];
        const corrMatrix = chartData.matrix || [];
        plotData = [{
          type: 'heatmap' as const,
          x: corrColumns,
          y: corrColumns,
          z: corrMatrix,
          colorscale: 'RdBu',
          zmin: -1,
          zmax: 1,
          text: corrMatrix.map(row => row.map(v => v.toFixed(2))),
          texttemplate: '%{text}',
          hovertemplate: '%{x} vs %{y}: %{z:.3f}<extra></extra>',
        }];
        break;
      }

      case 'violin': {
        const violinColumn = chartData.column || x;
        const violinGroup = chartData.groupColumn;
        if (violinGroup && violinGroup !== '') {
          const groups = [...new Set(rawData.map((row) => row[violinGroup]))];
          plotData = groups.map((group, idx) => ({
            type: 'violin' as const,
            name: String(group),
            y: rawData.filter((row) => row[violinGroup] === group).map((row) => row[violinColumn]),
            box: { visible: true },
            meanline: { visible: true },
            marker: { color: colors[idx % colors.length] },
          }));
        } else {
          plotData = [{
            type: 'violin' as const,
            y: rawData.map((row) => row[violinColumn]),
            box: { visible: true },
            meanline: { visible: true },
            marker: { color: colors[0] },
          }];
        }
        break;
      }

      case 'pair_plot': {
        const pairColumns = chartData.columns || [];
        const pairColor = chartData.colorColumn;
        const dimensions = pairColumns.map((col) => ({
          label: col,
          values: rawData.map((row) => row[col]),
        }));
        if (pairColor && pairColor !== '') {
          const uniqueColors = [...new Set(rawData.map((row) => row[pairColor]))];
          plotData = [{
            type: 'splom' as const,
            dimensions,
            marker: {
              color: rawData.map((row) => uniqueColors.indexOf(row[pairColor] as string)),
              colorscale: 'Viridis',
              size: 5,
            },
          }];
        } else {
          plotData = [{
            type: 'splom' as const,
            dimensions,
            marker: { color: colors[0], size: 5 },
          }];
        }
        break;
      }

      case 'area': {
        const areaColor = chartData.color;
        if (areaColor && areaColor !== x && areaColor !== y) {
          const groups = [...new Set(rawData.map((row) => row[areaColor]))];
          plotData = groups.map((group, idx) => ({
            type: 'scatter' as const,
            mode: 'lines' as const,
            fill: 'tozeroy' as const,
            name: String(group),
            x: rawData.filter((row) => row[areaColor] === group).map((row) => row[x]),
            y: rawData.filter((row) => row[areaColor] === group).map((row) => row[y]),
            line: { color: colors[idx % colors.length] },
          }));
        } else {
          plotData = [{
            type: 'scatter' as const,
            mode: 'lines' as const,
            fill: 'tozeroy' as const,
            x: xValues,
            y: yValues,
            line: { color: colors[0] },
          }];
        }
        break;
      }

      case 'stacked': {
        const stackYColumns = chartData.yColumns || [];
        const stackType = chartData.stackType || 'bar';
        plotData = stackYColumns.map((col, idx) => ({
          type: stackType === 'bar' ? 'bar' as const : 'scatter' as const,
          mode: stackType === 'area' ? 'lines' as const : undefined,
          fill: stackType === 'area' ? 'tonexty' as const : undefined,
          name: col,
          x: rawData.map((row) => row[x]),
          y: rawData.map((row) => row[col]),
          marker: { color: colors[idx % colors.length] },
          line: stackType === 'area' ? { color: colors[idx % colors.length] } : undefined,
          stackgroup: 'one',
        }));
        break;
      }

      case 'bubble': {
        const bubbleSize = chartData.size || '';
        const bubbleColor = chartData.color;
        const sizeValues = rawData.map((row) => Number(row[bubbleSize]) || 10);
        const maxSize = Math.max(...sizeValues);
        const normalizedSizes = sizeValues.map((s) => (s / maxSize) * 50 + 5);

        if (bubbleColor && bubbleColor !== x && bubbleColor !== y) {
          const groups = [...new Set(rawData.map((row) => row[bubbleColor]))];
          plotData = groups.map((group, idx) => {
            const groupData = rawData.filter((row) => row[bubbleColor] === group);
            const groupSizes = groupData.map((row) => {
              const s = Number(row[bubbleSize]) || 10;
              return (s / maxSize) * 50 + 5;
            });
            return {
              type: 'scatter' as const,
              mode: 'markers' as const,
              name: String(group),
              x: groupData.map((row) => row[x]),
              y: groupData.map((row) => row[y]),
              marker: {
                color: colors[idx % colors.length],
                size: groupSizes,
                sizemode: 'diameter' as const,
              },
            };
          });
        } else {
          plotData = [{
            type: 'scatter' as const,
            mode: 'markers' as const,
            x: xValues,
            y: yValues,
            marker: {
              color: colors[0],
              size: normalizedSizes,
              sizemode: 'diameter' as const,
            },
          }];
        }
        break;
      }

      case 'qq_plot': {
        plotData = [
          {
            type: 'scatter' as const,
            mode: 'markers' as const,
            name: 'Sample Data',
            x: rawData.map((row) => row['theoretical']),
            y: rawData.map((row) => row['sample']),
            marker: { color: colors[0], size: 6 },
          },
          {
            type: 'scatter' as const,
            mode: 'lines' as const,
            name: 'Reference Line',
            x: rawData.map((row) => row['theoretical']),
            y: rawData.map((row) => row['line']),
            line: { color: colors[3], dash: 'dash' },
          },
        ];
        break;
      }

      case 'confusion_matrix': {
        const cmLabels = chartData.labels || [];
        const cmMatrix = chartData.matrix || [];
        plotData = [{
          type: 'heatmap' as const,
          x: cmLabels,
          y: cmLabels,
          z: cmMatrix,
          colorscale: 'Blues',
          text: cmMatrix.map(row => row.map(v => typeof v === 'number' && v % 1 !== 0 ? v.toFixed(2) : String(v))),
          texttemplate: '%{text}',
          hovertemplate: 'Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
        }];
        break;
      }

      case 'roc_curve': {
        const aucValue = chartData.auc || 0;
        plotData = [
          {
            type: 'scatter' as const,
            mode: 'lines' as const,
            name: `ROC (AUC = ${aucValue.toFixed(3)})`,
            x: rawData.map((row) => row['fpr']),
            y: rawData.map((row) => row['tpr']),
            line: { color: colors[0], width: 2 },
          },
          {
            type: 'scatter' as const,
            mode: 'lines' as const,
            name: 'Random',
            x: [0, 1],
            y: [0, 1],
            line: { color: '#999999', dash: 'dash', width: 1 },
          },
        ];
        break;
      }

      default:
        plotData = [{
          type: 'bar' as const,
          x: xValues,
          y: yValues,
          marker: { color: colors[0] },
        }];
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const plotLayout: any = {
      title: title ? { text: title } : undefined,
      width,
      height,
      margin: { t: title ? 40 : 20, r: 20, b: 40, l: 50 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: {
        color: isDarkMode ? '#e5e7eb' : '#374151',
        family: 'Inter, system-ui, sans-serif',
        size: 11,
      },
      xaxis: {
        gridcolor: isDarkMode ? '#374151' : '#e5e7eb',
        zerolinecolor: isDarkMode ? '#4b5563' : '#d1d5db',
        title: x ? { text: x } : undefined,
      },
      yaxis: {
        gridcolor: isDarkMode ? '#374151' : '#e5e7eb',
        zerolinecolor: isDarkMode ? '#4b5563' : '#d1d5db',
        title: y ? { text: y } : undefined,
      },
      showlegend: color ? true : false,
      legend: {
        bgcolor: 'transparent',
        font: { size: 10 },
      },
    };

    return { data: plotData, layout: plotLayout };
  }, [chartData, width, height, isDarkMode]);

  if (!chartData.data || chartData.data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted text-small">
        No data available for chart
      </div>
    );
  }

  if (!chartData.x) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted text-small">
        Please configure X axis in block settings
      </div>
    );
  }

  return (
    <Plot
      data={data}
      layout={layout}
      config={{
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d'],
        responsive: true,
      }}
      style={{ width: '100%', height: '100%' }}
    />
  );
}
