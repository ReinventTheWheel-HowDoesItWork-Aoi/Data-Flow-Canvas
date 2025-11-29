/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React from 'react';
import type { BlockData, BlockType } from '@/types';

interface BlockBodyProps {
  type: BlockType;
  data: BlockData;
}

export function BlockBody({ type, data }: BlockBodyProps) {
  const config = data.config;

  const renderContent = () => {
    switch (type) {
      case 'load-data':
        return (
          <div className="text-small text-text-muted">
            {config.fileName ? (
              <span className="truncate">{config.fileName as string}</span>
            ) : (
              'Drop or select a file'
            )}
          </div>
        );

      case 'sample-data':
        return (
          <div className="text-small text-text-muted capitalize">
            {(config.dataset as string) || 'iris'} dataset
          </div>
        );

      case 'filter-rows':
        return (
          <div className="text-small text-text-muted">
            {config.column ? (
              <span>
                {config.column as string} {config.operator as string}{' '}
                {config.value as string}
              </span>
            ) : (
              'Configure filter'
            )}
          </div>
        );

      case 'select-columns':
        return (
          <div className="text-small text-text-muted">
            {(config.columns as string[])?.length > 0
              ? `${(config.columns as string[]).length} columns selected`
              : 'Select columns'}
          </div>
        );

      case 'sort':
        return (
          <div className="text-small text-text-muted">
            {(config.columns as string[])?.length > 0
              ? `By ${(config.columns as string[]).join(', ')}`
              : 'Configure sort'}
          </div>
        );

      case 'group-aggregate':
        return (
          <div className="text-small text-text-muted">
            {(config.groupBy as string[])?.length > 0
              ? `Group by ${(config.groupBy as string[]).join(', ')}`
              : 'Configure grouping'}
          </div>
        );

      case 'join':
        return (
          <div className="text-small text-text-muted">
            {config.how as string} join
          </div>
        );

      case 'derive-column':
        return (
          <div className="text-small text-text-muted truncate">
            {config.newColumn
              ? `Create: ${config.newColumn as string}`
              : 'Define expression'}
          </div>
        );

      case 'handle-missing':
        return (
          <div className="text-small text-text-muted capitalize">
            {(config.strategy as string)?.replace('_', ' ') || 'drop'}
          </div>
        );

      case 'statistics':
        return (
          <div className="text-small text-text-muted capitalize">
            {(config.type as string) || 'descriptive'}
          </div>
        );

      case 'regression':
        return (
          <div className="text-small text-text-muted capitalize">
            {(config.type as string) || 'linear'} regression
          </div>
        );

      case 'clustering':
        return (
          <div className="text-small text-text-muted">
            {(config.algorithm as string) || 'k-means'} (k=
            {(config.nClusters as number) || 3})
          </div>
        );

      case 'chart':
        return (
          <div className="text-small text-text-muted capitalize">
            {(config.chartType as string) || 'bar'} chart
          </div>
        );

      case 'table':
        return (
          <div className="text-small text-text-muted">
            Interactive table view
          </div>
        );

      case 'export':
        return (
          <div className="text-small text-text-muted uppercase">
            {(config.format as string) || 'csv'}
          </div>
        );

      case 'datetime-extract':
        return (
          <div className="text-small text-text-muted">
            {config.column
              ? `${config.column}: ${((config.extractions as string[]) || []).length} parts`
              : 'Select date column'}
          </div>
        );

      case 'string-operations':
        return (
          <div className="text-small text-text-muted capitalize">
            {config.column
              ? `${(config.operation as string)?.replace('_', ' ') || 'lowercase'}`
              : 'Configure operation'}
          </div>
        );

      case 'window-functions':
        return (
          <div className="text-small text-text-muted">
            {config.column
              ? `${(config.operation as string)?.replace('_', ' ') || 'rolling mean'}`
              : 'Configure window'}
          </div>
        );

      case 'bin-bucket':
        return (
          <div className="text-small text-text-muted">
            {config.column
              ? `${config.numBins || 5} bins (${(config.method as string)?.replace('_', ' ') || 'equal width'})`
              : 'Configure bins'}
          </div>
        );

      case 'rank':
        return (
          <div className="text-small text-text-muted">
            {config.column
              ? `${config.method || 'average'} rank`
              : 'Configure ranking'}
          </div>
        );

      case 'type-conversion':
        return (
          <div className="text-small text-text-muted">
            {config.column
              ? `→ ${(config.targetType as string) || 'string'}`
              : 'Configure conversion'}
          </div>
        );

      default:
        return null;
    }
  };

  return <div className="px-3 py-2 min-h-[32px]">{renderContent()}</div>;
}
