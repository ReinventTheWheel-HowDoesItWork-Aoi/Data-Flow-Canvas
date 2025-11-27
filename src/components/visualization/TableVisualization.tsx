/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { useState, useMemo } from 'react';
import { ChevronUp, ChevronDown, ChevronsUpDown } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface TableData {
  data: Record<string, unknown>[];
  columns: string[];
  dtypes: Record<string, string>;
  rowCount: number;
}

interface TableVisualizationProps {
  tableData: TableData;
  pageSize?: number;
  maxHeight?: number;
}

export function TableVisualization({
  tableData,
  pageSize = 50,
  maxHeight = 400
}: TableVisualizationProps) {
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [currentPage, setCurrentPage] = useState(0);
  const [filter, setFilter] = useState('');

  const { data, columns, dtypes, rowCount } = tableData;

  const filteredData = useMemo(() => {
    if (!filter) return data;
    const lowerFilter = filter.toLowerCase();
    return data.filter((row) =>
      columns.some((col) => {
        const value = row[col];
        return value != null && String(value).toLowerCase().includes(lowerFilter);
      })
    );
  }, [data, columns, filter]);

  const sortedData = useMemo(() => {
    if (!sortColumn) return filteredData;
    return [...filteredData].sort((a, b) => {
      const aVal = a[sortColumn];
      const bVal = b[sortColumn];

      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return sortDirection === 'asc' ? 1 : -1;
      if (bVal == null) return sortDirection === 'asc' ? -1 : 1;

      const dtype = dtypes[sortColumn];
      if (dtype?.includes('int') || dtype?.includes('float')) {
        const diff = Number(aVal) - Number(bVal);
        return sortDirection === 'asc' ? diff : -diff;
      }

      const strA = String(aVal);
      const strB = String(bVal);
      const comparison = strA.localeCompare(strB);
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [filteredData, sortColumn, sortDirection, dtypes]);

  const paginatedData = useMemo(() => {
    const start = currentPage * pageSize;
    return sortedData.slice(start, start + pageSize);
  }, [sortedData, currentPage, pageSize]);

  const totalPages = Math.ceil(sortedData.length / pageSize);

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  const formatValue = (value: unknown, dtype: string): string => {
    if (value == null) return '—';
    if (dtype?.includes('float')) {
      const num = Number(value);
      if (Number.isNaN(num)) return String(value);
      return num.toFixed(4);
    }
    if (dtype?.includes('bool')) {
      return value ? 'true' : 'false';
    }
    return String(value);
  };

  const getColumnWidth = (column: string): string => {
    const dtype = dtypes[column];
    if (dtype?.includes('int') || dtype?.includes('float')) return 'w-24';
    if (dtype?.includes('bool')) return 'w-16';
    return 'min-w-32 max-w-64';
  };

  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-text-muted text-small">
        No data available
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header with filter and row count */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border-default">
        <span className="text-small text-text-muted">
          {sortedData.length} of {rowCount} rows
        </span>
        <input
          type="text"
          placeholder="Filter..."
          value={filter}
          onChange={(e) => {
            setFilter(e.target.value);
            setCurrentPage(0);
          }}
          className="px-2 py-1 text-small bg-bg-tertiary border border-border-default rounded focus:outline-none focus:ring-1 focus:ring-electric-indigo"
        />
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto" style={{ maxHeight }}>
        <table className="w-full text-small">
          <thead className="sticky top-0 bg-bg-secondary z-10">
            <tr>
              <th className="px-2 py-2 text-left text-text-muted font-medium border-b border-border-default w-12">
                #
              </th>
              {columns.map((column) => (
                <th
                  key={column}
                  className={cn(
                    'px-2 py-2 text-left font-medium border-b border-border-default cursor-pointer hover:bg-bg-tertiary transition-colors',
                    getColumnWidth(column)
                  )}
                  onClick={() => handleSort(column)}
                >
                  <div className="flex items-center gap-1">
                    <span className="truncate text-text-primary">{column}</span>
                    <span className="text-text-muted text-xs">
                      ({dtypes[column]?.split('64')[0] || 'str'})
                    </span>
                    <span className="ml-auto">
                      {sortColumn === column ? (
                        sortDirection === 'asc' ? (
                          <ChevronUp size={14} className="text-electric-indigo" />
                        ) : (
                          <ChevronDown size={14} className="text-electric-indigo" />
                        )
                      ) : (
                        <ChevronsUpDown size={14} className="text-text-muted opacity-50" />
                      )}
                    </span>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paginatedData.map((row, rowIdx) => (
              <tr
                key={rowIdx}
                className="hover:bg-bg-tertiary/50 transition-colors"
              >
                <td className="px-2 py-1.5 text-text-muted border-b border-border-subtle">
                  {currentPage * pageSize + rowIdx + 1}
                </td>
                {columns.map((column) => (
                  <td
                    key={column}
                    className={cn(
                      'px-2 py-1.5 border-b border-border-subtle truncate',
                      getColumnWidth(column),
                      dtypes[column]?.includes('int') || dtypes[column]?.includes('float')
                        ? 'text-right font-mono'
                        : ''
                    )}
                    title={String(row[column] ?? '')}
                  >
                    {formatValue(row[column], dtypes[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-3 py-2 border-t border-border-default">
          <button
            onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
            disabled={currentPage === 0}
            className="px-2 py-1 text-small bg-bg-tertiary rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-bg-primary transition-colors"
          >
            Previous
          </button>
          <span className="text-small text-text-muted">
            Page {currentPage + 1} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={currentPage >= totalPages - 1}
            className="px-2 py-1 text-small bg-bg-tertiary rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-bg-primary transition-colors"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
