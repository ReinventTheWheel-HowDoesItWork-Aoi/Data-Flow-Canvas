/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React from 'react';
import { X, AlertCircle, CheckCircle, Info, AlertTriangle } from 'lucide-react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { useUIStore } from '@/stores/uiStore';
import { useExecutionStore } from '@/stores/executionStore';
import { cn } from '@/lib/utils/cn';

export function BottomPanel() {
  const { isBottomPanelOpen, bottomPanelTab, setBottomPanelTab, toggleBottomPanel } =
    useUIStore();
  const { logs, progress, isRunning } = useExecutionStore();

  if (!isBottomPanelOpen) {
    return null;
  }

  return (
    <div className="h-48 bg-bg-secondary border-t border-border-default flex flex-col">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border-default">
        <Tabs
          value={bottomPanelTab}
          onValueChange={(v) => setBottomPanelTab(v as typeof bottomPanelTab)}
        >
          <TabsList>
            <TabsTrigger value="logs">
              Logs
              {logs.length > 0 && (
                <Badge variant="info" className="ml-2">
                  {logs.length}
                </Badge>
              )}
            </TabsTrigger>
            <TabsTrigger value="output">Output</TabsTrigger>
          </TabsList>
        </Tabs>

        <div className="flex items-center gap-4">
          {isRunning && (
            <div className="flex items-center gap-2">
              <div className="h-1 w-32 bg-bg-tertiary rounded-full overflow-hidden">
                <div
                  className="h-full bg-electric-indigo transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <span className="text-small text-text-muted">{Math.round(progress)}%</span>
            </div>
          )}
          <Button variant="ghost" size="sm" onClick={toggleBottomPanel}>
            <X size={18} />
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {bottomPanelTab === 'logs' && <LogsView logs={logs} />}
        {bottomPanelTab === 'output' && <OutputView />}
      </div>
    </div>
  );
}

function LogsView({ logs }: { logs: any[] }) {
  const getIcon = (level: string) => {
    switch (level) {
      case 'error':
        return <AlertCircle size={14} className="text-warm-coral" />;
      case 'warning':
        return <AlertTriangle size={14} className="text-golden-amber" />;
      case 'info':
      default:
        return <Info size={14} className="text-electric-indigo" />;
    }
  };

  if (logs.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted">
        <p>No logs yet. Run the pipeline to see execution logs.</p>
      </div>
    );
  }

  return (
    <div className="p-2 space-y-1 font-mono text-small">
      {logs.map((log) => (
        <div
          key={log.id}
          className={cn(
            'flex items-start gap-2 px-2 py-1 rounded',
            log.level === 'error' && 'bg-warm-coral/10',
            log.level === 'warning' && 'bg-golden-amber/10'
          )}
        >
          {getIcon(log.level)}
          <span className="text-text-muted">
            {new Date(log.timestamp).toLocaleTimeString()}
          </span>
          <span className="text-text-primary flex-1">{log.message}</span>
        </div>
      ))}
    </div>
  );
}

function OutputView() {
  return (
    <div className="flex items-center justify-center h-full text-text-muted">
      <p>Pipeline output will appear here</p>
    </div>
  );
}
