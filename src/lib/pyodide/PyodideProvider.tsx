/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import type { PyodideInterface } from 'pyodide';

interface PyodideContextValue {
  pyodide: PyodideInterface | null;
  isLoading: boolean;
  isReady: boolean;
  error: string | null;
  loadProgress: number;
  excelSupport: boolean;
  runPython: (code: string) => Promise<unknown>;
  runPythonAsync: (code: string) => Promise<unknown>;
  initializePyodide: () => Promise<void>;
}

const PyodideContext = createContext<PyodideContextValue | null>(null);

export function PyodideProvider({ children }: { children: React.ReactNode }) {
  const [pyodide, setPyodide] = useState<PyodideInterface | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadProgress, setLoadProgress] = useState(0);
  const [excelSupport, setExcelSupport] = useState(false);
  const pyodideRef = useRef<PyodideInterface | null>(null);
  const initializingRef = useRef(false);

  const initPyodide = useCallback(async () => {
    if (pyodideRef.current || initializingRef.current) return;
    initializingRef.current = true;

    try {
      setIsLoading(true);
      setLoadProgress(10);

      // Dynamically import Pyodide
      const pyodideModule = await import('pyodide');
      setLoadProgress(30);

      const py = await pyodideModule.loadPyodide({
        indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.29.0/full/',
      });
      setLoadProgress(60);

      // Load essential packages
      await py.loadPackage(['pandas', 'numpy', 'micropip']);
      setLoadProgress(80);

      // Load additional packages via micropip
      const micropip = py.pyimport('micropip');

      // Install scikit-learn (required for ML features)
      await micropip.install(['scikit-learn']);
      setLoadProgress(85);

      // Install openpyxl for Excel (.xlsx) support - this is critical for Excel files
      // openpyxl depends on et_xmlfile, so install both
      let openpyxlInstalled = false;

      // Helper function to verify openpyxl can actually be imported
      const verifyOpenpyxl = async (): Promise<boolean> => {
        try {
          await py.runPythonAsync('import openpyxl; print("openpyxl version:", openpyxl.__version__)');
          return true;
        } catch {
          return false;
        }
      };

      // Helper function with retry logic for network failures
      const installWithRetry = async (
        installFn: () => Promise<void>,
        maxRetries: number = 3
      ): Promise<boolean> => {
        for (let attempt = 0; attempt < maxRetries; attempt++) {
          try {
            await installFn();
            return true;
          } catch (e) {
            if (attempt < maxRetries - 1) {
              // Wait before retrying (exponential backoff)
              await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, attempt)));
            }
          }
        }
        return false;
      };

      // Try multiple installation strategies with retries
      const installStrategies = [
        // Strategy 1: Install dependencies first, then openpyxl
        async () => {
          await micropip.install('et_xmlfile');
          await micropip.install('openpyxl');
        },
        // Strategy 2: Install both together
        async () => {
          await micropip.install(['et_xmlfile', 'openpyxl']);
        },
        // Strategy 3: Install openpyxl alone (dependencies might be bundled in newer versions)
        async () => {
          await micropip.install('openpyxl');
        },
        // Strategy 4: Try specific versions known to work with Pyodide
        async () => {
          await micropip.install(['et_xmlfile==1.1.0', 'openpyxl==3.1.2']);
        },
        // Strategy 5: Try latest openpyxl with keep_going flag
        async () => {
          await micropip.install('openpyxl', { keep_going: true });
        },
      ];

      for (let i = 0; i < installStrategies.length && !openpyxlInstalled; i++) {
        const installed = await installWithRetry(installStrategies[i], 2);
        if (installed) {
          // Verify the import actually works
          openpyxlInstalled = await verifyOpenpyxl();
          if (openpyxlInstalled) {
            console.log(`openpyxl installed successfully (strategy ${i + 1})`);
          }
        } else {
          console.warn(`openpyxl installation strategy ${i + 1} failed after retries`);
        }
      }

      if (!openpyxlInstalled) {
        console.error('All openpyxl installation strategies failed. Excel (.xlsx) files will not be supported.');
        console.error('Users will need to convert Excel files to CSV format.');
      }

      setExcelSupport(openpyxlInstalled);
      setLoadProgress(92);

      // Install xlrd as fallback for older .xls files (xlrd 2.0+ only supports .xls, not .xlsx)
      try {
        await installWithRetry(async () => {
          await micropip.install('xlrd');
        }, 2);
        console.log('xlrd installed for legacy Excel support');
      } catch {
        console.warn('xlrd not available - older .xls files may not be supported');
      }
      setLoadProgress(95);

      pyodideRef.current = py;
      setPyodide(py);
      setIsReady(true);
      setLoadProgress(100);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load Pyodide');
      console.error('Pyodide initialization error:', err);
    } finally {
      setIsLoading(false);
      initializingRef.current = false;
    }
  }, []);

  // Auto-initialize Pyodide when provider mounts
  useEffect(() => {
    initPyodide();
  }, [initPyodide]);

  const runPython = useCallback(
    async (code: string) => {
      const py = pyodideRef.current;
      if (!py) throw new Error('Pyodide not initialized');
      return py.runPython(code);
    },
    []
  );

  const runPythonAsync = useCallback(
    async (code: string) => {
      const py = pyodideRef.current;
      if (!py) throw new Error('Pyodide not initialized');
      return py.runPythonAsync(code);
    },
    []
  );

  return (
    <PyodideContext.Provider
      value={{
        pyodide,
        isLoading,
        isReady,
        error,
        loadProgress,
        excelSupport,
        runPython,
        runPythonAsync,
        initializePyodide: initPyodide,
      }}
    >
      {children}
    </PyodideContext.Provider>
  );
}

export function usePyodide() {
  const context = useContext(PyodideContext);
  if (!context) {
    throw new Error('usePyodide must be used within a PyodideProvider');
  }
  return context;
}
