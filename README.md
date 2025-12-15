# Data Flow Canvas

English | [日本語](README.ja.md)

**Canvas for Data Science** — A browser-based visual data science platform that enables users to build, execute, and share data transformation and analysis pipelines through an intuitive drag-and-drop interface.

🌐 **Live Website**: [https://dataflowcanvas.com](https://dataflowcanvas.com)

🚀 **Product Hunt**: [https://www.producthunt.com/products/data-flow-canvas](https://www.producthunt.com/products/data-flow-canvas)

![Data Flow Canvas](https://img.shields.io/badge/version-1.1.4-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)
![React](https://img.shields.io/badge/react-19.2-61DAFB)
![TypeScript](https://img.shields.io/badge/typescript-5.9-3178C6)

## ✨ Features

### 🔒 Privacy First
- **100% client-side processing** — All data transformation and analysis happens in your browser using WebAssembly
- **Your data stays local** — CSV files and pipeline data are processed entirely in your browser and never uploaded to any server
- **Secure authentication** — Account powered by Supabase; only your profile info is stored, not your data

### 🎨 Visual Pipeline Builder
- **Drag-and-drop interface** — Build pipelines visually with React Flow
- **303 block types** — Data input (3), transformation (86), analysis (142), visualization (70), and export (2)
- **Real-time preview** — See data changes as you build

### 🐍 Python Powered
- **Pyodide integration** — Full Python data science stack (pandas, numpy, scikit-learn) running in WebAssembly
- **No installation** — Everything runs in your browser

### 👥 Real-time Collaboration
- **Secure server-relayed connections** — Work together through Supabase Realtime (IP addresses hidden)
- **Authentication required** — Only signed-in users can collaborate
- **Smart limits** — 30 max concurrent collaborators globally, 3 per session

### 🌍 Internationalization
- **Multi-language support** — Available in English and Japanese

## 📦 Block Types

| Category | Blocks |
|----------|--------|
| **Data Input** | Load Data, Sample Data, Create Dataset |
| **Transform** | Filter Rows, Select Columns, Sort, Group & Aggregate, Join, Derive Column, Handle Missing, Rename Columns, Deduplicate, Sample Rows, Limit Rows, Pivot, Unpivot, Union, Split Column, Merge Columns, Conditional Column, Date/Time Extract, String Operations, Window Functions, Bin/Bucket, Rank, Type Conversion, Fill Forward/Backward, Lag/Lead, Row Number, Date Difference, Transpose, String Pad, Cumulative Operations, Replace Values, Percent Change, Round Numbers, Percent of Total, Absolute Value, Column Math, Extract Substring, Parse Date, Split to Rows, Clip Values, Standardize Text, Case When, Log Transform, Interpolate Missing, Date Truncate, Period-over-Period, Hash Column, Expand Date Range, String Similarity, Generate Sequence, Top N per Group, First/Last per Group, One-Hot Encode, Label Encode, Ordinal Encode, Min-Max Normalize, Z-Score Standardize, Rolling Statistics, Resample Time Series, Regex Replace, Expand JSON Column, Add Unique ID, Missing Value Indicator, Quantile Transform, Fuzzy Join, Memory Optimizer, Cyclical Time Encoder, Geographic Distance, Rare Category Combiner, Smart Auto-Cleaner, Interaction Generator, Fuzzy Deduplicator, Array Aggregator, Target-Aware Binning |
| **Analysis** | Statistics, Regression, Clustering, PCA, Outlier Detection, Classification, Normality Test, Hypothesis Testing, Time Series, Feature Importance, Cross Validation, Data Profiling, Value Counts, Cross Tabulation, Scaling, Encoding, A/B Test, Cohort Analysis, RFM Analysis, ANOVA, Chi-Square Test, Correlation Analysis, Survival Analysis, Association Rules, Sentiment Analysis, Moving Average, Train/Test Split, Model Evaluation, K-Nearest Neighbors, Naive Bayes, Gradient Boosting, Pareto Analysis, Trend Analysis, Forecasting, Percentile Analysis, Distribution Fit, Text Preprocessing, TF-IDF Vectorization, Topic Modeling, Similarity Analysis, SVM, XGBoost, Model Explainability, Regression Diagnostics, VIF Analysis, Funnel Analysis, Customer Lifetime Value, Churn Prediction, Growth Metrics, Attribution Modeling, Break-even Analysis, Confidence Intervals, Bootstrap Analysis, Post-hoc Tests, Power Analysis, Bayesian Inference, Data Quality Score, Change Point Detection, Isolation Forest, ARIMA Forecasting, Seasonal Decomposition, Monte Carlo Simulation, Propensity Score Matching, Difference-in-Differences, Factor Analysis, DBSCAN Clustering, Elastic Net Regression, Vector Autoregression (VAR), Interrupted Time Series, Granger Causality Test, Local Outlier Factor, Feature Selection, Outlier Treatment, Data Drift Detection, Polynomial Features, Multi-Output Prediction, Probability Calibration, t-SNE Reduction, Statistical Tests Suite, Optimal Binning, Correlation Finder, A/B Test Calculator, Target Encoding, Learning Curves, Imbalanced Data Handler, Hyperparameter Tuning, Ensemble Stacking, Advanced Imputation, UMAP Reduction, Cluster Validation, Model Comparison, Time Series CV, Uplift Modeling, Quantile Regression, Adversarial Validation, Custom Python Code, SQL Query, Auto-EDA, Data Validation, Neural Network, Auto Feature Engineering, SHAP Interpretation, AutoML, Multivariate Anomaly Detection, Causal Impact Analysis, Model Registry, Comprehensive EDA Report, SHAP Deep Explainer, STL Time Series Decomposition, Multi-Algorithm Anomaly Detection, Automated Feature Engineering Pipeline, Distribution Drift Monitor, Smart Resampling for Imbalanced Data, Collinearity Diagnostics, Bayesian A/B Test Calculator, Nested Cross-Validation, Gaussian Mixture Model, Dynamic Time Warping, LIME Explainer, Bayesian Optimization, Time Series Features, Robust Regression, Kernel Density Estimation, Spectral Clustering, Cross-Correlation Analysis, Manifold Learning, Semi-Supervised Learning, Multi-Label Classification, Conformal Prediction, One-Class SVM, Elliptic Envelope, Isotonic Regression, Power Transform, Mutual Information Selection, Sequential Feature Selection, Permutation Importance, ACF/PACF Analysis, Stationarity Testing, Exponential Smoothing, Copula Analysis, Variance Threshold, Hierarchical Clustering |
| **Visualization** | Chart, Table, Correlation Matrix, Violin Plot, Pair Plot, Area Chart, Stacked Chart, Bubble Chart, Q-Q Plot, Confusion Matrix, ROC Curve, Funnel Chart, Sankey Diagram, Treemap, Sunburst Chart, Gauge Chart, Radar Chart, Waterfall Chart, Candlestick Chart, Choropleth Map, Word Cloud, Pareto Chart, Parallel Coordinates, Dendrogram, Box Plot, Heatmap, Scatter Map, Grouped Histogram, Network Graph, Calendar Heatmap, Faceted Chart, Density Plot, Error Bar Chart, Dot Plot, Slope Chart, Grouped Bar Chart, Bump Chart, Donut Chart, Horizontal Bar Chart, 3D Scatter Plot, Contour Plot, Hexbin Plot, Ridge Plot, Strip Plot, Bullet Chart, Pyramid Chart, Timeline Chart, 3D Surface Plot, Marginal Histogram, Dumbbell Chart, SHAP Summary Plot, Partial Dependence Plot, Feature Importance Plot, ICE Plot, Precision-Recall Curve, Learning Curve Plot, Residual Plot, Actual vs Predicted Plot, Calibration Curve, Lift Chart, Elbow Plot, Silhouette Plot, t-SNE/UMAP Plot, Missing Value Heatmap, Outlier Detection Plot, Distribution Comparison Plot, ECDF Plot, Andrews Curves, CV Results Plot, Hyperparameter Heatmap |
| **Output** | Export, Pipeline Export |

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- npm 9+

### Installation

```bash
# Clone the repository
git clone https://github.com/ReinventTheWheel-HowDoesItWork-Aoi/Data-Flow-Canvas.git
cd Data-Flow-Canvas

# Install dependencies
npm install

# Start development server
npm run dev
```

### Building for Production

```bash
# Build the application
npm run build

# Preview the build
npm run preview
```

## 🛠️ Tech Stack

- **Frontend**: React 19, TypeScript, Tailwind CSS 4
- **Canvas**: React Flow (@xyflow/react)
- **State Management**: Zustand with Zundo (undo/redo)
- **Python Runtime**: Pyodide (WebAssembly)
- **Authentication**: Supabase
- **Storage**: IndexedDB via Dexie.js
- **Collaboration**: Supabase Realtime (Broadcast + Presence)
- **Build Tool**: Vite

## 📁 Project Structure

```
src/
├── components/
│   ├── ui/            # Design system components
│   ├── blocks/        # Pipeline block components
│   ├── canvas/        # React Flow canvas
│   ├── visualization/ # Chart and table renderers
│   ├── layout/        # Layout components
│   └── auth/          # Authentication components
├── pages/             # Route pages
├── stores/            # Zustand stores
├── hooks/             # Custom React hooks
├── lib/
│   ├── pyodide/       # Pyodide integration
│   ├── execution/     # Pipeline execution engine
│   ├── storage/       # IndexedDB storage
│   ├── collaboration/ # Supabase Realtime collaboration
│   ├── utils/         # Utility functions
│   ├── i18n/          # Internationalization
│   └── supabase/      # Supabase client
├── types/             # TypeScript type definitions
├── constants/         # Block definitions, etc.
└── test/              # Test utilities
```

## 📜 Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |
| `npm run test` | Run unit tests |
| `npm run test:e2e` | Run E2E tests |

## 👤 Author

**Lavelle Hatcher Jr** — Creator and maintainer of Data Flow Canvas

## 📄 License

This project is licensed under the **AGPL-3.0 License** — see the [LICENSE](LICENSE) file for details.

### Commercial Licensing

The AGPL-3.0 license requires that any modifications or derivative works also be open-sourced under the same license.

**For commercial use without AGPL-3.0 obligations**, a separate commercial license is available. This includes:
- Using Data Flow Canvas in proprietary/closed-source products
- Offering Data Flow Canvas as a hosted service without source disclosure
- Enterprise deployment with custom terms

💼 **Contact**: [Lavelle Hatcher Jr](https://www.linkedin.com/in/lavellemhatcherjr)

## 🙏 Acknowledgments

- [Pyodide](https://pyodide.org/) — Python in the browser
- [React Flow](https://reactflow.dev/) — Node-based graph library
- [Supabase](https://supabase.com/) — Authentication, database, and real-time collaboration
- [Tailwind CSS](https://tailwindcss.com/) — Utility-first CSS framework
