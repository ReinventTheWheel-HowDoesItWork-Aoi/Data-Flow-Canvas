# Data Flow Canvas

**Canvas for Data Science** — A browser-based visual data science platform that enables users to build, execute, and share data transformation and analysis pipelines through an intuitive drag-and-drop interface.

🌐 **Live Website**: [https://dataflowcanvas.com](https://dataflowcanvas.com)

![Data Flow Canvas](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)
![React](https://img.shields.io/badge/react-18.3-61DAFB)
![TypeScript](https://img.shields.io/badge/typescript-5.6-3178C6)

## ✨ Features

### 🔒 Privacy First
- **100% client-side processing** — All data transformation and analysis happens in your browser using WebAssembly
- **Your data stays local** — CSV files and pipeline data are processed entirely in your browser and never uploaded to any server
- **Secure authentication** — Account powered by Supabase; only your profile info is stored, not your data

### 🎨 Visual Pipeline Builder
- **Drag-and-drop interface** — Build pipelines visually with React Flow
- **150 block types** — Data input (3), transformation (64), analysis (58), visualization (24), and export (1)
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
| **Transform** | Filter Rows, Select Columns, Sort, Group & Aggregate, Join, Derive Column, Handle Missing, Rename Columns, Deduplicate, Sample Rows, Limit Rows, Pivot, Unpivot, Union, Split Column, Merge Columns, Conditional Column, Date/Time Extract, String Operations, Window Functions, Bin/Bucket, Rank, Type Conversion, Fill Forward/Backward, Lag/Lead, Row Number, Date Difference, Transpose, String Pad, Cumulative Operations, Replace Values, Percent Change, Round Numbers, Percent of Total, Absolute Value, Column Math, Extract Substring, Parse Date, Split to Rows, Clip Values, Standardize Text, Case When, Log Transform, Interpolate Missing, Date Truncate, Period-over-Period, Hash Column, Expand Date Range, String Similarity, Generate Sequence, Top N per Group, First/Last per Group |
| **Analysis** | Statistics, Regression, Clustering, PCA, Outlier Detection, Classification, Normality Test, Hypothesis Testing, Time Series, Feature Importance, Cross Validation, Data Profiling, Value Counts, Cross Tabulation, Scaling, Encoding, A/B Test, Cohort Analysis, RFM Analysis, ANOVA, Chi-Square Test, Correlation Analysis, Survival Analysis, Association Rules, Sentiment Analysis, Moving Average, Train/Test Split, Model Evaluation, K-Nearest Neighbors, Naive Bayes, Gradient Boosting, Pareto Analysis, Trend Analysis, Forecasting, Percentile Analysis, Distribution Fit, Text Preprocessing, TF-IDF Vectorization, Topic Modeling, Similarity Analysis, SVM, XGBoost, Model Explainability, Regression Diagnostics, VIF Analysis, Funnel Analysis, Customer Lifetime Value, Churn Prediction, Growth Metrics, Attribution Modeling, Break-even Analysis, Confidence Intervals, Bootstrap Analysis, Post-hoc Tests, Power Analysis, Bayesian Inference, Data Quality Score, Change Point Detection |
| **Visualization** | Chart, Table, Correlation Matrix, Violin Plot, Pair Plot, Area Chart, Stacked Chart, Bubble Chart, Q-Q Plot, Confusion Matrix, ROC Curve, Funnel Chart, Sankey Diagram, Treemap, Sunburst Chart, Gauge Chart, Radar Chart, Waterfall Chart, Candlestick Chart, Choropleth Map, Word Cloud, Pareto Chart, Parallel Coordinates, Dendrogram |
| **Output** | Export |

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

- **Frontend**: React 18, TypeScript, Tailwind CSS
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
