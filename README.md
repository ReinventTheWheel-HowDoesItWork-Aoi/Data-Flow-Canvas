# Data Flow Canvas

**Canvas for Data Science** — A browser-based visual data science platform that enables users to build, execute, and share data transformation and analysis pipelines through an intuitive drag-and-drop interface.

🌐 **Live Website**: [https://dataflowcanvas.com](https://dataflowcanvas.com)

![Data Flow Canvas](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)
![React](https://img.shields.io/badge/react-18.3-61DAFB)
![TypeScript](https://img.shields.io/badge/typescript-5.6-3178C6)

## Features

### Privacy First
- **100% client-side processing** — All data processing happens in your browser using WebAssembly
- **No data upload** — Your data never leaves your device
- **No account required** — Start using immediately without sign-up

### Visual Pipeline Builder
- **Drag-and-drop interface** — Build pipelines visually with React Flow
- **30 block types** — Data input, transformation, analysis, visualization, and export
- **Real-time preview** — See data changes as you build

### Python Powered
- **Pyodide integration** — Full Python data science stack (pandas, numpy, scikit-learn) running in WebAssembly
- **No installation** — Everything runs in your browser

### Real-time Collaboration
- **P2P connections** — Work together using WebRTC
- **CRDT sync** — Conflict-free collaborative editing with Y.js

## Block Types

| Category | Blocks |
|----------|--------|
| **Data Input** | Load Data, Sample Data, Create Dataset |
| **Transform** | Filter Rows, Select Columns, Sort, Group & Aggregate, Join, Derive Column, Handle Missing, Rename Columns, Deduplicate, Sample Rows, Limit Rows, Pivot, Unpivot, Union, Split Column, Merge Columns, Conditional Column |
| **Analysis** | Statistics, Regression, Clustering, PCA, Outlier Detection, Classification, Normality Test |
| **Visualization** | Chart, Table |
| **Output** | Export |

## Getting Started

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

## Tech Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS
- **Canvas**: React Flow (@xyflow/react)
- **State Management**: Zustand with Zundo (undo/redo)
- **Python Runtime**: Pyodide (WebAssembly)
- **Storage**: IndexedDB via Dexie.js
- **Collaboration**: Y.js + WebRTC
- **Build Tool**: Vite

## Project Structure

```
src/
├── components/
│   ├── ui/            # Design system components
│   ├── blocks/        # Pipeline block components
│   ├── canvas/        # React Flow canvas
│   ├── visualization/ # Chart and table renderers
│   └── layout/        # Layout components
├── pages/             # Route pages
├── stores/            # Zustand stores
├── hooks/             # Custom React hooks
├── lib/
│   ├── pyodide/       # Pyodide integration
│   ├── execution/     # Pipeline execution engine
│   ├── storage/       # IndexedDB storage
│   ├── collaboration/ # WebRTC collaboration
│   └── utils/         # Utility functions
├── types/             # TypeScript type definitions
├── constants/         # Block definitions, etc.
└── test/              # Test utilities
```

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |
| `npm run test` | Run unit tests |
| `npm run test:e2e` | Run E2E tests |

## Author

**Lavelle Hatcher Jr** — Creator and maintainer of Data Flow Canvas

## License

This project is licensed under the **AGPL-3.0 License** — see the [LICENSE](LICENSE) file for details.

### Commercial Licensing

The AGPL-3.0 license requires that any modifications or derivative works also be open-sourced under the same license.

**For commercial use without AGPL-3.0 obligations**, a separate commercial license is available. This includes:
- Using Data Flow Canvas in proprietary/closed-source products
- Offering Data Flow Canvas as a hosted service without source disclosure
- Enterprise deployment with custom terms

💼 **Contact**: [Lavelle Hatcher Jr](https://www.linkedin.com/in/lavellemhatcherjr)

## Acknowledgments

- [Pyodide](https://pyodide.org/) — Python in the browser
- [React Flow](https://reactflow.dev/) — Node-based graph library
- [Y.js](https://yjs.dev/) — CRDT framework for collaborative editing
- [Tailwind CSS](https://tailwindcss.com/) — Utility-first CSS framework
