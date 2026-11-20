# ⚛️ Quantum Material Discovery Platform

<div align="center">

![Platform Banner](Frontend/src/assets/quantum-hero.png)

**An AI + Quantum Computing platform for discovering novel materials using inverse-design methodology.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-black?logo=flask)](https://flask.palletsprojects.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-3178C6?logo=typescript)](https://typescriptlang.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.3-6929C4?logo=qiskit)](https://qiskit.org)

[🚀 Getting Started](#-getting-started) • [📖 API Docs](#-api-endpoints) • [🧠 ML Model](#-ml-model) • [⚛️ Quantum Pipeline](#️-quantum-simulation) • [🖥️ Frontend](#️-frontend-pages)

</div>

---

## 🧬 What Is This?

The **Quantum Material Discovery Platform** is a full-stack research application that combines:

- 🤖 **Graph Neural Networks (GNN)** — predict formation energy & band gap of any compound
- ⚛️ **Variational Quantum Eigensolver (VQE)** — quantum simulation on Qiskit Aer
- 🔬 **Inverse Material Design** — give target properties, receive novel compound candidates
- 🏗️ **3D Crystal Structure Generation** — via pymatgen with CIF / POSCAR / XYZ export
- 📐 **Thermodynamic Stability Analysis** — convex hull & e_above_hull scoring
- 🌐 **Interactive Web UI** — React dashboard with 3D molecule viewer, charts, and simulation controls

> Instead of testing random materials in a lab, researchers input desired properties and the platform generates, screens, and validates novel compounds computationally — before any physical synthesis.

---

## 🗂️ Project Structure

```
QuantumMaterialDiscovery/
├── Backend/                          ← Python Flask API + ML/Quantum engine
│   ├── app/
│   │   ├── main.py                   ← Flask app entry point (port 5000)
│   │   ├── api/
│   │   │   └── routes.py             ← All REST API endpoints
│   │   ├── models/
│   │   │   └── gnn_model.py          ← MEGNet-inspired GNN (PyTorch Geometric)
│   │   ├── services/
│   │   │   ├── discovery_service.py          ← Inverse-design pipeline
│   │   │   ├── prediction_service.py         ← GNN inference + uncertainty
│   │   │   ├── vqe_service.py                ← VQE circuit & runner (Qiskit)
│   │   │   ├── structure_generator.py        ← Crystal structure + export
│   │   │   ├── convex_hull_estimator.py      ← Phase stability scoring
│   │   │   ├── novelty_checker.py            ← Novelty vs known databases
│   │   │   ├── chemistry_validator.py        ← Chemical validity checks
│   │   │   ├── physics_constraints.py        ← Physical plausibility
│   │   │   ├── synthesis_estimator.py        ← Synthetic accessibility
│   │   │   └── usecase_predictor.py          ← Application domain predictor
│   │   └── utils/
│   │       ├── element_loader.py             ← Periodic table data
│   │       ├── feature_engine.py             ← Formula → feature vector
│   │       └── graph_constructor.py          ← Atom graph builder (PyG)
│   ├── Dataset/                      ← Curated materials datasets (~18 MB total)
│   │   ├── qm9_final.csv             ← 9.4 MB — QM9 molecular dataset
│   │   ├── final_dataset.csv         ← 3.5 MB — merged master dataset
│   │   ├── real_materials.csv        ← 2.4 MB — known real materials
│   │   ├── mp_dataset.csv            ← 2.2 MB — Materials Project DFT data
│   │   ├── discovered_materials.csv  ← AI-discovered novel candidates
│   │   ├── aflow_formulas.csv        ← AFLOW database formulas
│   │   ├── oqmd_formulas.csv         ← Open Quantum Materials Database
│   │   └── jarvis_formulas.csv       ← JARVIS-DFT formulas
│   ├── .env                          ← API keys (Materials Project)
│   └── requirements.txt              ← All Python dependencies (pinned)
│
└── Frontend/                         ← React + Vite + TypeScript web app
    ├── src/
    │   ├── App.tsx                   ← Root app + React Router
    │   ├── pages/
    │   │   ├── Index.tsx             ← Dashboard / home
    │   │   ├── Materials.tsx         ← Materials database browser
    │   │   ├── MoleculeBuilder.tsx   ← Interactive molecule builder
    │   │   ├── Simulator.tsx         ← Quantum simulation runner
    │   │   ├── CandidateGenerator.tsx← Novel candidate generation
    │   │   ├── Insights.tsx          ← Analytics & trends
    │   │   └── Experiments.tsx       ← Experiment management
    │   └── components/
    │       ├── dashboard/            ← Hero, MetricCards
    │       ├── molecule-viewer/      ← 3D visualization (3Dmol.js)
    │       ├── layout/               ← Navbar, sidebar
    │       └── ui/                   ← 40+ shadcn/ui components
    ├── electron/                     ← Electron desktop wrapper
    ├── package.json
    └── vite.config.ts                ← Dev server on port 8080
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+**
- **Node.js 18+** and **npm**
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/GaneshbabuChinta2005/material-discovey-using-quantum.git
cd material-discovey-using-quantum
```

### 2. Backend Setup

```bash
cd Backend

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

> ⚠️ **Note:** `requirements.txt` includes PyTorch (CPU), Qiskit Aer, and pymatgen — expect ~5–10 min install time.

### 3. Configure Environment

Create `Backend/.env`:
```env
MATERIALS_PROJECT_API_KEY=your_api_key_here
```

Get a free API key at [materialsproject.org](https://materialsproject.org/api).

### 4. Start the Backend

```bash
cd Backend
python -m app.main
```

Backend runs at **http://localhost:5000**

### 5. Frontend Setup & Start

```bash
cd Frontend
npm install
npm run dev
```

Frontend runs at **http://localhost:8080**

---

## 🌐 Running Both Servers

Open two terminals:

```bash
# Terminal 1 — Backend
cd Backend && .venv\Scripts\python.exe -m app.main

# Terminal 2 — Frontend
cd Frontend && npm run dev
```

Then open **http://localhost:8080** in your browser.

---

## 📖 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `GET` | `/elements` | List elements + available groups |
| `GET` | `/model-info` | GNN model metadata & accuracy |
| `POST` | `/discover` | **Inverse-design material discovery** |
| `POST` | `/simulate` | **AI screening + VQE quantum simulation** |
| `GET` | `/export/<fmt>` | Download structure as `cif`, `poscar`, or `xyz` |
| `GET` | `/molecule/pubchem-structure?formula=X` | Fetch 3D structure from PubChem |

### Example: Discover Materials

```bash
curl -X POST http://localhost:5000/discover \
  -H "Content-Type: application/json" \
  -d '{
    "target_energy_min": -4.0,
    "target_energy_max": -0.5,
    "target_bandgap_min": 1.0,
    "target_bandgap_max": 3.0,
    "element_groups": ["transition_metals", "nonmetals"],
    "include_elements": ["Ti"],
    "max_atoms": 8,
    "num_candidates": 5,
    "stability_level": "high"
  }'
```

### Example: Quantum Simulate

```bash
curl -X POST http://localhost:5000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "formula": "TiO2",
    "n_qubits": 4,
    "max_iterations": 30
  }'
```

---

## 🧠 ML Model

**Architecture:** MEGNet-Inspired Graph Neural Network

```
Atomic Graph (9D nodes + 5D edges)
         ↓
Input Embedding (Linear → SiLU)
         ↓
6× EdgeConv Message-Passing Layers
  · Edge MLP: [xᵢ ‖ xⱼ ‖ eᵢⱼ] → message
  · Node MLP: [xᵢ ‖ Σ messages] → x'ᵢ
  · Skip connections + BatchNorm
         ↓
Global Pooling (Mean + Max + size feature)
         ↓
Readout MLP (4 layers, Dropout 0.15)
         ↓
Scalar Output: Formation Energy (eV/atom) or Band Gap (eV)
```

| Parameter | Value |
|-----------|-------|
| Hidden dim | 256 |
| Conv layers | 6 |
| Node features | 9D |
| Edge features | 5D |
| Dropout | 0.15 |
| Framework | PyTorch Geometric |

---

## ⚛️ Quantum Simulation

**Framework:** Qiskit 2.3.0 + Qiskit Aer 0.17.2

**Algorithm:** Variational Quantum Eigensolver (VQE)

**Pipeline:**
1. Build parameterized ansatz quantum circuit
2. Construct element-aware Hamiltonian from chemical formula
3. Optimize circuit parameters using classical optimizer
4. Return ground-state energy estimate on `AerSimulator`

**Parameters:** `n_qubits` (2–6) · `max_iterations` (10–60)

---

## 🖥️ Frontend Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | **Dashboard** | KPI metrics, recent simulations, top materials |
| `/materials` | **Materials** | Browse and filter the materials database |
| `/molecule-builder` | **Molecule Builder** | Interactive 3D molecule construction |
| `/simulator` | **Simulator** | Run quantum simulations for any formula |
| `/candidate-generator` | **Candidate Generator** | AI-powered novel material generation |
| `/insights` | **Insights** | Analytics, charts, and trend analysis |
| `/experiments` | **Experiments** | Manage and track discovery experiments |

---

## 🛠️ Full Tech Stack

### Backend
| Library | Version | Purpose |
|---------|---------|---------|
| Flask | 3.1.2 | REST API server |
| PyTorch | 2.10.0+cpu | GNN model |
| PyTorch Geometric | 2.7.0 | Graph neural network layers |
| Qiskit | 2.3.0 | Quantum circuit design |
| Qiskit Aer | 0.17.2 | Quantum simulation backend |
| pymatgen | 2025.10.7 | Crystal structure & analysis |
| scikit-learn | 1.7.2 | Supporting ML utilities |
| mp-api | 0.39.5 | Materials Project API client |
| ASE | 3.27.0 | Atomic Simulation Environment |
| spglib | 2.7.0 | Crystallographic symmetry |
| numpy / scipy / pandas | latest | Scientific computing |

### Frontend
| Library | Version | Purpose |
|---------|---------|---------|
| React | 18.3.1 | UI framework |
| TypeScript | 5.8.3 | Type safety |
| Vite | 5.4.19 | Build tool |
| Tailwind CSS | 3.4.17 | Styling |
| shadcn/ui | latest | UI components (40+) |
| React Router | 6.30.1 | Client-side routing |
| TanStack Query | 5.83.0 | Server state management |
| Three.js + R3F | 0.160.1 | 3D rendering |
| 3Dmol.js | 2.5.4 | Molecular visualization |
| Recharts | 2.15.4 | Data charts |
| Electron | 41.2.1 | Desktop app (optional) |

---

## 🔬 Discovery Pipeline

```
User: target properties (energy, bandgap, stability, elements)
                    ↓
         Composition Explorer
     (element groups → formulas)
                    ↓
         Compound Generator
      (enumerate stoichiometries)
                    ↓
       Chemistry Validator
   (oxidation states, charge balance)
                    ↓
       Physics Constraints
    (bond lengths, coordination)
                    ↓
       GNN Property Prediction
   (formation energy + band gap + uncertainty)
                    ↓
      Convex Hull Estimator
   (phase stability, e_above_hull)
                    ↓
         Novelty Checker
  (vs Materials Project + AFLOW + OQMD)
                    ↓
       Stability Scorer (composite)
                    ↓
     VQE Quantum Validation
   (Qiskit ground-state energy)
                    ↓
     Application Predictor
  (energy / battery / semiconductor / defense)
                    ↓
     3D Structure Generator
  (crystal structure + CIF / POSCAR / XYZ)
                    ↓
            Results → Frontend
```

---

## 🖥️ Desktop App (Electron)

Build a standalone desktop installer:

```bash
cd Frontend
npm run electron:build
# Output: Frontend/release/Quantum Discovery Platform Setup.exe
```

---

## 📊 Datasets Used

| Dataset | Size | Source |
|---------|------|--------|
| QM9 | 9.4 MB | ~130k small molecules with DFT properties |
| Master Dataset | 3.5 MB | Merged multi-source training data |
| Real Materials | 2.4 MB | Known experimentally verified materials |
| Materials Project | 2.2 MB | DFT-computed crystal properties |
| AFLOW | 78 KB | Crystal structure formula database |
| OQMD | 45 KB | Open Quantum Materials Database |
| JARVIS-DFT | 13 KB | 2D & 3D material formulas |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is for research and educational purposes.

---

## 👨‍💻 Authors

- **Ganeshbabu Chinta** — [GitHub](https://github.com/GaneshbabuChinta2005)

---

<div align="center">
  <sub>Built with ⚛️ quantum physics, 🤖 machine learning, and ☕ coffee</sub>
</div>
