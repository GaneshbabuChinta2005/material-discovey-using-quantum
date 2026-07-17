/**
 * CompoundViewer3D — 2D Topology Renderer (Overridden to 2D per user request)
 *
 * Uses RDKit to render a sleek 2D topological mapping of a given formula,
 * discarding the 3D crystal lattice structure to focus entirely on the valid
 * generic structural formula representation. 
 */
import { useEffect, useRef, useState, useMemo } from "react";

interface Atom {
    id: number;
    symbol: string;
    x: number;
    y: number;
    z: number;
    color: string;
}

interface Bond {
    from: number;
    to: number;
}

interface LatticeInfo {
    crystal_system?: string;
    symmetry_confidence?: number;
    prototype_tag?: string;
    dimensionality?: string;
}

interface CompoundStructure {
    atoms: Atom[];
    bonds: Bond[];
    total_atoms: number;
    lattice?: LatticeInfo;
}

interface CompoundViewer3DProps {
    structure: CompoundStructure;
    formula?: string;
    height?: string;
    className?: string;
    compoundName?: string;
}

// ─── CPK colors and parameters ───
const ELEMENT_COLORS: Record<string, string> = {
    H: "#FFFFFF", He: "#D9FFFF", Li: "#CC80FF", Be: "#C2FF00", B: "#FFB5B5",
    C: "#909090", N: "#3050F8", O: "#FF0D0D", F: "#90E050", Ne: "#B3E3F5",
    Na: "#AB5CF2", Mg: "#8AFF00", Al: "#BFA6A6", Si: "#F0C8A0", P: "#FF8000",
    S: "#FFFF30", Cl: "#1FF01F", Ar: "#80D1E3", K: "#8F40D4", Ca: "#3DFF00",
    Ti: "#BFC2C7", V: "#A6A6AB", Cr: "#8A99C7", Mn: "#9C7AC7", Fe: "#E06633",
    Co: "#F090A0", Ni: "#50D050", Cu: "#C88033", Zn: "#7D80B0", Br: "#A62929",
    I: "#940094", Au: "#FFD123", default: "#A0A0A0"
};

const COVALENT_RADII: Record<string, number> = {
    H: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71,
    O: 0.66, F: 0.57, Ne: 0.58, Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11,
    P: 1.07, S: 1.05, Cl: 1.02, K: 2.03, Ca: 1.76, Ti: 1.60, V: 1.53,
    Cr: 1.39, Mn: 1.39, Fe: 1.32, Co: 1.26, Ni: 1.24, Cu: 1.32, Zn: 1.22
};

function getColor(symbol: string): string {
    return ELEMENT_COLORS[symbol] || ELEMENT_COLORS.default;
}

// ─── Utility: Parse Formula ("Y2CrO4" -> UI Elements) ───
function parseFormulaToElements(formula: string): { symbol: string; count: number }[] {
    const rx = /([A-Z][a-z]*)(\d*)/g;
    let match;
    const res: { symbol: string; count: number }[] = [];
    while ((match = rx.exec(formula)) !== null) {
        res.push({ symbol: match[1], count: parseInt(match[2] || "1") });
    }
    return res;
}

// ─── Utility: Fallback Structure ───
function generateTargetedFallbackSDF(formulaStr: string): string {
    const elements = parseFormulaToElements(formulaStr || "C");
    const atoms: { symbol: string; x: number; y: number; z: number }[] = [];
    const bonds: { from: number; to: number; order: number }[] = [];
    const totalAtoms = elements.reduce((s, e) => s + e.count, 0);

    if (totalAtoms === 0) return "";

    let idx = 0;
    for (const el of elements) {
        const cr = COVALENT_RADII[el.symbol] || 1.3;
        for (let i = 0; i < el.count; i++) {
            if (totalAtoms === 1) {
                atoms.push({ symbol: el.symbol, x: 0, y: 0, z: 0 });
            } else if (totalAtoms === 2) {
                atoms.push({ symbol: el.symbol, x: idx === 0 ? -cr : cr, y: 0, z: 0 });
            } else {
                const phi = Math.acos(1 - (2 * (idx + 0.5)) / totalAtoms);
                const theta = Math.PI * (1 + Math.sqrt(5)) * idx;
                const r = 0.9 + cr * 0.5 + Math.sqrt(totalAtoms) * 0.25;
                atoms.push({
                    symbol: el.symbol,
                    x: r * Math.sin(phi) * Math.cos(theta),
                    y: r * Math.sin(phi) * Math.sin(theta),
                    z: r * Math.cos(phi),
                });
            }
            idx++;
        }
    }

    const bondSet = new Set<string>();
    function addBond(i: number, j: number) {
        const key = i < j ? `${i}-${j}` : `${j}-${i}`;
        if (bondSet.has(key)) return;
        bondSet.add(key);
        bonds.push({ from: i + 1, to: j + 1, order: 1 });
    }
    function dist(a: any, b: any) {
        return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
    }

    for (let i = 0; i < atoms.length; i++) {
        let nj = -1, nd = Infinity;
        for (let j = 0; j < atoms.length; j++) {
            if (i === j) continue;
            const d = dist(atoms[i], atoms[j]);
            if (d < nd) { nd = d; nj = j; }
        }
        if (nj >= 0) addBond(i, nj);
    }
    for (let i = 0; i < atoms.length; i++) {
        for (let j = i + 1; j < atoms.length; j++) {
            const ri = COVALENT_RADII[atoms[i].symbol] || 1.0;
            const rj = COVALENT_RADII[atoms[j].symbol] || 1.0;
            if (dist(atoms[i], atoms[j]) < (ri + rj) * 1.5 + 0.8) addBond(i, j);
        }
    }

    const lines: string[] = [];
    lines.push("FallbackStructure");
    lines.push("  QuantumMD   2D");
    lines.push("Generated locally");
    lines.push(`${String(atoms.length).padStart(3)}${String(bonds.length).padStart(3)}  0  0  0  0  0  0  0  0999 V2000`);
    for (const a of atoms) {
        lines.push(`${a.x.toFixed(4).padStart(10)}${a.y.toFixed(4).padStart(10)}${a.z.toFixed(4).padStart(10)} ${a.symbol.padEnd(3)} 0  0  0  0  0  0  0  0  0  0  0  0`);
    }
    for (const b of bonds) {
        lines.push(`${String(b.from).padStart(3)}${String(b.to).padStart(3)}${String(b.order).padStart(3)}  0  0  0  0`);
    }
    lines.push("M  END");
    lines.push("$$$$");
    return lines.join("\n");
}

let rdkitLoaded = false;
let rdkitPromise: Promise<any> | null = null;
function loadRDKit(): Promise<any> {
    if (rdkitLoaded && (window as any).RDKit) return Promise.resolve((window as any).RDKit);
    if (rdkitPromise) return rdkitPromise;
    rdkitPromise = new Promise((resolve, reject) => {
        if ((window as any).initRDKitModule) {
            (window as any).initRDKitModule().then((RDKit: any) => {
                (window as any).RDKit = RDKit;
                rdkitLoaded = true;
                resolve(RDKit);
            }).catch(reject);
            return;
        }
        const script = document.createElement("script");
        script.src = "https://unpkg.com/@rdkit/rdkit/dist/RDKit_minimal.js";
        script.onload = () => {
            if ((window as any).initRDKitModule) {
                (window as any).initRDKitModule().then((RDKit: any) => {
                    (window as any).RDKit = RDKit;
                    rdkitLoaded = true;
                    resolve(RDKit);
                }).catch(reject);
            } else {
                reject(new Error("RDKit loaded loosely"));
            }
        };
        script.onerror = () => reject(new Error("Failed to load RDKit.js"));
        document.head.appendChild(script);
    });
    return rdkitPromise;
}

export function CompoundViewer3D({
    structure,
    formula,
    height = "360px",
    className = "",
    compoundName,
}: CompoundViewer3DProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [viewerState, setViewerState] = useState<"loading" | "ready" | "error">("loading");
    const [isPubChem, setIsPubChem] = useState(false);

    const safeFormula = formula || compoundName || "C";
    const uniqueElements = useMemo(() => {
        return parseFormulaToElements(safeFormula).map((e) => e.symbol);
    }, [safeFormula]);

    useEffect(() => {
        let mounted = true;
        
        async function render2D() {
            if (!containerRef.current) return;
            setViewerState("loading");

            try {
                await loadRDKit();
            } catch {
                console.error("Could not load RDKit.");
                if (mounted) setViewerState("error");
                return;
            }

            const RDKit = (window as any).RDKit;
            if (!RDKit || !mounted) return;
            
            let finalSDF = generateTargetedFallbackSDF(safeFormula);
            let fetchedPubChem = false;
            
            try {
                const resp = await fetch(`http://localhost:5000/api/molecule/pubchem-structure?formula=${encodeURIComponent(safeFormula)}`, { signal: AbortSignal.timeout(10000) });
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.sdf) {
                        finalSDF = data.sdf;
                        fetchedPubChem = true;
                    }
                }
            } catch (e) {
                // Ignore API failure, default to fallback topology
            }
            
            if (!mounted) return;
            setIsPubChem(fetchedPubChem);

            let mol = null;
            try {
                mol = RDKit.get_mol(finalSDF, '{"sanitize": false}');
                if (!mol) mol = RDKit.get_mol(finalSDF);
            } catch (e) {
                console.warn("RDKit generic parse error");
                mol = null;
            }

            if (mol) {
                const width = containerRef.current.clientWidth || 400;
                const h = containerRef.current.clientHeight || 300;
                let svg = mol.get_svg(width, h);
                
                // SVG post-processing for dark mode UI
                svg = svg.replace(/<rect([^>]*)fill=['"]#FFFFFF['"]([^>]*)>/g, "<rect$1opacity='0'$2>");
                svg = svg.replace(/<rect([^>]*)>/, "<rect$1 opacity='0'>");
                svg = svg.replace(/fill=['"]#000000['"]/g, "fill='#E2E8F0'");
                svg = svg.replace(/stroke=['"]#000000['"]/g, "stroke='#E2E8F0'");
                svg = svg.replace(/stroke-width:1.0/g, "stroke-width:2.0");

                containerRef.current.innerHTML = `<div style="display: flex; justify-content: center; align-items: center; width: 100%; height: 100%;">${svg}</div>`;
                mol.delete();
                setViewerState("ready");
            } else {
                setViewerState("error");
            }
        }

        render2D();

        return () => { mounted = false; };
    }, [safeFormula]);

    const lattice = structure?.lattice;

    return (
        <div
            className={`relative rounded-xl overflow-hidden ${className}`}
            style={{
                height,
                background: "linear-gradient(135deg, #0a0f1e 0%, #111827 40%, #1a1f2e 100%)",
                border: "1px solid rgba(179, 182, 255, 0.72)",
                boxShadow: "0 8px 32px rgba(0, 0, 0, 0.17), inset 0 1px 0 rgba(255, 255, 255, 0.46)",
            }}
        >
            <div ref={containerRef} style={{ width: "100%", height: "100%" }} />

            {viewerState === "loading" && (
                <div className="absolute inset-0 flex items-center justify-center">
                    <p className="text-xs text-muted-foreground animate-pulse">Generating 2D Topology...</p>
                </div>
            )}
            
            {viewerState === "error" && (
                <div className="absolute inset-0 flex items-center justify-center flex-col text-center px-4">
                    <p className="text-sm font-medium text-amber-500">Parsing Timeout</p>
                    <p className="text-xs text-amber-500/70 mt-1">SDF could not be resolved topologically.</p>
                </div>
            )}

            {/* Top Left Menu */}
            <div className="absolute top-2 left-2 flex items-center gap-1 bg-black/60 backdrop-blur-sm rounded-lg p-1 border border-white/10">
                <span className="px-2.5 py-1 text-[10px] font-bold text-white tracking-widest uppercase opacity-80">
                    2D Topology
                </span>
                <div className="w-px h-3 bg-white/20 mx-1"></div>
                <span className={`px-2 py-1 text-[9px] font-medium rounded-md ${isPubChem ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}`}>
                    {isPubChem ? "PubChem Direct" : "Estimated Variant"}
                </span>
            </div>

            {/* Element legend — top right */}
            {uniqueElements.length > 0 && (
                <div className="absolute top-2 right-2 flex gap-1.5 flex-wrap justify-end max-w-[200px]">
                    {uniqueElements.map((el) => (
                        <span key={el} className="flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[9px] font-bold bg-black/60 backdrop-blur-sm border border-white/10">
                            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: getColor(el) }} />
                            <span className="text-white/80">{el}</span>
                        </span>
                    ))}
                </div>
            )}

            {/* Bottom Left — Compound Info */}
            <div className="absolute bottom-2 left-2 px-3 py-1.5 rounded-lg bg-black/60 backdrop-blur-sm border border-white/10 flex items-center gap-3 pointer-events-none">
                <span className="text-[10px] text-cyan-400 font-mono font-bold">
                    {safeFormula}
                </span>
                {lattice?.crystal_system && (
                    <>
                        <span className="text-[10px] text-slate-500">•</span>
                        <span className="text-[10px] text-purple-400 font-medium capitalize">
                            {lattice.crystal_system}
                        </span>
                    </>
                )}
                {compoundName && (
                    <>
                        <span className="text-[10px] text-slate-500">•</span>
                        <span className="text-[10px] text-emerald-400 font-medium">
                            {compoundName}
                        </span>
                    </>
                )}
            </div>
        </div>
    );
}

export default CompoundViewer3D;
