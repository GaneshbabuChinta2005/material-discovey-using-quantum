import { cn } from "@/lib/utils";

interface Element {
  symbol: string;
  name: string;
  atomicNumber: number;
  atomicMass: number;
  category: string;
  row: number;
  col: number;
}

const elements: Element[] = [
  // Row 1
  { symbol: "H", name: "Hydrogen", atomicNumber: 1, atomicMass: 1.008, category: "nonmetal", row: 1, col: 1 },
  { symbol: "He", name: "Helium", atomicNumber: 2, atomicMass: 4.003, category: "noble-gas", row: 1, col: 18 },
  // Row 2
  { symbol: "Li", name: "Lithium", atomicNumber: 3, atomicMass: 6.94, category: "alkali", row: 2, col: 1 },
  { symbol: "Be", name: "Beryllium", atomicNumber: 4, atomicMass: 9.012, category: "alkaline", row: 2, col: 2 },
  { symbol: "B", name: "Boron", atomicNumber: 5, atomicMass: 10.81, category: "metalloid", row: 2, col: 13 },
  { symbol: "C", name: "Carbon", atomicNumber: 6, atomicMass: 12.01, category: "nonmetal", row: 2, col: 14 },
  { symbol: "N", name: "Nitrogen", atomicNumber: 7, atomicMass: 14.01, category: "nonmetal", row: 2, col: 15 },
  { symbol: "O", name: "Oxygen", atomicNumber: 8, atomicMass: 16.00, category: "nonmetal", row: 2, col: 16 },
  { symbol: "F", name: "Fluorine", atomicNumber: 9, atomicMass: 19.00, category: "halogen", row: 2, col: 17 },
  { symbol: "Ne", name: "Neon", atomicNumber: 10, atomicMass: 20.18, category: "noble-gas", row: 2, col: 18 },
  // Row 3
  { symbol: "Na", name: "Sodium", atomicNumber: 11, atomicMass: 22.99, category: "alkali", row: 3, col: 1 },
  { symbol: "Mg", name: "Magnesium", atomicNumber: 12, atomicMass: 24.31, category: "alkaline", row: 3, col: 2 },
  { symbol: "Al", name: "Aluminum", atomicNumber: 13, atomicMass: 26.98, category: "metal", row: 3, col: 13 },
  { symbol: "Si", name: "Silicon", atomicNumber: 14, atomicMass: 28.09, category: "metalloid", row: 3, col: 14 },
  { symbol: "P", name: "Phosphorus", atomicNumber: 15, atomicMass: 30.97, category: "nonmetal", row: 3, col: 15 },
  { symbol: "S", name: "Sulfur", atomicNumber: 16, atomicMass: 32.07, category: "nonmetal", row: 3, col: 16 },
  { symbol: "Cl", name: "Chlorine", atomicNumber: 17, atomicMass: 35.45, category: "halogen", row: 3, col: 17 },
  { symbol: "Ar", name: "Argon", atomicNumber: 18, atomicMass: 39.95, category: "noble-gas", row: 3, col: 18 },
  // Row 4
  { symbol: "K", name: "Potassium", atomicNumber: 19, atomicMass: 39.10, category: "alkali", row: 4, col: 1 },
  { symbol: "Ca", name: "Calcium", atomicNumber: 20, atomicMass: 40.08, category: "alkaline", row: 4, col: 2 },
  { symbol: "Sc", name: "Scandium", atomicNumber: 21, atomicMass: 44.96, category: "transition", row: 4, col: 3 },
  { symbol: "Ti", name: "Titanium", atomicNumber: 22, atomicMass: 47.87, category: "transition", row: 4, col: 4 },
  { symbol: "V", name: "Vanadium", atomicNumber: 23, atomicMass: 50.94, category: "transition", row: 4, col: 5 },
  { symbol: "Cr", name: "Chromium", atomicNumber: 24, atomicMass: 52.00, category: "transition", row: 4, col: 6 },
  { symbol: "Mn", name: "Manganese", atomicNumber: 25, atomicMass: 54.94, category: "transition", row: 4, col: 7 },
  { symbol: "Fe", name: "Iron", atomicNumber: 26, atomicMass: 55.85, category: "transition", row: 4, col: 8 },
  { symbol: "Co", name: "Cobalt", atomicNumber: 27, atomicMass: 58.93, category: "transition", row: 4, col: 9 },
  { symbol: "Ni", name: "Nickel", atomicNumber: 28, atomicMass: 58.69, category: "transition", row: 4, col: 10 },
  { symbol: "Cu", name: "Copper", atomicNumber: 29, atomicMass: 63.55, category: "transition", row: 4, col: 11 },
  { symbol: "Zn", name: "Zinc", atomicNumber: 30, atomicMass: 65.38, category: "transition", row: 4, col: 12 },
  { symbol: "Ga", name: "Gallium", atomicNumber: 31, atomicMass: 69.72, category: "metal", row: 4, col: 13 },
  { symbol: "Ge", name: "Germanium", atomicNumber: 32, atomicMass: 72.63, category: "metalloid", row: 4, col: 14 },
  { symbol: "As", name: "Arsenic", atomicNumber: 33, atomicMass: 74.92, category: "metalloid", row: 4, col: 15 },
  { symbol: "Se", name: "Selenium", atomicNumber: 34, atomicMass: 78.97, category: "nonmetal", row: 4, col: 16 },
  { symbol: "Br", name: "Bromine", atomicNumber: 35, atomicMass: 79.90, category: "halogen", row: 4, col: 17 },
  { symbol: "Kr", name: "Krypton", atomicNumber: 36, atomicMass: 83.80, category: "noble-gas", row: 4, col: 18 },
  // Row 5
  { symbol: "Rb", name: "Rubidium", atomicNumber: 37, atomicMass: 85.47, category: "alkali", row: 5, col: 1 },
  { symbol: "Sr", name: "Strontium", atomicNumber: 38, atomicMass: 87.62, category: "alkaline", row: 5, col: 2 },
  { symbol: "Y", name: "Yttrium", atomicNumber: 39, atomicMass: 88.91, category: "transition", row: 5, col: 3 },
  { symbol: "Zr", name: "Zirconium", atomicNumber: 40, atomicMass: 91.22, category: "transition", row: 5, col: 4 },
  { symbol: "Nb", name: "Niobium", atomicNumber: 41, atomicMass: 92.91, category: "transition", row: 5, col: 5 },
  { symbol: "Mo", name: "Molybdenum", atomicNumber: 42, atomicMass: 95.95, category: "transition", row: 5, col: 6 },
  { symbol: "Tc", name: "Technetium", atomicNumber: 43, atomicMass: 98, category: "transition", row: 5, col: 7 },
  { symbol: "Ru", name: "Ruthenium", atomicNumber: 44, atomicMass: 101.07, category: "transition", row: 5, col: 8 },
  { symbol: "Rh", name: "Rhodium", atomicNumber: 45, atomicMass: 102.91, category: "transition", row: 5, col: 9 },
  { symbol: "Pd", name: "Palladium", atomicNumber: 46, atomicMass: 106.42, category: "transition", row: 5, col: 10 },
  { symbol: "Ag", name: "Silver", atomicNumber: 47, atomicMass: 107.87, category: "transition", row: 5, col: 11 },
  { symbol: "Cd", name: "Cadmium", atomicNumber: 48, atomicMass: 112.41, category: "transition", row: 5, col: 12 },
  { symbol: "In", name: "Indium", atomicNumber: 49, atomicMass: 114.82, category: "metal", row: 5, col: 13 },
  { symbol: "Sn", name: "Tin", atomicNumber: 50, atomicMass: 118.71, category: "metal", row: 5, col: 14 },
  { symbol: "Sb", name: "Antimony", atomicNumber: 51, atomicMass: 121.76, category: "metalloid", row: 5, col: 15 },
  { symbol: "Te", name: "Tellurium", atomicNumber: 52, atomicMass: 127.60, category: "metalloid", row: 5, col: 16 },
  { symbol: "I", name: "Iodine", atomicNumber: 53, atomicMass: 126.90, category: "halogen", row: 5, col: 17 },
  { symbol: "Xe", name: "Xenon", atomicNumber: 54, atomicMass: 131.29, category: "noble-gas", row: 5, col: 18 },
  // Row 6
  { symbol: "Cs", name: "Cesium", atomicNumber: 55, atomicMass: 132.91, category: "alkali", row: 6, col: 1 },
  { symbol: "Ba", name: "Barium", atomicNumber: 56, atomicMass: 137.33, category: "alkaline", row: 6, col: 2 },
  { symbol: "Lu", name: "Lutetium", atomicNumber: 71, atomicMass: 174.97, category: "transition", row: 6, col: 3 },
  { symbol: "Hf", name: "Hafnium", atomicNumber: 72, atomicMass: 178.49, category: "transition", row: 6, col: 4 },
  { symbol: "Ta", name: "Tantalum", atomicNumber: 73, atomicMass: 180.95, category: "transition", row: 6, col: 5 },
  { symbol: "W", name: "Tungsten", atomicNumber: 74, atomicMass: 183.84, category: "transition", row: 6, col: 6 },
  { symbol: "Re", name: "Rhenium", atomicNumber: 75, atomicMass: 186.21, category: "transition", row: 6, col: 7 },
  { symbol: "Os", name: "Osmium", atomicNumber: 76, atomicMass: 190.23, category: "transition", row: 6, col: 8 },
  { symbol: "Ir", name: "Iridium", atomicNumber: 77, atomicMass: 192.22, category: "transition", row: 6, col: 9 },
  { symbol: "Pt", name: "Platinum", atomicNumber: 78, atomicMass: 195.08, category: "transition", row: 6, col: 10 },
  { symbol: "Au", name: "Gold", atomicNumber: 79, atomicMass: 196.97, category: "transition", row: 6, col: 11 },
  { symbol: "Hg", name: "Mercury", atomicNumber: 80, atomicMass: 200.59, category: "transition", row: 6, col: 12 },
  { symbol: "Tl", name: "Thallium", atomicNumber: 81, atomicMass: 204.38, category: "metal", row: 6, col: 13 },
  { symbol: "Pb", name: "Lead", atomicNumber: 82, atomicMass: 207.2, category: "metal", row: 6, col: 14 },
  { symbol: "Bi", name: "Bismuth", atomicNumber: 83, atomicMass: 208.98, category: "metal", row: 6, col: 15 },
  { symbol: "Po", name: "Polonium", atomicNumber: 84, atomicMass: 209, category: "metalloid", row: 6, col: 16 },
  { symbol: "At", name: "Astatine", atomicNumber: 85, atomicMass: 210, category: "halogen", row: 6, col: 17 },
  { symbol: "Rn", name: "Radon", atomicNumber: 86, atomicMass: 222, category: "noble-gas", row: 6, col: 18 },
];

const categoryColors: Record<string, string> = {
  "nonmetal": "bg-emerald-500/20 hover:bg-emerald-500/30 border-emerald-500/40 text-emerald-700 dark:text-emerald-300",
  "noble-gas": "bg-purple-500/20 hover:bg-purple-500/30 border-purple-500/40 text-purple-700 dark:text-purple-300",
  "alkali": "bg-red-500/20 hover:bg-red-500/30 border-red-500/40 text-red-700 dark:text-red-300",
  "alkaline": "bg-orange-500/20 hover:bg-orange-500/30 border-orange-500/40 text-orange-700 dark:text-orange-300",
  "metalloid": "bg-cyan-500/20 hover:bg-cyan-500/30 border-cyan-500/40 text-cyan-700 dark:text-cyan-300",
  "halogen": "bg-yellow-500/20 hover:bg-yellow-500/30 border-yellow-500/40 text-yellow-700 dark:text-yellow-300",
  "metal": "bg-blue-500/20 hover:bg-blue-500/30 border-blue-500/40 text-blue-700 dark:text-blue-300",
  "transition": "bg-pink-500/20 hover:bg-pink-500/30 border-pink-500/40 text-pink-700 dark:text-pink-300",
};

interface PeriodicTableProps {
  onElementSelect: (element: { symbol: string; name: string; atomicMass: number }) => void;
}

export function PeriodicTable({ onElementSelect }: PeriodicTableProps) {
  const handleDragStart = (e: React.DragEvent, element: Element) => {
    e.dataTransfer.setData("application/json", JSON.stringify({
      symbol: element.symbol,
      name: element.name,
      atomicMass: element.atomicMass,
    }));
  };

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Periodic Table</h2>
      <p className="text-sm text-muted-foreground">Drag elements to build your molecule</p>
      
      <div 
        className="grid gap-1" 
        style={{ gridTemplateColumns: "repeat(18, minmax(0, 1fr))" }}
      >
        {elements.map((element) => (
          <div
            key={element.symbol}
            draggable
            onDragStart={(e) => handleDragStart(e, element)}
            onClick={() => onElementSelect({ symbol: element.symbol, name: element.name, atomicMass: element.atomicMass })}
            style={{ gridColumnStart: element.col, gridRowStart: element.row }}
            className={cn(
              "w-full aspect-square rounded-md border cursor-grab active:cursor-grabbing transition-all flex flex-col items-center justify-center p-0.5",
              categoryColors[element.category] || "bg-muted hover:bg-muted/80"
            )}
            title={element.name}
          >
            <span className="text-[7px] md:text-[9px] opacity-60 leading-none">{element.atomicNumber}</span>
            <span className="text-xs md:text-sm font-bold leading-none">{element.symbol}</span>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-2 pt-2 border-t border-border">
        <span className="text-xs text-muted-foreground">Legend:</span>
        {Object.entries(categoryColors).slice(0, 4).map(([cat, color]) => (
          <div key={cat} className="flex items-center gap-1">
            <div className={cn("w-3 h-3 rounded", color.split(" ")[0])} />
            <span className="text-xs text-muted-foreground capitalize">{cat.replace("-", " ")}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
