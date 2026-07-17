import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { 
  LayoutDashboard, 
  Atom, 
  Cpu, 
  Brain, 
  FlaskConical,
  Moon,
  Sun,
  User,
  Hexagon
} from "lucide-react";
import { useState, useEffect } from "react";

const navItems = [
  { name: "Dashboard", path: "/", icon: LayoutDashboard },
  { name: "Materials Explorer", path: "/materials", icon: Atom },
  { name: "Candidate Generator", path: "/candidate-generator", icon: Brain },
  { name: "Molecule Builder", path: "/molecule-builder", icon: Hexagon },
  { name: "Quantum Simulator", path: "/simulator", icon: Cpu },
  { name: "Experiments", path: "/experiments", icon: FlaskConical },
];

export function Navbar() {
  const location = useLocation();
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const isDarkMode = document.documentElement.classList.contains('dark');
    setIsDark(isDarkMode);
  }, []);

  const toggleTheme = () => {
    setIsDark(!isDark);
    document.documentElement.classList.toggle('dark');
  };

  return (
    <aside className="fixed left-0 top-0 h-screen w-60 backdrop-blur-xl bg-white/60 dark:bg-slate-900/40 border-r border-white/20 dark:border-white/10 rounded-r-3xl px-4 py-6 space-y-6 flex flex-col z-50 transition-all duration-300 shadow-lg">
      {/* Logo */}
      <Link to="/" className="flex items-center gap-3 px-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg">
          <img src="/pure_logo.png" alt="Logo" className="w-full h-full object-contain" />
        </div>
        <span className="text-lg font-semibold tracking-tight">QDiscover</span>
      </Link>

      {/* Navigation Links */}
      <div className="flex flex-col gap-2 flex-1 pt-4">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={cn(
                "flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 hover:scale-[1.02]",
                isActive
                  ? "bg-black text-white dark:bg-white dark:text-black shadow-md"
                  : "text-muted-foreground hover:bg-white/40 dark:hover:bg-white/10"
              )}
            >
              <item.icon className="h-4 w-4" />
              {item.name}
            </Link>
          );
        })}
      </div>

      {/* Bottom Section */}
      <div className="flex items-center justify-between gap-2 px-2 pt-4 border-t border-border/40">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-full bg-gradient-to-br from-accent/20 to-quantum/20 flex items-center justify-center">
            <User className="h-4 w-4 text-accent" />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold">Dr. Smith</span>
            <span className="text-[10px] text-muted-foreground">Researcher</span>
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleTheme}
          className="rounded-full hover:bg-white/40 dark:hover:bg-white/10 transition-all"
        >
          {isDark ? (
            <Sun className="h-4 w-4" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
        </Button>
      </div>
    </aside>
  );
}
