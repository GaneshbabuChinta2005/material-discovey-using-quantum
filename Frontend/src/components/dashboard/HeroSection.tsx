import { Button } from "@/components/ui/button";
import { ArrowRight, Play } from "lucide-react";
import { Link } from "react-router-dom";

export function HeroSection() {
  return (
    <section className="relative overflow-hidden rounded-3xl p-6 md:p-8 lg:p-10 glass-card border border-white/50 dark:border-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.04)]">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-[0.03] dark:opacity-[0.05]">
        <svg className="h-full w-full" viewBox="0 0 800 600" fill="none">
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>
      
      {/* Quantum Orb Effect */}
      <div className="absolute top-0 right-0 -translate-y-1/4 translate-x-1/4 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-accent/20 to-quantum/10 blur-[80px] pointer-events-none" />
      <div className="absolute bottom-0 left-0 translate-y-1/4 -translate-x-1/4 w-[400px] h-[400px] rounded-full bg-gradient-to-tr from-quantum/20 to-transparent blur-[60px] pointer-events-none" />
      
      <div className="relative z-10 flex flex-col lg:flex-row items-center gap-8">
        {/* Text Content */}
        <div className="flex-1 space-y-6 text-foreground">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-accent/10 border border-accent/20 text-accent text-xs font-semibold uppercase tracking-wider shadow-[0_0_15px_rgba(var(--accent),0.1)]">
            <span className="w-2 h-2 rounded-full bg-quantum animate-pulse" />
            Next-Gen Materials Science
          </div>
          
          <h1 className="text-4xl md:text-5xl lg:text-5xl font-extrabold leading-tight tracking-tight text-balance text-foreground">
            Accelerating Materials Discovery with{" "}
            <span className="bg-gradient-to-r from-accent to-quantum bg-clip-text text-transparent drop-shadow-sm">
              AI & Quantum Computing
            </span>
          </h1>
          
          <p className="text-base md:text-lg text-muted-foreground max-w-2xl leading-relaxed">
            Simulate, predict, and optimize advanced materials at quantum scale. 
            Harness the power of artificial intelligence and quantum mechanics for breakthrough discoveries.
          </p>
          
          <div className="flex flex-wrap gap-4 pt-6">
            <Link to="/simulator">
              <div className="relative group">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-accent to-quantum rounded-full blur opacity-40 group-hover:opacity-70 transition duration-500" />
                <Button className="relative rounded-full px-6 py-4 text-sm font-bold text-white dark:text-black bg-slate-900 dark:bg-white hover:bg-slate-800 dark:hover:bg-slate-100 border border-slate-700/50 dark:border-white/20 overflow-hidden">
                  <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 dark:via-black/10 to-transparent -translate-x-[150%] group-hover:translate-x-[150%] transition-transform duration-700 ease-in-out" />
                  <span className="relative z-10 flex items-center gap-2">
                    Start Simulation
                    <ArrowRight className="h-5 w-5" />
                  </span>
                </Button>
              </div>
            </Link>
            <Link to="/materials">
              <Button variant="outline" className="rounded-full px-6 py-4 text-sm font-bold bg-background/50 backdrop-blur border-border hover:bg-accent/5 hover:text-accent transition-all duration-300">
                <Play className="h-5 w-5 mr-2" />
                Explore Materials
              </Button>
            </Link>
          </div>
        </div>
        
        {/* Illustration */}
        <div className="flex-1 relative hidden md:block">
          <div className="relative w-full max-w-sm mx-auto">
            {/* Orbital rings */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-56 h-56 border border-accent/30 rounded-full animate-spin" style={{ animationDuration: '20s' }} />
              <div className="absolute w-40 h-40 border border-quantum/40 rounded-full animate-spin" style={{ animationDuration: '15s', animationDirection: 'reverse' }} />
              <div className="absolute w-28 h-28 border border-accent/50 rounded-full animate-spin" style={{ animationDuration: '10s' }} />
            </div>
            
            {/* Central atom */}
            <div className="relative w-56 h-56 mx-auto flex items-center justify-center">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-accent to-quantum shadow-quantum animate-pulse-slow" />
              
              {/* Electrons */}
              <div className="absolute w-full h-full animate-spin" style={{ animationDuration: '8s' }}>
                <div className="absolute top-0 left-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-accent shadow-lg" />
              </div>
              <div className="absolute w-full h-full animate-spin" style={{ animationDuration: '6s', animationDirection: 'reverse' }}>
                <div className="absolute bottom-4 right-4 w-3 h-3 rounded-full bg-quantum shadow-lg" />
              </div>
              <div className="absolute w-full h-full animate-spin" style={{ animationDuration: '12s' }}>
                <div className="absolute top-1/2 right-0 w-3.5 h-3.5 rounded-full bg-accent/80 shadow-lg" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
