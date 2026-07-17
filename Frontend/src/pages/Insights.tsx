import { Navbar } from "@/components/layout/Navbar";
import { AiInsightsPanel } from "@/components/insights/AiInsightsPanel";

const Insights = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="ml-60 p-6">
        <AiInsightsPanel />
      </main>
    </div>
  );
};

export default Insights;