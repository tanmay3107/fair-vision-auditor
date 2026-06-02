# report_generator.py
import matplotlib.pyplot as plt
import numpy as np

class AuditReportGenerator:
    def __init__(self, base_metrics, hybrid_metrics, output_filename="fairness_audit_report.pdf"):
        """
        Initializes the reporting engine.
        :param base_metrics: Dictionary of fairness metrics for the baseline model.
        :param hybrid_metrics: Dictionary of fairness metrics for the hybrid model.
        """
        self.base_metrics = base_metrics
        self.hybrid_metrics = hybrid_metrics
        self.output_filename = output_filename
        self.metrics_names = ["Demographic Parity (DPD)", "Disparate Impact (DI)", "Equal Opp (EOD)"]
        
    def generate_pdf_report(self):
        """
        Generates a grouped bar chart comparing the models and exports it to PDF.
        """
        # Extract values
        base_vals = [
            self.base_metrics["DPD"], 
            self.base_metrics["DI"], 
            self.base_metrics["EOD"]
        ]
        hybrid_vals = [
            self.hybrid_metrics["DPD"], 
            self.hybrid_metrics["DI"], 
            self.hybrid_metrics["EOD"]
        ]

        x = np.arange(len(self.metrics_names))
        width = 0.35  

        # Setup the matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the bars
        rects1 = ax.bar(x - width/2, base_vals, width, label='Baseline Model', color='#ef4444')
        rects2 = ax.bar(x + width/2, hybrid_vals, width, label='Hybrid Model (Ours)', color='#22c55e')

        # Add zero-lines and ideal thresholds
        ax.axhline(0, color='black', linewidth=1)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label="Ideal DI (1.0)")
        
        # Highlight the 80% rule safe zone for Disparate Impact
        ax.axhspan(0.8, 1.25, xmin=0.33, xmax=0.66, color='green', alpha=0.1, label="DI Safe Zone (0.8 - 1.25)")

        # Format the axes and labels
        ax.set_ylabel('Metric Value')
        ax.set_title('Algorithmic Fairness Audit: Baseline vs. Hybrid Mitigation', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.metrics_names, fontsize=11)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Add data labels directly on top of the bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                # Determine text placement based on whether the bar goes up or down
                xytext = (0, 3) if height >= 0 else (0, -12)
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=xytext,
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()
        
        # Save straight to PDF
        plt.savefig(self.output_filename, format='pdf', bbox_inches='tight')
        print(f"✅ Visual audit report successfully generated and saved to: {self.output_filename}")


# --- Quick Test ---
if __name__ == "__main__":
    # Simulate the delta we achieved at the end of Day 4
    # Baseline failed DI (0.65) and had a large EOD gap (-0.30)
    simulated_base_metrics = {"DPD": -0.25, "DI": 0.65, "EOD": -0.30}
    
    # Hybrid improved all metrics, bringing DI into the safe zone (0.85)
    simulated_hybrid_metrics = {"DPD": -0.10, "DI": 0.85, "EOD": -0.05}
    
    print("📊 Compiling Fairness Audit Visualizations...")
    report_gen = AuditReportGenerator(simulated_base_metrics, simulated_hybrid_metrics)
    report_gen.generate_pdf_report()