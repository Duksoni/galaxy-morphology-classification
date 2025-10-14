import gc
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import ttkbootstrap as ttk_boot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.metrics import classification_report, confusion_matrix

from src.constants import CLASS_NAMES, CLASS_NAMES_BRIEF


class ResultsViewer:
    """UI for viewing training and evaluation results."""
    
    def __init__(self, root: ttk_boot.Window):
        self.root = root
        self.window_width = 1400
        self.window_height = 900

        root.title("Galaxy Classifier - Results Viewer")
        root.minsize(self.window_width, self.window_height)
        root.resizable(False, False)
        root.geometry(f"{self.window_width}x{self.window_height}")
        self._center_window()

        # Loaded results
        self.training_history = None
        self.y_true = None
        self.y_pred = None
        self.test_accuracy = None
        self.test_loss = None
        self.model_name = None

        self._setup_ui()

    def _center_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        position_top = int((screen_height - self.window_height) / 2)
        position_left = int((screen_width - self.window_width) / 2)
        self.root.geometry(f'{self.window_width}x{self.window_height}+{position_left}+{position_top}')

    def _setup_ui(self):
        # Main frame with scrollbar
        main_canvas = tk.Canvas(self.root, bg='#222222', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)

        scrollable_frame = ttk.Frame(main_canvas, padding="20")
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )

        canvas_window = main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        def _resize_canvas(event):
            main_canvas.itemconfig(canvas_window, width=event.width)

        main_canvas.bind("<Configure>", _resize_canvas)

        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            delta = event.delta
            if delta == 0:
                return
            main_canvas.yview_scroll(int(-1 * (delta / 120)), "units")

        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        main_canvas.bind_all("<Button-4>", lambda e: main_canvas.yview_scroll(-3, "units"))
        main_canvas.bind_all("<Button-5>", lambda e: main_canvas.yview_scroll(3, "units"))

        main_frame = scrollable_frame
        main_frame.columnconfigure(0, weight=1)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        header_frame.columnconfigure(0, weight=1)

        title_label = ttk.Label(header_frame, text="Galaxy Classifier - Results Viewer",
                               font=("Helvetica", 18, "bold"))
        title_label.grid(row=0, column=0, sticky=tk.W)

        self.model_label = ttk.Label(header_frame, text="Model: None",
                                     font=("Helvetica", 10))
        self.model_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))

        # Load Results Section
        load_frame = ttk.LabelFrame(main_frame, text="Load Results", padding=10)
        load_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 20))
        load_frame.columnconfigure(0, weight=1)

        button_subframe = ttk.Frame(load_frame)
        button_subframe.grid(row=0, column=0, sticky=tk.W)

        load_btn = ttk_boot.Button(button_subframe, text="Load Results from File",
                                   command=self._load_results,
                                   bootstyle="info", width=20)
        load_btn.grid(row=0, column=0, padx=5)

        clear_btn = ttk_boot.Button(button_subframe, text="Clear All",
                                    command=self._clear_all,
                                    bootstyle="warning", width=20)
        clear_btn.grid(row=0, column=1, padx=5)

        row = 2

        # Training history container
        self.history_fig = None
        self.history_canvas = None
        history_container = ttk.LabelFrame(main_frame, text="Training History")
        history_container.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        history_container.columnconfigure(0, weight=1)
        history_container.rowconfigure(0, weight=1)
        self.history_container = history_container

        row += 1

        # Bottom section: confusion matrix (left) and metrics/report (right)
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=2)

        # Left: Confusion Matrix
        self.cm_fig = None
        self.cm_canvas = None
        cm_container = ttk.LabelFrame(bottom_frame, text="Confusion Matrix")
        cm_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        cm_container.columnconfigure(0, weight=1)
        cm_container.rowconfigure(0, weight=1)
        self.cm_container = cm_container

        # Right: Metrics + Report
        right_column = ttk.Frame(bottom_frame)
        right_column.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_column.columnconfigure(0, weight=1)
        right_column.rowconfigure(0, weight=0)
        right_column.rowconfigure(1, weight=1)

        # Test Metrics
        metrics_container = ttk.LabelFrame(right_column, text="Test Metrics")
        metrics_container.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        metrics_container.columnconfigure(0, weight=1)

        metrics_frame = ttk.Frame(metrics_container)
        metrics_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)

        ttk.Label(metrics_frame, text="Test Accuracy:", font=("Helvetica", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.test_acc_label = ttk.Label(metrics_frame, text="N/A", font=("Helvetica", 10))
        self.test_acc_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 40))

        ttk.Label(metrics_frame, text="Test Loss:", font=("Helvetica", 10, "bold")).grid(
            row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.test_loss_label = ttk.Label(metrics_frame, text="N/A", font=("Helvetica", 10))
        self.test_loss_label.grid(row=0, column=3, sticky=tk.W)

        # Classification Report
        report_frame = ttk.LabelFrame(right_column, text="Classification Report")
        report_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        report_frame.columnconfigure(0, weight=1)
        report_frame.rowconfigure(0, weight=1)

        self.report_text = tk.Text(report_frame, height=12, wrap='none', font=('Courier', 9))
        self.report_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        report_scrollbar_y = ttk.Scrollbar(report_frame, orient="vertical", command=self.report_text.yview)
        report_scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.report_text.configure(yscrollcommand=report_scrollbar_y.set)

        report_scrollbar_x = ttk.Scrollbar(report_frame, orient="horizontal", command=self.report_text.xview)
        report_scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.report_text.configure(xscrollcommand=report_scrollbar_x.set)

        self.report_text.configure(state='disabled')

        # Hide results by default
        self._set_results_visibility(visible=False)

    def _set_results_visibility(self, visible: bool):
        """Show/hide results widgets."""
        try:
            if not visible:
                self.history_container.grid_remove()
                for child in self.history_container.master.winfo_children():
                    if isinstance(child, ttk.Frame) and child is not self.history_container:
                        for gchild in child.winfo_children():
                            if gchild is getattr(self, "cm_container", None):
                                child.grid_remove()
                                break
            else:
                self.history_container.grid()
                for child in self.history_container.master.winfo_children():
                    if isinstance(child, ttk.Frame) and child is not self.history_container:
                        for gchild in child.winfo_children():
                            if gchild is getattr(self, "cm_container", None):
                                child.grid()
                                break
        except Exception:
            pass

    def _load_results(self):
        """Load results from saved pickle files."""
        file_path = filedialog.askopenfilename(
            title="Load results.pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)

            self.training_history = results.get('history')
            self.y_true = results.get('y_true')
            self.y_pred = results.get('y_pred')
            self.test_accuracy = results.get('test_accuracy')
            self.test_loss = results.get('test_loss')
            self.model_name = results.get('model_name', 'Unknown')

            self.model_label.config(text=f"Model: {self.model_name}")
            self._display_results()
            self._set_results_visibility(visible=True)

            messagebox.showinfo("Success", "Results loaded successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load results: {str(e)}")

    def _display_results(self):
        """Display loaded results in the UI."""
        try:
            # Clear previous plots
            self._clear_canvases()

            # Update metrics
            if self.test_accuracy is not None:
                self.test_acc_label.config(
                    text=f"{self.test_accuracy:.4f} ({self.test_accuracy * 100:.2f}%)"
                )
            if self.test_loss is not None:
                self.test_loss_label.config(text=f"{self.test_loss:.4f}")

            # Plot training history
            if self.training_history is not None:
                self._plot_training_history()

            # Plot confusion matrix
            if self.y_true is not None and self.y_pred is not None:
                self._plot_confusion_matrix()
                self._display_classification_report()

        except Exception as e:
            messagebox.showerror("Display Error", f"Could not display results: {str(e)}")

    def _plot_training_history(self):
        """Plot training history from loaded data."""
        try:
            hist = self.training_history
            acc = hist.get('accuracy') or []
            val_acc = hist.get('val_accuracy') or []
            loss = hist.get('loss') or []
            val_loss = hist.get('val_loss') or []

            if not acc and not loss:
                return

            epochs = range(1, len(acc) + 1 if len(acc) > 0 else len(loss) + 1)

            fig = Figure(figsize=(12, 5), dpi=100)
            
            ax1 = fig.add_subplot(1, 2, 1)
            if len(acc) > 0:
                ax1.plot(epochs, acc, 'b-', linewidth=2, label='Train Acc')
            if len(val_acc) > 0:
                ax1.plot(epochs, val_acc, 'r-', linewidth=2, label='Val Acc')
            ax1.set_title(f'{self.model_name} - Accuracy', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Epoch', fontsize=10)
            ax1.set_ylabel('Accuracy', fontsize=10)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = fig.add_subplot(1, 2, 2)
            if len(loss) > 0:
                ax2.plot(epochs, loss, 'b-', linewidth=2, label='Train Loss')
            if len(val_loss) > 0:
                ax2.plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss')
            ax2.set_title(f'{self.model_name} - Loss', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=10)
            ax2.set_ylabel('Loss', fontsize=10)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            self.history_fig = fig
            self.history_canvas = FigureCanvasTkAgg(fig, master=self.history_container)
            self.history_canvas.draw()
            self.history_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        except Exception as e:
            messagebox.showerror("Plot Error", f"Could not plot training history: {str(e)}")

    def _plot_confusion_matrix(self):
        """Plot confusion matrix from loaded data."""
        try:
            cm = confusion_matrix(self.y_true, self.y_pred)

            fig_cm = Figure(figsize=(6, 6), dpi=100)
            ax = fig_cm.add_subplot(111)
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            fig_cm.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=CLASS_NAMES_BRIEF,
                   yticklabels=CLASS_NAMES)

            ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel('True label', fontsize=10)
            ax.set_xlabel('Predicted label', fontsize=10)

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor", fontsize=8)
            plt.setp(ax.get_yticklabels(), fontsize=8)

            fmt = 'd'
            thresh = cm.max() / 2. if cm.size else 0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(int(cm[i, j]), fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=7)

            fig_cm.tight_layout()
            self.cm_fig = fig_cm
            self.cm_canvas = FigureCanvasTkAgg(fig_cm, master=self.cm_container)
            self.cm_canvas.draw()
            self.cm_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        except Exception as e:
            messagebox.showerror("Plot Error", f"Could not plot confusion matrix: {str(e)}")

    def _display_classification_report(self):
        """Display classification report from loaded data."""
        try:
            self.report_text.configure(state='normal')
            self.report_text.delete('1.0', tk.END)

            if self.y_true is not None and self.y_pred is not None:
                report = classification_report(
                    self.y_true,
                    self.y_pred,
                    target_names=CLASS_NAMES,
                    digits=4
                )
                self.report_text.insert(tk.END, report)
            else:
                self.report_text.insert(tk.END, "No classification data available.")

            self.report_text.configure(state='disabled')

        except Exception as e:
            messagebox.showerror("Report Error", f"Could not display report: {str(e)}")

    def _clear_canvases(self):
        """Clear matplotlib figures."""
        try:
            if self.history_canvas is not None:
                self.history_canvas.get_tk_widget().destroy()
                self.history_canvas = None
            if self.history_fig is not None:
                plt.close(self.history_fig)
                self.history_fig = None

            if self.cm_canvas is not None:
                self.cm_canvas.get_tk_widget().destroy()
                self.cm_canvas = None
            if self.cm_fig is not None:
                plt.close(self.cm_fig)
                self.cm_fig = None
        except Exception:
            pass

    def _clear_all(self):
        """Clear all displayed results."""
        try:
            self._clear_canvases()
            self.report_text.configure(state='normal')
            self.report_text.delete('1.0', tk.END)
            self.report_text.configure(state='disabled')

            self.test_acc_label.config(text="N/A")
            self.test_loss_label.config(text="N/A")
            self.model_label.config(text="Model: None")

            self.training_history = None
            self.y_true = None
            self.y_pred = None
            self.test_accuracy = None
            self.test_loss = None
            self.model_name = None

            self._set_results_visibility(visible=False)
            gc.collect()

        except Exception as e:
            messagebox.showerror("Clear Error", f"Could not clear results: {str(e)}")


def main():
    root = ttk_boot.Window(themename="darkly")
    app = ResultsViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()