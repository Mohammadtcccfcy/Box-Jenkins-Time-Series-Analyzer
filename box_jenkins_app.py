import warnings
warnings.filterwarnings("ignore")

import os
import textwrap
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class ImportWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Box-Jenkins Import")
        self.root.geometry("520x240")
        self.root.resizable(False, False)

        self.file_path = None
        self.data = None

        self.build_ui()

    def build_ui(self):
        title = tk.Label(
            self.root,
            text="Box-Jenkins Time Series Analyzer",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=20)

        subtitle = tk.Label(
            self.root,
            text="Import an Excel or CSV file to start the analysis",
            font=("Arial", 10)
        )
        subtitle.pack(pady=5)

        import_btn = tk.Button(
            self.root,
            text="Import Data File",
            width=22,
            height=2,
            command=self.import_file
        )
        import_btn.pack(pady=20)

        note = tk.Label(
            self.root,
            text="Required format: first column = Year/Date, second column = Value",
            font=("Arial", 9),
            fg="gray"
        )
        note.pack(pady=5)

        note2 = tk.Label(
            self.root,
            text="This version estimates non-seasonal ARIMA models only.",
            font=("Arial", 9),
            fg="gray"
        )
        note2.pack(pady=2)

    def import_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("Excel and CSV files", "*.xlsx *.xls *.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv")
            ]
        )

        if not file_path:
            return

        try:
            df = self.read_data_file(file_path)
            series = self.prepare_series(df)

            self.file_path = file_path
            self.data = series

            self.root.destroy()

            main_root = tk.Tk()
            app = AnalysisWindow(main_root, self.data, self.file_path)
            main_root.mainloop()

        except Exception as e:
            messagebox.showerror("Import Error", str(e))

    def read_data_file(self, file_path):
        lower_path = file_path.lower()

        if lower_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif lower_path.endswith(".xlsx") or lower_path.endswith(".xls"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format.")

        return df

    def prepare_series(self, df):
        if df.shape[1] < 2:
            raise ValueError(
                "The file must contain at least two columns: Year/Date and Value."
            )

        clean_df = pd.DataFrame({
            "date": df.iloc[:, 0],
            "value": df.iloc[:, 1]
        })

        clean_df = clean_df.dropna()
        clean_df["value"] = pd.to_numeric(clean_df["value"], errors="coerce")
        clean_df = clean_df.dropna()

        if len(clean_df) < 20:
            raise ValueError(
                "At least 20 valid observations are recommended for reliable ARIMA analysis."
            )

        try:
            clean_df["date_numeric"] = pd.to_numeric(clean_df["date"], errors="coerce")
            if clean_df["date_numeric"].notna().all():
                clean_df = clean_df.sort_values("date_numeric")
                index = clean_df["date_numeric"].astype(int).astype(str)
            else:
                clean_df = clean_df.sort_values("date")
                index = clean_df["date"].astype(str)
        except Exception:
            index = clean_df["date"].astype(str)

        series = pd.Series(clean_df["value"].values, index=index)
        series.name = "Value"

        if series.nunique() <= 1:
            raise ValueError("The series is constant and cannot be modeled by ARIMA.")

        return series


class AnalysisWindow:
    def __init__(self, root, series, file_path):
        self.root = root
        self.series = series.dropna()
        self.file_path = file_path

        self.pages = []
        self.page_titles = []
        self.current_page = 0

        self.d = 0
        self.stationary_series = None
        self.model = None
        self.best_order = None
        self.forecast_steps = 5
        self.model_candidates = []

        # برای خروجی PDF
        self.pdf_items = []

        self.root.title("Box-Jenkins Time Series Analyzer")
        self.root.geometry("1150x800")
        self.root.minsize(930, 680)

        self.build_layout()
        self.run_analysis()
        self.show_page(0)

    def build_layout(self):
        top_frame = tk.Frame(self.root, height=80)
        top_frame.pack(side="top", fill="x")

        left_top_frame = tk.Frame(top_frame)
        left_top_frame.pack(side="left", padx=15, pady=5)

        title_label = tk.Label(
            left_top_frame,
            text="Box-Jenkins Time Series Analyzer",
            font=("Arial", 14, "bold")
        )
        title_label.pack(anchor="w")

        export_pdf_btn = tk.Button(
            left_top_frame,
            text="Export PDF",
            width=16,
            command=self.export_pdf
        )
        export_pdf_btn.pack(anchor="w", pady=5)

        file_label = tk.Label(
            top_frame,
            text=f"File: {os.path.basename(self.file_path)}",
            font=("Arial", 9),
            fg="gray"
        )
        file_label.pack(side="left", padx=15, pady=15)

        exit_btn = tk.Button(
            top_frame,
            text="Exit",
            width=10,
            command=self.root.destroy
        )
        exit_btn.pack(side="right", padx=15, pady=10)

        self.content_frame = tk.Frame(self.root)
        self.content_frame.pack(side="top", fill="both", expand=True)

        nav_frame = tk.Frame(self.root, height=50)
        nav_frame.pack(side="bottom", fill="x")

        self.prev_btn = tk.Button(
            nav_frame,
            text="Previous",
            width=15,
            command=self.previous_page
        )
        self.prev_btn.pack(side="left", padx=25, pady=10)

        self.page_indicator = tk.Label(
            nav_frame,
            text="",
            font=("Arial", 11, "bold")
        )
        self.page_indicator.pack(side="left", expand=True)

        self.next_btn = tk.Button(
            nav_frame,
            text="Next",
            width=15,
            command=self.next_page
        )
        self.next_btn.pack(side="right", padx=25, pady=10)

    def run_analysis(self):
        y = self.series.copy().dropna()

        self.add_plot_page(
            title="Original Time Series",
            plot_function=lambda fig, ax: self.plot_original_series(fig, ax)
        )

        level_adf_c = self.safe_adf(y, regression="c")
        level_adf_ct = self.safe_adf(y, regression="ct")
        level_kpss_c = self.safe_kpss(y, regression="c")
        level_kpss_ct = self.safe_kpss(y, regression="ct")

        self.add_text_page(
            title="Stationarity Tests - Level Series",
            content=self.format_level_stationarity_report(
                level_adf_c, level_adf_ct, level_kpss_c, level_kpss_ct
            )
        )

        self.d, self.stationary_series = self.determine_d(y)

        diff_adf = self.safe_adf(self.stationary_series, regression="c")
        diff_kpss = self.safe_kpss(self.stationary_series, regression="c")

        self.add_text_page(
            title="Stationarity and Differencing",
            content=self.format_differencing_result(diff_adf, diff_kpss)
        )

        self.add_plot_page(
            title="ACF and PACF",
            plot_function=lambda fig, ax: self.plot_acf_pacf(fig)
        )

        self.best_order, self.model = self.select_best_arima(y, self.d)

        self.add_text_page(
            title="Estimated ARIMA Model",
            content=self.format_model_summary()
        )

        self.add_plot_page(
            title="Residual Diagnostics",
            plot_function=lambda fig, ax: self.plot_residuals(fig)
        )

        self.add_text_page(
            title="Residual Tests",
            content=self.format_residual_tests()
        )

        self.add_plot_page(
            title="Forecast",
            plot_function=lambda fig, ax: self.plot_forecast(fig, ax)
        )

    def safe_adf(self, series, regression="c"):
        series = pd.Series(series).dropna()

        if len(series) < 8:
            return None

        if series.nunique() <= 1:
            return None

        try:
            return adfuller(series, regression=regression, autolag="AIC")
        except Exception:
            return None

    def safe_kpss(self, series, regression="c"):
        series = pd.Series(series).dropna()

        if len(series) < 8:
            return None

        if series.nunique() <= 1:
            return None

        try:
            return kpss(series, regression=regression, nlags="auto")
        except Exception:
            return None

    def determine_d(self, y):
        current = y.copy().dropna()

        for d in range(3):
            adf_result = self.safe_adf(current, regression="c")
            kpss_result = self.safe_kpss(current, regression="c")

            adf_stationary = (
                adf_result is not None and adf_result[1] <= 0.05
            )
            kpss_stationary = (
                kpss_result is not None and kpss_result[1] > 0.05
            )

            if adf_stationary and kpss_stationary:
                return d, current

            if adf_stationary and kpss_result is None:
                return d, current

            if d < 2:
                current = current.diff().dropna()

        return 2, current

    def select_best_arima(self, y, d):
        results = []
        max_p = 5
        max_q = 5

        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0 and d == 0:
                    continue

                try:
                    trend_option = "n" if d > 0 else "c"

                    model = ARIMA(
                        y,
                        order=(p, d, q),
                        trend=trend_option,
                        enforce_stationarity=True,
                        enforce_invertibility=True
                    ).fit()

                    results.append({
                        "order": (p, d, q),
                        "aic": model.aic,
                        "bic": model.bic,
                        "hqic": model.hqic if hasattr(model, "hqic") else np.nan,
                        "model": model
                    })

                except Exception:
                    continue

        if not results:
            raise ValueError("No ARIMA model could be estimated successfully.")

        results = sorted(results, key=lambda x: x["aic"])
        self.model_candidates = results

        best = results[0]
        return best["order"], best["model"]

    def get_residuals(self):
        residuals = pd.Series(self.model.resid).dropna()

        if len(residuals) > 2:
            residuals = residuals.iloc[min(2, len(residuals)-1):]

        return residuals.dropna()

    def add_plot_page(self, title, plot_function):
        page = tk.Frame(self.content_frame)

        title_label = tk.Label(
            page,
            text=title,
            font=("Arial", 15, "bold")
        )
        title_label.pack(pady=10)

        plot_area = tk.Frame(page)
        plot_area.pack(fill="both", expand=True)

        fig, ax = plt.subplots(figsize=(9.8, 5.6), dpi=100)

        plot_function(fig, ax)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_area)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        plt.close(fig)

        self.pages.append(page)
        self.page_titles.append(title)

        self.pdf_items.append({
            "type": "plot",
            "title": title,
            "plot_function": plot_function
        })

    def add_text_page(self, title, content):
        page = tk.Frame(self.content_frame)

        title_label = tk.Label(
            page,
            text=title,
            font=("Arial", 15, "bold")
        )
        title_label.pack(pady=10)

        text_frame = tk.Frame(page)
        text_frame.pack(fill="both", expand=True, padx=20, pady=10)

        scrollbar_y = tk.Scrollbar(text_frame, orient="vertical")
        scrollbar_y.pack(side="right", fill="y")

        scrollbar_x = tk.Scrollbar(text_frame, orient="horizontal")
        scrollbar_x.pack(side="bottom", fill="x")

        text_box = tk.Text(
            text_frame,
            wrap="none",
            font=("Consolas", 10),
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set
        )
        text_box.pack(side="left", fill="both", expand=True)

        scrollbar_y.config(command=text_box.yview)
        scrollbar_x.config(command=text_box.xview)

        text_box.insert("1.0", content)
        text_box.config(state="disabled")

        self.pages.append(page)
        self.page_titles.append(title)

        self.pdf_items.append({
            "type": "text",
            "title": title,
            "content": content
        })

    def plot_original_series(self, fig, ax):
        y = self.series

        ax.plot(
            range(len(y)),
            y.values,
            marker="o",
            linewidth=2,
            color="#1f77b4"
        )

        tick_positions = np.arange(len(y))
        if len(y) > 15:
            step = max(1, len(y) // 10)
            tick_positions = tick_positions[::step]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(y.index[i]) for i in tick_positions], rotation=45)

        ax.set_title("Original Time Series")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    def plot_acf_pacf(self, fig):
        fig.clear()

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        series = self.stationary_series.dropna()
        n = len(series)

        if n < 8:
            ax1.text(0.5, 0.5, "Too few observations for ACF", ha="center", va="center")
            ax2.text(0.5, 0.5, "Too few observations for PACF", ha="center", va="center")
            ax1.set_axis_off()
            ax2.set_axis_off()
            return

        max_lags = min(10, max(1, n // 3))

        try:
            plot_acf(series, ax=ax1, lags=max_lags, zero=False)
            ax1.set_title("ACF")
        except Exception:
            ax1.text(0.5, 0.5, "ACF could not be plotted", ha="center", va="center")
            ax1.set_axis_off()

        try:
            plot_pacf(series, ax=ax2, lags=max_lags, zero=False, method="ywm")
            ax2.set_title("PACF")
        except Exception:
            ax2.text(0.5, 0.5, "PACF could not be plotted", ha="center", va="center")
            ax2.set_axis_off()

    def plot_residuals(self, fig):
        fig.clear()

        residuals = self.get_residuals()

        if len(residuals) < 5:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Too few residuals for diagnostics", ha="center", va="center")
            ax.set_axis_off()
            return

        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        ax1.plot(residuals.values, color="#1f77b4")
        ax1.axhline(0, color="black", linewidth=1)
        ax1.set_title("Residuals")
        ax1.grid(True, alpha=0.3)

        ax2.hist(residuals.values, bins=12, edgecolor="black", color="#1f77b4")
        ax2.set_title("Residual Histogram")
        ax2.grid(True, alpha=0.3)

        try:
            max_lags = min(10, max(1, len(residuals) // 3))
            plot_acf(residuals, ax=ax3, lags=max_lags, zero=False)
            ax3.set_title("Residual ACF")
        except Exception:
            ax3.text(0.5, 0.5, "Residual ACF could not be plotted", ha="center", va="center")
            ax3.set_axis_off()

    def plot_forecast(self, fig, ax):
        y = self.series

        forecast = self.model.get_forecast(steps=self.forecast_steps)
        predicted = forecast.predicted_mean
        conf_int = forecast.conf_int()

        historical_x = list(range(len(y)))
        forecast_x = list(range(len(y), len(y) + self.forecast_steps))

        ax.plot(
            historical_x,
            y.values,
            label="Actual",
            marker="o",
            linewidth=2,
            color="#1f77b4"
        )

        ax.plot(
            forecast_x,
            predicted.values,
            label="Forecast",
            marker="o",
            linewidth=2,
            color="red"
        )

        ax.fill_between(
            forecast_x,
            conf_int.iloc[:, 0].values,
            conf_int.iloc[:, 1].values,
            color="red",
            alpha=0.2,
            label="95% Confidence Interval"
        )

        labels = list(y.index) + [f"F{i+1}" for i in range(self.forecast_steps)]
        full_x = historical_x + forecast_x

        if len(full_x) > 15:
            step = max(1, len(full_x) // 10)
            xticks = full_x[::step]
        else:
            xticks = full_x

        ax.set_xticks(xticks)
        ax.set_xticklabels([str(labels[i]) for i in xticks], rotation=45)

        ax.set_title("Forecast")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def format_adf_result(self, result, series_name, regression_label):
        if result is None:
            return (
                f"ADF test for {series_name} ({regression_label}) could not be calculated.\n"
                "Possible causes:\n"
                "- Too few observations\n"
                "- Series is constant\n"
                "- Numerical instability\n\n"
            )

        adf_stat = result[0]
        p_value = result[1]
        used_lag = result[2]
        nobs = result[3]
        critical_values = result[4]

        text = ""
        text += f"ADF Test for {series_name} [{regression_label}]\n"
        text += "-" * 80 + "\n"
        text += f"ADF Statistic : {adf_stat:.6f}\n"
        text += f"p-value       : {p_value:.6f}\n"
        text += f"Used Lag      : {used_lag}\n"
        text += f"Observations  : {nobs}\n\n"

        text += "Critical Values:\n"
        for k, v in critical_values.items():
            text += f"{k}: {v:.6f}\n"

        text += "\nHypotheses:\n"
        text += "H0 : The series has a unit root (non-stationary under this specification).\n"
        text += "H1 : The series is stationary.\n\n"

        if p_value <= 0.05:
            text += "Decision at 5%: Reject H0.\n"
            text += "Interpretation: The test provides statistical evidence in favor of stationarity.\n\n"
        else:
            text += "Decision at 5%: Fail to reject H0.\n"
            text += "Interpretation: The test does not provide sufficient evidence against a unit root.\n\n"

        return text

    def format_kpss_result(self, result, series_name, regression_label):
        if result is None:
            return (
                f"KPSS test for {series_name} ({regression_label}) could not be calculated.\n"
                "Possible causes:\n"
                "- Too few observations\n"
                "- Series is constant\n"
                "- Numerical instability\n\n"
            )

        stat = result[0]
        p_value = result[1]
        lags = result[2]
        critical_values = result[3]

        text = ""
        text += f"KPSS Test for {series_name} [{regression_label}]\n"
        text += "-" * 80 + "\n"
        text += f"KPSS Statistic : {stat:.6f}\n"
        text += f"p-value        : {p_value:.6f}\n"
        text += f"Used Lags      : {lags}\n\n"

        text += "Critical Values:\n"
        for k, v in critical_values.items():
            text += f"{k}: {v:.6f}\n"

        text += "\nHypotheses:\n"
        text += "H0 : The series is stationary.\n"
        text += "H1 : The series is non-stationary.\n\n"

        if p_value > 0.05:
            text += "Decision at 5%: Fail to reject H0.\n"
            text += "Interpretation: The test is consistent with stationarity.\n\n"
        else:
            text += "Decision at 5%: Reject H0.\n"
            text += "Interpretation: The test suggests non-stationarity.\n\n"

        return text

    def format_level_stationarity_report(self, adf_c, adf_ct, kpss_c, kpss_ct):
        text = ""
        text += "Stationarity Analysis at Level\n"
        text += "=" * 80 + "\n\n"

        text += "This report combines ADF and KPSS tests.\n"
        text += "ADF null hypothesis: unit root (non-stationary).\n"
        text += "KPSS null hypothesis: stationary.\n\n"

        text += self.format_adf_result(adf_c, "Level Series", "constant")
        text += self.format_adf_result(adf_ct, "Level Series", "constant + trend")
        text += self.format_kpss_result(kpss_c, "Level Series", "constant")
        text += self.format_kpss_result(kpss_ct, "Level Series", "constant + trend")

        text += "Interpretation guide:\n"
        text += "- If ADF rejects unit root and KPSS does not reject stationarity, evidence favors stationarity.\n"
        text += "- If ADF fails to reject unit root and KPSS rejects stationarity, evidence favors non-stationarity.\n"
        text += "- Mixed results suggest caution, possible trend effects, structural change, or limited sample size.\n\n"

        return text

    def format_differencing_result(self, adf_result, kpss_result):
        text = ""
        text += "Automatic Differencing Analysis\n"
        text += "=" * 80 + "\n\n"

        text += f"Selected differencing order (d): {self.d}\n\n"

        if self.d == 0:
            text += (
                "Interpretation:\n"
                "The original series was judged sufficiently stationary for non-seasonal ARIMA modeling.\n\n"
            )
        elif self.d == 1:
            text += (
                "Interpretation:\n"
                "One difference was required to reduce non-stationarity.\n\n"
            )
        elif self.d == 2:
            text += (
                "Interpretation:\n"
                "Two differences were required under the automatic procedure.\n"
                "This may indicate strong persistence or trend behavior.\n\n"
            )

        text += "General considerations:\n"
        text += "- Differencing is used to remove unit roots and persistent stochastic trends.\n"
        text += "- The objective is to use the smallest d that produces a sufficiently stationary series.\n"
        text += "- Over-differencing can introduce unnecessary noise and artificial dynamics.\n\n"

        text += self.format_adf_result(adf_result, f"Differenced Series (d={self.d})", "constant")
        text += self.format_kpss_result(kpss_result, f"Differenced Series (d={self.d})", "constant")

        return text

    def format_model_summary(self):
        text = ""
        text += "ARIMA Model Selection Result\n"
        text += "=" * 80 + "\n\n"

        p, d, q = self.best_order
        text += f"Selected model: ARIMA({p},{d},{q})\n\n"

        text += "Parameter meanings:\n"
        text += "p : number of autoregressive terms\n"
        text += "d : number of differences\n"
        text += "q : number of moving-average terms\n\n"

        text += "Selection procedure:\n"
        text += "- Candidate models were estimated over a grid of p and q values.\n"
        text += "- The final model was selected by minimum AIC.\n"
        text += "- BIC and HQIC are also reported for comparison.\n\n"

        if self.model_candidates:
            text += "Top candidate models by AIC:\n"
            text += "-" * 80 + "\n"
            text += f"{'Order':<15}{'AIC':>15}{'BIC':>15}{'HQIC':>15}\n"
            text += "-" * 80 + "\n"

            for item in self.model_candidates[:10]:
                order = str(item["order"])
                aic = item["aic"]
                bic = item["bic"]
                hqic = item["hqic"]
                text += f"{order:<15}{aic:>15.4f}{bic:>15.4f}{hqic:>15.4f}\n"

            text += "\n"

        text += "Selected model summary from statsmodels:\n"
        text += "-" * 80 + "\n"

        try:
            text += str(self.model.summary())
        except Exception:
            text += "Model summary unavailable.\n"

        return text

    def format_residual_tests(self):
        residuals = self.get_residuals()

        text = ""
        text += "Residual Diagnostic Tests\n"
        text += "=" * 80 + "\n\n"

        if len(residuals) < 10:
            text += "Too few residuals for reliable residual diagnostics.\n"
            return text

        p, d, q = self.best_order

        try:
            max_lag = min(10, len(residuals) // 2)
            max_lag = max(max_lag, p + q + 1)

            if max_lag >= len(residuals):
                max_lag = max(1, len(residuals) - 1)

            lb = acorr_ljungbox(
                residuals,
                lags=[max_lag],
                model_df=p + q,
                return_df=True
            )

            lb_stat = lb["lb_stat"].iloc[0]
            lb_pvalue = lb["lb_pvalue"].iloc[0]

            text += "1) Ljung-Box Test\n"
            text += "-" * 80 + "\n"
            text += f"Lag tested : {max_lag}\n"
            text += f"Model df   : {p + q}\n"
            text += f"Statistic  : {lb_stat:.6f}\n"
            text += f"p-value    : {lb_pvalue:.6f}\n\n"
            text += "H0 : Residuals are white noise.\n"
            text += "H1 : Residuals are autocorrelated.\n\n"

            if lb_pvalue >= 0.05:
                text += "Decision at 5%: Fail to reject H0.\n"
                text += "Interpretation: No statistically significant residual autocorrelation is detected.\n\n"
            else:
                text += "Decision at 5%: Reject H0.\n"
                text += "Interpretation: Residual autocorrelation remains; model re-specification may be needed.\n\n"

        except Exception as e:
            text += "1) Ljung-Box Test\n"
            text += "-" * 80 + "\n"
            text += f"Test failed: {e}\n\n"

        try:
            jb_stat, jb_pvalue, skewness, kurt = jarque_bera(residuals)

            text += "2) Jarque-Bera Normality Test\n"
            text += "-" * 80 + "\n"
            text += f"Statistic  : {jb_stat:.6f}\n"
            text += f"p-value    : {jb_pvalue:.6f}\n"
            text += f"Skewness   : {skewness:.6f}\n"
            text += f"Kurtosis   : {kurt:.6f}\n\n"
            text += "H0 : Residuals are normally distributed.\n"
            text += "H1 : Residuals are not normally distributed.\n\n"

            if jb_pvalue >= 0.05:
                text += "Decision at 5%: Fail to reject H0.\n"
                text += "Interpretation: Residual normality is not rejected.\n\n"
            else:
                text += "Decision at 5%: Reject H0.\n"
                text += "Interpretation: Residuals deviate from normality.\n\n"

        except Exception as e:
            text += "2) Jarque-Bera Normality Test\n"
            text += "-" * 80 + "\n"
            text += f"Test failed: {e}\n\n"

        try:
            arch_lags = min(5, max(1, len(residuals) // 5))
            arch_test = het_arch(residuals, nlags=arch_lags)

            arch_stat = arch_test[0]
            arch_pvalue = arch_test[1]

            text += "3) ARCH Heteroskedasticity Test\n"
            text += "-" * 80 + "\n"
            text += f"Lags used  : {arch_lags}\n"
            text += f"Statistic  : {arch_stat:.6f}\n"
            text += f"p-value    : {arch_pvalue:.6f}\n\n"
            text += "H0 : No ARCH effects (conditional homoskedasticity).\n"
            text += "H1 : ARCH effects are present.\n\n"

            if arch_pvalue >= 0.05:
                text += "Decision at 5%: Fail to reject H0.\n"
                text += "Interpretation: No statistically significant ARCH effect is detected.\n\n"
            else:
                text += "Decision at 5%: Reject H0.\n"
                text += "Interpretation: Conditional heteroskedasticity may be present.\n\n"

        except Exception as e:
            text += "3) ARCH Heteroskedasticity Test\n"
            text += "-" * 80 + "\n"
            text += f"Test failed: {e}\n\n"

        text += "Overall diagnostic interpretation:\n"
        text += "- A good ARIMA model should leave residuals approximately uncorrelated.\n"
        text += "- Normality is useful for inference, but mild deviations do not necessarily invalidate forecasting.\n"
        text += "- Significant ARCH effects suggest time-varying volatility not captured by the ARIMA mean equation.\n"

        return text

    # ==========================================================
    # PDF EXPORT SECTION
    # ==========================================================

    def export_pdf(self):
        default_name = "box_jenkins_full_report.pdf"

        pdf_path = filedialog.asksaveasfilename(
            title="Save PDF Report",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF files", "*.pdf")]
        )

        if not pdf_path:
            return

        try:
            with PdfPages(pdf_path) as pdf:
                self.add_pdf_cover_page(pdf)

                for item in self.pdf_items:
                    if item["type"] == "plot":
                        self.add_pdf_plot_page(
                            pdf=pdf,
                            title=item["title"],
                            plot_function=item["plot_function"]
                        )
                    elif item["type"] == "text":
                        self.add_pdf_text_pages(
                            pdf=pdf,
                            title=item["title"],
                            content=item["content"]
                        )

            messagebox.showinfo(
                "PDF Export Complete",
                f"PDF report was successfully saved:\n{pdf_path}"
            )

        except Exception as e:
            messagebox.showerror(
                "PDF Export Error",
                f"Could not export PDF report.\n\nError:\n{e}"
            )

    def add_pdf_cover_page(self, pdf):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")

        title = "Box-Jenkins Time Series Analyzer"
        subtitle = "Full Mathematical and Econometric Report"

        fig.text(
            0.5, 0.78,
            title,
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold"
        )

        fig.text(
            0.5, 0.72,
            subtitle,
            ha="center",
            va="center",
            fontsize=14
        )

        fig.text(
            0.12, 0.60,
            f"Input file: {os.path.basename(self.file_path)}",
            ha="left",
            va="center",
            fontsize=11
        )

        fig.text(
            0.12, 0.56,
            f"Number of observations: {len(self.series)}",
            ha="left",
            va="center",
            fontsize=11
        )

        if self.best_order is not None:
            p, d, q = self.best_order
            fig.text(
                0.12, 0.52,
                f"Selected ARIMA model: ARIMA({p},{d},{q})",
                ha="left",
                va="center",
                fontsize=11
            )

        fig.text(
            0.12, 0.45,
            "Report contents:",
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold"
        )

        contents = [
            "1. Original time series plot",
            "2. ADF and KPSS stationarity tests",
            "3. Automatic differencing decision",
            "4. ACF and PACF identification plots",
            "5. ARIMA model selection and estimation results",
            "6. Residual diagnostic plots",
            "7. Residual diagnostic statistical tests",
            "8. Forecast plot with confidence interval"
        ]

        y_pos = 0.41
        for line in contents:
            fig.text(
                0.15, y_pos,
                line,
                ha="left",
                va="center",
                fontsize=10
            )
            y_pos -= 0.035

        fig.text(
            0.5, 0.08,
            "Generated automatically by the Box-Jenkins Time Series Analyzer",
            ha="center",
            va="center",
            fontsize=9,
            color="gray"
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def add_pdf_plot_page(self, pdf, title, plot_function):
        fig, ax = plt.subplots(figsize=(11.69, 8.27))

        try:
            plot_function(fig, ax)
        except TypeError:
            plot_function(fig)

        fig.suptitle(title, fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def add_pdf_text_pages(self, pdf, title, content):
        lines = []

        for raw_line in content.splitlines():
            if len(raw_line) <= 105:
                lines.append(raw_line)
            else:
                wrapped = textwrap.wrap(
                    raw_line,
                    width=105,
                    replace_whitespace=False,
                    drop_whitespace=False
                )
                lines.extend(wrapped)

        lines_per_page = 48
        chunks = [
            lines[i:i + lines_per_page]
            for i in range(0, len(lines), lines_per_page)
        ]

        if not chunks:
            chunks = [[""]]

        for page_num, chunk in enumerate(chunks, start=1):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")

            display_title = title
            if len(chunks) > 1:
                display_title = f"{title} - Part {page_num}"

            fig.text(
                0.05, 0.965,
                display_title,
                ha="left",
                va="top",
                fontsize=14,
                fontweight="bold"
            )

            y = 0.925
            line_height = 0.018

            for line in chunk:
                fig.text(
                    0.05,
                    y,
                    line,
                    ha="left",
                    va="top",
                    fontsize=7.8,
                    family="monospace"
                )
                y -= line_height

            fig.text(
                0.5,
                0.025,
                f"Box-Jenkins Time Series Analyzer | {os.path.basename(self.file_path)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="gray"
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    # ==========================================================
    # GUI NAVIGATION
    # ==========================================================

    def show_page(self, page_number):
        for page in self.pages:
            page.pack_forget()

        self.current_page = page_number
        self.pages[self.current_page].pack(fill="both", expand=True)

        self.page_indicator.config(
            text=f"Page {self.current_page + 1} of {len(self.pages)}"
        )

        if self.current_page == 0:
            self.prev_btn.config(state="disabled")
        else:
            self.prev_btn.config(state="normal")

        if self.current_page == len(self.pages) - 1:
            self.next_btn.config(state="disabled")
        else:
            self.next_btn.config(state="normal")

    def next_page(self):
        if self.current_page < len(self.pages) - 1:
            self.show_page(self.current_page + 1)

    def previous_page(self):
        if self.current_page > 0:
            self.show_page(self.current_page - 1)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImportWindow(root)
    root.mainloop()
