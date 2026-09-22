"""Simple desktop app for predicting realized load."""

import datetime
import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from predict import calculate_balance, fetch_temperature, load_model, predict_realized_load
from train_daily_xgboost import main as train_model


ROOT = Path(__file__).parent
METRICS_FILE = ROOT / "artifacts_daily" / "metrics.json"

BG = "#0f1419"
CARD = "#1a2129"
TEXT = "#e6edf3"
MUTED = "#8b98a5"
BLUE = "#4da3ff"
GREEN = "#38c98b"
RED = "#ff6b65"
AMBER = "#e8a33d"


def read_metrics():
    """Read saved model scores. Return an empty dictionary if unavailable."""
    try:
        return json.loads(METRICS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


class EnergyApp(tk.Frame):
    def __init__(self, root):
        super().__init__(root, bg=BG, padx=25, pady=25)
        self.pack(fill="both", expand=True)

        self.model = None
        self.events = queue.Queue()

        self.date = tk.StringVar(value=datetime.date.today().isoformat())
        self.generation = tk.StringVar(value="14000")
        self.temperature = tk.StringVar(value="10.0")
        self.model_mode = tk.StringVar(value="lag")
        self.load_lag_1 = tk.StringVar(value="")
        self.load_lag_7 = tk.StringVar(value="")
        self.prediction = tk.StringVar(value="—")
        self.difference = tk.StringVar(value="—")
        self.percentage = tk.StringVar(value="—")
        self.balance = tk.StringVar(value="AWAITING INPUT")
        self.status = tk.StringVar(value="Loading model...")
        self.training_status = tk.StringVar(value="")

        self.build_window()
        self.load_saved_model()
        self.check_training_events()

    def build_window(self):
        """Create all GUI widgets."""
        self.columnconfigure(0, weight=1, uniform="cards")
        self.columnconfigure(1, weight=1, uniform="cards")
        self.rowconfigure(1, weight=1)

        header = tk.Frame(self, bg=BG)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        header.columnconfigure(0, weight=1)

        tk.Label(
            header,
            text="Energy Production Monitor",
            bg=BG,
            fg=TEXT,
            font=("Arial", 29, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            text="Predict daily realized load and compare it with generation",
            bg=BG,
            fg=MUTED,
        ).grid(row=1, column=0, sticky="w")

        self.score_label = tk.Label(
            header, bg=CARD, fg=BLUE, padx=12, pady=8, font=("Arial", 13, "bold")
        )
        self.score_label.grid(row=0, column=1, rowspan=2)
        self.update_score()

        self.build_input_card()
        self.build_result_card()

        tk.Label(self, textvariable=self.status, bg=BG, fg=MUTED).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(18, 0)
        )

    def new_card(self, column):
        card = tk.Frame(self, bg=CARD, padx=24, pady=24)
        card.grid(row=1, column=column, sticky="nsew", padx=(0, 10) if column == 0 else (10, 0))
        return card

    def new_scrollable_card(self, column):
        """Create a card whose contents can scroll vertically."""
        container = tk.Frame(self, bg=CARD)
        container.grid(
            row=1,
            column=column,
            sticky="nsew",
            padx=(0, 10) if column == 0 else (10, 0),
        )

        canvas = tk.Canvas(container, bg=CARD, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        card = tk.Frame(canvas, bg=CARD, padx=24, pady=24)
        window = canvas.create_window((0, 0), window=card, anchor="nw")
        card.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window, width=event.width),
        )

        def scroll(event):
            direction = -1 if event.num == 4 or event.delta > 0 else 1
            canvas.yview_scroll(direction, "units")

        def enable_scroll(_event):
            canvas.bind_all("<MouseWheel>", scroll)
            canvas.bind_all("<Button-4>", scroll)
            canvas.bind_all("<Button-5>", scroll)

        def disable_scroll(_event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        container.bind("<Enter>", enable_scroll)
        container.bind("<Leave>", disable_scroll)
        return card

    def build_input_card(self):
        card = self.new_scrollable_card(0)
        self.card_title(card, "INPUT")

        self.input_field(card, "Date", "YYYY-MM-DD", self.date)
        self.input_field(card, "Current energy generation", "daily average", self.generation)
        self.input_field(card, "Temperature", "daily mean °C", self.temperature)

        tk.Label(card, text="Prediction model", bg=CARD, fg=TEXT).pack(
            anchor="w", pady=(16, 4)
        )
        tk.Radiobutton(
            card,
            text="Lag model (uses load from 1 and 7 days earlier)",
            variable=self.model_mode,
            value="lag",
            bg=CARD,
            fg=TEXT,
            selectcolor=BG,
            activebackground=CARD,
            activeforeground=TEXT,
            command=self.update_model_inputs,
        ).pack(anchor="w")
        tk.Radiobutton(
            card,
            text="Fallback model (no previous load required)",
            variable=self.model_mode,
            value="fallback",
            bg=CARD,
            fg=TEXT,
            selectcolor=BG,
            activebackground=CARD,
            activeforeground=TEXT,
            command=self.update_model_inputs,
        ).pack(anchor="w")

        self.lag_input_frame = tk.Frame(card, bg=CARD)
        self.lag_input_frame.pack(fill="x")
        self.input_field(
            self.lag_input_frame,
            "Yesterday's realized load",
            "1 day earlier",
            self.load_lag_1,
        )
        self.input_field(
            self.lag_input_frame,
            "Realized load 7 days ago",
            "7 days earlier",
            self.load_lag_7,
        )

        self.fetch_button = self.button(
            card, "Fetch temperature for date", self.fetch_weather, "#2b3742"
        )
        self.fetch_button.pack(fill="x", ipady=7, pady=(16, 0))

        self.predict_button = self.button(
            card, "Predict consumption", self.predict, BLUE
        )
        self.predict_button.pack(fill="x", ipady=9, pady=(10, 0))

        ttk.Separator(card).pack(fill="x", pady=22)
        self.card_title(card, "MODEL TRAINING")
        tk.Label(
            card,
            text="Uses combined.csv",
            bg=CARD,
            fg=MUTED,
        ).pack(anchor="w", pady=(5, 8))

        self.train_button = self.button(card, "Train model", self.start_training, "#2b3742")
        self.train_button.pack(fill="x", ipady=7)
        self.progress = ttk.Progressbar(card, mode="indeterminate")

        tk.Label(card, textvariable=self.training_status, bg=CARD, fg=MUTED).pack(
            anchor="w", pady=(7, 0)
        )
        self.error_label = tk.Label(card, text="", bg=CARD, fg=RED, wraplength=340)
        self.error_label.pack(anchor="w", pady=(7, 0))

    def build_result_card(self):
        card = self.new_card(1)
        self.card_title(card, "PREDICTION")

        tk.Label(
            card,
            textvariable=self.prediction,
            bg=CARD,
            fg=TEXT,
            font=("Courier", 36, "bold"),
        ).pack(anchor="w", pady=(25, 0))
        tk.Label(card, text="predicted realized load", bg=CARD, fg=MUTED).pack(anchor="w")

        self.balance_label = tk.Label(
            card,
            textvariable=self.balance,
            bg="#2b3742",
            fg=MUTED,
            padx=12,
            pady=7,
            font=("Arial", 14, "bold"),
        )
        self.balance_label.pack(anchor="w", pady=(30, 20))

        self.result_row(card, "Generation - predicted load", self.difference)
        self.result_row(card, "Difference vs predicted load", self.percentage)

    def card_title(self, parent, text):
        tk.Label(parent, text=text, bg=CARD, fg=BLUE, font=("Arial", 13, "bold")).pack(
            anchor="w"
        )

    def input_field(self, parent, label, hint, variable):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=(20, 6))
        tk.Label(row, text=label, bg=CARD, fg=TEXT).pack(side="left")
        tk.Label(row, text=hint, bg=CARD, fg=MUTED).pack(side="right")

        entry = tk.Entry(parent, textvariable=variable, bg=BG, fg=TEXT, insertbackground=BLUE)
        entry.pack(fill="x", ipady=7)
        entry.bind("<Return>", lambda event: self.predict())

    def result_row(self, parent, label, value):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=8)
        tk.Label(row, text=label, bg=CARD, fg=MUTED).pack(side="left")
        tk.Label(row, textvariable=value, bg=CARD, fg=TEXT, font=("Courier", 14, "bold")).pack(
            side="right"
        )

    def button(self, parent, text, command, color):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg="#08131d" if color == BLUE else TEXT,
            relief="flat",
            cursor="hand2",
            font=("Arial", 14, "bold"),
        )

    def update_score(self):
        metrics = read_metrics()
        lag_score = metrics.get("r2")
        fallback_score = metrics.get("fallback_r2")
        text = (
            f"LAG R²  {lag_score:.4f}\nFALLBACK R²  {fallback_score:.4f}"
            if lag_score is not None and fallback_score is not None
            else "MODEL"
        )
        self.score_label.config(text=text)

    def update_model_inputs(self):
        """Show manual lag inputs only when the lag model is selected."""
        if self.model_mode.get() == "lag":
            self.lag_input_frame.pack(fill="x", before=self.fetch_button)
        else:
            self.lag_input_frame.pack_forget()

    def load_saved_model(self):
        try:
            self.model = load_model()
            self.predict_button.config(state="normal")
            self.status.set("Model ready")
        except Exception as error:
            self.predict_button.config(state="disabled")
            self.status.set(str(error))

    def predict(self):
        """Predict load and display the production balance."""
        self.error_label.config(text="")
        try:
            generation = float(self.generation.get())
            temperature = float(self.temperature.get())
            use_lags = self.model_mode.get() == "lag"
            if use_lags:
                try:
                    load_lag_1 = float(self.load_lag_1.get())
                    load_lag_7 = float(self.load_lag_7.get())
                except ValueError as error:
                    raise ValueError(
                        "Enter realized load for yesterday and seven days ago."
                    ) from error
            else:
                load_lag_1 = load_lag_7 = None
            load = predict_realized_load(
                self.date.get(),
                generation,
                temperature,
                self.model,
                use_lags=use_lags,
                load_lag_1=load_lag_1,
                load_lag_7=load_lag_7,
            )
            result = calculate_balance(generation, load)
        except (TypeError, ValueError) as error:
            self.error_label.config(text=str(error))
            return
        except Exception as error:
            self.error_label.config(text=f"Prediction failed: {error}")
            return

        self.prediction.set(f"{load:,.2f}")
        self.difference.set(f"{result['difference']:+,.2f}")
        self.percentage.set(f"{result['percentage']:+.2f}%")
        self.balance.set(result["status"].upper())

        colors = {"Overproducing": GREEN, "Underproducing": RED, "Balanced": AMBER}
        self.balance_label.config(fg=colors[result["status"]])

    def fetch_weather(self):
        """Fill the temperature field from local data or Open-Meteo."""
        self.error_label.config(text="")
        try:
            temperature = fetch_temperature(self.date.get())
        except Exception as error:
            self.error_label.config(text=f"Could not fetch temperature: {error}")
            return
        self.temperature.set(f"{temperature:.2f}")
        self.status.set("Temperature loaded for the selected date")

    def start_training(self):
        """Start model training without freezing the window."""
        self.train_button.config(state="disabled", text="Training...")
        self.predict_button.config(state="disabled")
        self.progress.pack(fill="x", pady=(10, 0))
        self.progress.start()
        self.training_status.set("Training XGBoost from the cleaned daily data...")
        threading.Thread(target=self.train_in_background, daemon=True).start()

    def train_in_background(self):
        try:
            train_model()
            self.events.put(None)
        except Exception as error:
            self.events.put(error)

    def check_training_events(self):
        """Check whether the background training thread has finished."""
        try:
            result = self.events.get_nowait()
            self.finish_training(result)
        except queue.Empty:
            pass
        self.after(100, self.check_training_events)

    def finish_training(self, error):
        self.progress.stop()
        self.progress.pack_forget()
        self.train_button.config(state="normal", text="Train model")

        if error:
            self.training_status.set(f"Training failed: {error}")
            self.load_saved_model()
            return

        self.load_saved_model()
        self.update_score()
        metrics = read_metrics()
        self.training_status.set(
            "Training complete - "
            f"lag R² {metrics['r2']:.4f}, "
            f"fallback R² {metrics['fallback_r2']:.4f}"
        )


def main():
    root = tk.Tk()
    root.option_add("*Font", ("Arial", 13))
    root.title("Energy Production Monitor")
    root.geometry("980x900")
    root.minsize(860, 820)
    root.configure(bg=BG)
    ttk.Style(root).theme_use("clam")
    EnergyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
