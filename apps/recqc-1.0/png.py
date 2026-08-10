import io
import logging
import tempfile
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk
from ui_theme import COLORS, configure_theme

LOGGER = logging.getLogger(__name__)


class ViewPage(ttk.Frame):
    def __init__(self, master, callback, search_results):
        super().__init__(master)
        self.callback = callback
        self.search_results = search_results[:6]
        self._images = []
        self._build()

    def _build(self):
        for column in range(3):
            self.columnconfigure(column, weight=1, uniform="result")
        for row in range(2):
            self.rowconfigure(row, weight=1, uniform="result")

        for index, item in enumerate(self.search_results):
            compound_id, data = item
            card = ttk.Frame(self, style="Surface.TFrame", padding=12)
            card.grid(
                row=index // 3,
                column=index % 3,
                padx=8,
                pady=8,
                sticky="nsew",
            )
            card.columnconfigure(0, weight=1)
            card.rowconfigure(0, weight=1)

            image = Image.open(io.BytesIO(data[4])).resize(
                (220, 220), Image.Resampling.LANCZOS
            )
            photo = ImageTk.PhotoImage(image)
            self._images.append(photo)
            tk.Label(card, image=photo, background=COLORS["surface"]).grid(
                row=0, column=0, sticky="nsew"
            )
            ttk.Label(
                card,
                text=f"Score  {data[0]}",
                style="Score.TLabel",
            ).grid(row=1, column=0, sticky="w", pady=(10, 2))
            ttk.Label(
                card,
                text=f"ID  {compound_id}",
                style="Muted.TLabel",
            ).grid(row=2, column=0, sticky="w")
            ttk.Button(
                card,
                text="View details",
                command=lambda idx=index: self.callback(idx),
            ).grid(row=3, column=0, sticky="e", pady=(8, 0))


class ResultViewerApp(tk.Toplevel):
    def __init__(self, root, search_results, callback=None):
        super().__init__(root)
        self.title("ReCQC 1.0 - Candidate results")
        self.geometry("1080x800")
        self.minsize(920, 700)
        configure_theme(self)
        self.callback = callback
        self.search_results = list(search_results.items())
        self.pages = [
            self.search_results[index : index + 6]
            for index in range(0, len(self.search_results), 6)
        ]
        self.current_index = 0
        self._build()

    def _build(self):
        header = ttk.Frame(self, style="Header.TFrame", padding=(20, 12))
        header.pack(fill="x")
        ttk.Label(header, text="Candidate results", style="Header.TLabel").pack(side="left")
        self.page_label = ttk.Label(
            header,
            text="",
            style="Header.TLabel",
            font=("Segoe UI", 10),
        )
        self.page_label.pack(side="right")

        self.content = ttk.Frame(self, padding=(16, 10))
        self.content.pack(fill="both", expand=True)

        footer = ttk.Frame(self, padding=(20, 8, 20, 16))
        footer.pack(fill="x")
        self.previous_button = ttk.Button(footer, text="Previous", command=self.prev_page)
        self.previous_button.pack(side="left")
        self.next_button = ttk.Button(footer, text="Next", command=self.next_page)
        self.next_button.pack(side="right")

        if self.pages:
            self.show_page(0)
        else:
            ttk.Label(
                self.content,
                text="No candidates were returned.",
                style="Subtitle.TLabel",
            ).pack(expand=True)
            self._update_navigation()

    def show_page(self, index):
        if not 0 <= index < len(self.pages):
            return
        for widget in self.content.winfo_children():
            widget.destroy()
        page = ViewPage(self.content, self.on_button_click, self.pages[index])
        page.pack(fill="both", expand=True)
        self.current_index = index
        self._update_navigation()

    def _update_navigation(self):
        page_count = len(self.pages)
        self.page_label.configure(
            text=f"Page {self.current_index + 1} of {page_count}" if page_count else "0 results"
        )
        self.previous_button.configure(
            state="normal" if self.current_index > 0 else "disabled"
        )
        self.next_button.configure(
            state="normal" if self.current_index < page_count - 1 else "disabled"
        )

    def prev_page(self):
        self.show_page(self.current_index - 1)

    def next_page(self):
        self.show_page(self.current_index + 1)

    def on_button_click(self, index):
        global_index = self.current_index * 6 + index
        if not 0 <= global_index < len(self.search_results):
            return
        if self.callback is not None:
            self.callback(global_index)
        DetailPage(self, self.search_results[global_index])


class DetailPage(tk.Toplevel):
    def __init__(self, root, item):
        super().__init__(root)
        self.title("ReCQC 1.0 - Candidate detail")
        self.geometry("1320x860")
        self.minsize(1100, 760)
        configure_theme(self)
        self.item = item
        self.images = {}
        self.ppm_images = {}
        self._build()

    def _build(self):
        compound_id, data = self.item
        header = ttk.Frame(self, style="Header.TFrame", padding=(22, 12))
        header.pack(fill="x")
        ttk.Label(header, text=f"Candidate {compound_id}", style="Header.TLabel").pack(
            side="left"
        )
        ttk.Label(
            header,
            text=f"Rank {data[1]}   •   Score {data[0]}",
            style="Header.TLabel",
            font=("Segoe UI", 10),
        ).pack(side="right")

        metadata = ttk.Frame(self, style="Surface.TFrame", padding=(18, 12))
        metadata.pack(fill="x", padx=18, pady=(18, 10))
        metadata.columnconfigure(1, weight=1)
        ttk.Label(metadata, text=f"Molecular weight: {data[2]}", style="Surface.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 24)
        )
        ttk.Label(metadata, text=f"Formula: {data[7]}", style="Surface.TLabel").grid(
            row=0, column=1, sticky="w"
        )
        ttk.Button(
            metadata,
            text="Open matched-atom table",
            command=lambda: self.open_html_table(data[3]),
        ).grid(row=0, column=2, sticky="e")
        ttk.Label(metadata, text="SMILES", style="Muted.TLabel").grid(
            row=1, column=0, sticky="nw", pady=(10, 0)
        )
        smiles = ttk.Entry(metadata)
        smiles.insert(0, data[8])
        smiles.configure(state="readonly")
        smiles.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(10, 0))

        visual_body = ttk.Frame(self, padding=(18, 8, 18, 18))
        visual_body.pack(fill="both", expand=True)
        visual_body.columnconfigure(0, weight=1)
        visual_body.columnconfigure(1, weight=1)
        visual_body.rowconfigure(0, weight=1)

        molecule_panel = ttk.LabelFrame(visual_body, text="Molecular annotation")
        molecule_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        molecule_panel.columnconfigure(0, weight=1)
        molecule_panel.rowconfigure(1, weight=1)

        molecule_options = ttk.Frame(molecule_panel, style="Surface.TFrame")
        molecule_options.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        molecule_sources = {
            "Atom indices": data[4],
            "Database shifts": data[5],
            "User shifts": data[6],
        }
        self.molecule_var = tk.StringVar(value="Atom indices")
        for column, (label, image_bytes) in enumerate(molecule_sources.items()):
            image = Image.open(io.BytesIO(image_bytes)).resize(
                (500, 500), Image.Resampling.LANCZOS
            )
            self.images[label] = ImageTk.PhotoImage(image)
            ttk.Radiobutton(
                molecule_options,
                text=label,
                variable=self.molecule_var,
                value=label,
                command=self.update_molecule_image,
            ).grid(row=0, column=column, padx=(0, 8))
        self.molecule_image = tk.Label(
            molecule_panel,
            image=self.images[self.molecule_var.get()],
            background=COLORS["surface"],
        )
        self.molecule_image.grid(row=1, column=0, sticky="nsew")

        spectrum_panel = ttk.LabelFrame(visual_body, text="Chemical-shift comparison")
        spectrum_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        spectrum_panel.columnconfigure(0, weight=1)
        spectrum_panel.rowconfigure(1, weight=1)

        spectrum_options = ttk.Frame(spectrum_panel, style="Surface.TFrame")
        spectrum_options.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        spectrum_sources = {"Database": data[9], "User": data[10]}
        self.spectrum_var = tk.StringVar(value="Database")
        for column, (label, image_bytes) in enumerate(spectrum_sources.items()):
            image = Image.open(io.BytesIO(image_bytes)).resize(
                (600, 380), Image.Resampling.LANCZOS
            )
            self.ppm_images[label] = ImageTk.PhotoImage(image)
            ttk.Radiobutton(
                spectrum_options,
                text=label,
                variable=self.spectrum_var,
                value=label,
                command=self.update_spectrum_image,
            ).grid(row=0, column=column, padx=(0, 8))
        self.spectrum_image = tk.Label(
            spectrum_panel,
            image=self.ppm_images[self.spectrum_var.get()],
            background=COLORS["surface"],
        )
        self.spectrum_image.grid(row=1, column=0, sticky="nsew")

    def update_molecule_image(self):
        image = self.images[self.molecule_var.get()]
        self.molecule_image.configure(image=image)
        self.molecule_image.image = image

    def update_spectrum_image(self):
        image = self.ppm_images[self.spectrum_var.get()]
        self.spectrum_image.configure(image=image)
        self.spectrum_image.image = image

    def open_html_table(self, html_content):
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", prefix="recqc_match_", encoding="utf-8", delete=False
            ) as html_file:
                html_file.write(html_content)
                path = Path(html_file.name)
            webbrowser.open(path.as_uri())
        except Exception as exc:
            LOGGER.exception("Failed to open the matched-atom table")
            messagebox.showerror("Result table", str(exc), parent=self)
