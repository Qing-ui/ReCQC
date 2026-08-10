import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from DBORGANIZATION import Add_file_path, DBOrganization
from DRAWCOMPOUND import DrawCompound
from MATCHINSTITUTE import ProductBestCompare, ScoreInstitution
from PIL import Image, ImageTk
from png import ResultViewerApp
from SDF import Add_user_path, File_path, Process_sdf_files
from similarsearch import GetSubstructures, SimilarSearch
from ui_theme import COLORS, configure_theme, make_text
from USERINPUT import ProcesInputdata


class Main:
    def __init__(self, master: tk.Tk):
        self.root = master
        self.root.title("ReCQC 1.0")
        self.root.geometry("1040x760")
        self.root.minsize(920, 680)
        configure_theme(self.root)
        self._set_icon()

        self.header = ttk.Frame(self.root, style="Header.TFrame", padding=(24, 12))
        self.header.pack(fill="x")
        ttk.Label(self.header, text="ReCQC 1.0", style="Header.TLabel").pack(side="left")
        ttk.Label(
            self.header,
            text="13C NMR dereplication",
            style="Header.TLabel",
            font=("Segoe UI", 10),
        ).pack(side="left", padx=(12, 0))

        nav = ttk.Frame(self.header, style="Header.TFrame")
        nav.pack(side="right")
        ttk.Button(
            nav,
            text="Analysis",
            style="Header.TButton",
            command=lambda: self.show_page("analysis", reset=True),
        ).pack(side="left", padx=2)
        ttk.Button(
            nav,
            text="Database tools",
            style="Header.TButton",
            command=lambda: self.show_page("database"),
        ).pack(side="left", padx=2)
        ttk.Button(
            nav,
            text="About",
            style="Header.TButton",
            command=lambda: self.show_page("about"),
        ).pack(side="left", padx=2)

        self.content = ttk.Frame(self.root, padding=24)
        self.content.pack(fill="both", expand=True)
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)

        self.pages = {
            "analysis": PrimaryPage(self.content),
            "database": DBPage(self.content),
            "about": AboutPage(self.content),
        }
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")
        self.show_page("analysis", reset=True)

    def _set_icon(self):
        icon_path = Path(__file__).resolve().with_name("tcalxy.jpg")
        try:
            icon = Image.open(icon_path).resize((48, 48), Image.Resampling.LANCZOS)
            self.icon = ImageTk.PhotoImage(icon)
            self.root.iconphoto(True, self.icon)
        except (OSError, tk.TclError):
            self.icon = None

    def show_page(self, page_name, reset=False):
        page = self.pages[page_name]
        if reset and isinstance(page, PrimaryPage):
            page.show_welcome()
        page.tkraise()


class PrimaryPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.filepath = File_path()
        self.custom_library = Add_user_path()
        self.current_child = None
        self.welcome = self._build_welcome()
        self.parameter_page = ParameterPage(
            self,
            on_back=self.show_welcome,
            on_continue=self.show_search,
            filepath_obj=self.filepath,
            usepath_obj=self.custom_library,
        )
        self.show_welcome()

    def _build_welcome(self):
        page = ttk.Frame(self)
        page.columnconfigure(0, weight=1)
        page.rowconfigure(0, weight=1)
        card = ttk.Frame(page, style="Surface.TFrame", padding=(60, 50))
        card.grid(row=0, column=0, sticky="nsew", padx=70, pady=55)
        card.columnconfigure(0, weight=1)

        ttk.Label(card, text="Natural-product mixture analysis", style="Section.TLabel").grid(
            row=0, column=0, pady=(10, 14)
        )
        ttk.Label(
            card,
            text="ReCQC 1.0",
            style="Surface.TLabel",
            font=("Segoe UI Semibold", 34),
        ).grid(row=1, column=0, pady=(0, 12))
        ttk.Label(
            card,
            text=(
                "Screen candidate structures from experimental 13C NMR and DEPT data, "
                "with maximum-core substructure scoring for mixture dereplication."
            ),
            style="Muted.TLabel",
            font=("Segoe UI", 11),
            wraplength=620,
            justify="center",
        ).grid(row=2, column=0, pady=(0, 28))
        ttk.Button(
            card,
            text="Start spectrum analysis",
            style="Primary.TButton",
            command=lambda: self._show_child(self.parameter_page),
        ).grid(row=3, column=0, pady=(6, 14))
        ttk.Label(
            card,
            text="Step 1: choose libraries and thresholds  •  Step 2: enter spectra  •  Step 3: review candidates",
            style="Muted.TLabel",
        ).grid(row=4, column=0, pady=(8, 0))
        return page

    def _show_child(self, child):
        if self.current_child is not None:
            self.current_child.grid_remove()
        self.current_child = child
        child.grid(row=0, column=0, sticky="nsew")

    def show_welcome(self):
        self._show_child(self.welcome)

    def show_search(self, config):
        search_page = SearchPage(
            self,
            config=config,
            on_back=lambda: self._show_child(self.parameter_page),
        )
        self._show_child(search_page)


class ParameterPage(ttk.Frame):
    def __init__(self, master, on_back, on_continue, filepath_obj, usepath_obj):
        super().__init__(master)
        self.on_back = on_back
        self.on_continue = on_continue
        self.filepath_obj = filepath_obj
        self.usepath_obj = usepath_obj
        self.custom_paths = []
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        ttk.Label(self, text="Analysis setup", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            self,
            text="Select candidate libraries and define atom-error and maximum-core thresholds.",
            style="Subtitle.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(42, 0))

        body = ttk.Frame(self)
        body.grid(row=1, column=0, sticky="nsew", pady=(24, 18))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        library = ttk.LabelFrame(body, text="Candidate libraries")
        library.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        library.columnconfigure(0, weight=1)
        self.experimental_var = tk.BooleanVar(value=False)
        self.model_var = tk.BooleanVar(value=False)
        self.prediction_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            library,
            text="Experimental database",
            variable=self.experimental_var,
            command=self._toggle_experimental,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            library,
            text="Simulated database",
            variable=self.model_var,
            command=self._toggle_model,
        ).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(
            library,
            text="Predicted database",
            variable=self.prediction_var,
            command=self._toggle_prediction,
        ).grid(row=2, column=0, sticky="w")
        ttk.Separator(library).grid(row=3, column=0, sticky="ew", pady=14)
        ttk.Button(library, text="Add custom SDF…", command=self._add_custom_library).grid(
            row=4, column=0, sticky="w"
        )
        ttk.Button(library, text="Clear custom libraries", command=self._clear_custom).grid(
            row=5, column=0, sticky="w", pady=(8, 0)
        )
        self.custom_label = ttk.Label(
            library,
            text="No custom library selected",
            style="Muted.TLabel",
            wraplength=360,
            justify="left",
        )
        self.custom_label.grid(row=6, column=0, sticky="ew", pady=(14, 0))

        scoring = ttk.LabelFrame(body, text="Scoring and candidate filters")
        scoring.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        scoring.columnconfigure(1, weight=1)
        self.green_var = tk.StringVar(value="2.0")
        self.yellow_var = tk.StringVar(value="4.0")
        self.min_mw_var = tk.StringVar(value="0")
        self.max_mw_var = tk.StringVar(value="1000")
        self.min_c_var = tk.StringVar(value="0")
        self.max_c_var = tk.StringVar(value="100")
        self.skeleton_var = tk.StringVar(value="0.85")

        self._entry_row(scoring, 0, "Green atom tolerance (ppm)", self.green_var)
        self._entry_row(scoring, 1, "Yellow atom tolerance (ppm)", self.yellow_var)
        self._range_row(scoring, 2, "Molecular weight", self.min_mw_var, self.max_mw_var)
        self._range_row(scoring, 3, "Carbon count", self.min_c_var, self.max_c_var)
        self._entry_row(scoring, 4, "Maximum-core threshold", self.skeleton_var)
        ttk.Label(
            scoring,
            text=(
                "Atom colours are converted to continuous atom scores. Substructures are "
                "ranked and accumulated; the largest qualifying union is treated as the core."
            ),
            style="Muted.TLabel",
            wraplength=390,
            justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(18, 0))

        actions = ttk.Frame(self)
        actions.grid(row=2, column=0, sticky="ew")
        ttk.Button(actions, text="Back", command=self.on_back).pack(side="left")
        ttk.Button(
            actions,
            text="Continue to spectrum input",
            style="Primary.TButton",
            command=self._continue,
        ).pack(side="right")

    @staticmethod
    def _entry_row(parent, row, label, variable):
        ttk.Label(parent, text=label, style="Surface.TLabel").grid(
            row=row, column=0, sticky="w", pady=7
        )
        ttk.Entry(parent, textvariable=variable, width=14).grid(
            row=row, column=1, sticky="e", pady=7
        )

    @staticmethod
    def _range_row(parent, row, label, min_var, max_var):
        ttk.Label(parent, text=label, style="Surface.TLabel").grid(
            row=row, column=0, sticky="w", pady=7
        )
        range_frame = ttk.Frame(parent, style="Surface.TFrame")
        range_frame.grid(row=row, column=1, sticky="e", pady=7)
        ttk.Entry(range_frame, textvariable=min_var, width=7).pack(side="left")
        ttk.Label(range_frame, text="to", style="Surface.TLabel").pack(side="left", padx=7)
        ttk.Entry(range_frame, textvariable=max_var, width=7).pack(side="left")

    def _add_custom_library(self):
        self.custom_paths = self.usepath_obj.select_user_path(parent=self)
        self._update_custom_label()

    def _clear_custom(self):
        self.usepath_obj.user_path.clear()
        self.custom_paths = []
        self._update_custom_label()

    def _update_custom_label(self):
        if self.custom_paths:
            self.custom_label.configure(
                text="\n".join(Path(path).name for path in self.custom_paths)
            )
        else:
            self.custom_label.configure(text="No custom library selected")

    def _toggle_experimental(self):
        if self.experimental_var.get():
            self.filepath_obj.select_realdata_path()
        else:
            self.filepath_obj.remove_realdata_path()

    def _toggle_model(self):
        if self.model_var.get():
            self.filepath_obj.select_model_path()
        else:
            self.filepath_obj.remove_model_path()

    def _toggle_prediction(self):
        if self.prediction_var.get():
            self.filepath_obj.select_prediction_path()
        else:
            self.filepath_obj.remove_prediction_path()

    def _continue(self):
        try:
            green = float(self.green_var.get())
            yellow = float(self.yellow_var.get())
            min_mw, max_mw = float(self.min_mw_var.get()), float(self.max_mw_var.get())
            min_c, max_c = int(self.min_c_var.get()), int(self.max_c_var.get())
            skeleton = float(self.skeleton_var.get())
            if not 0 <= green < yellow:
                raise ValueError("Atom tolerances must satisfy 0 <= green < yellow")
            if min_mw < 0 or min_mw > max_mw:
                raise ValueError("Molecular-weight range must satisfy 0 <= minimum <= maximum")
            if min_c < 0 or min_c > max_c:
                raise ValueError("Carbon-count range must satisfy 0 <= minimum <= maximum")
            if not 0 <= skeleton <= 1:
                raise ValueError("Maximum-core threshold must be between 0 and 1")
            if not (self.filepath_obj.system_path or self.custom_paths):
                raise ValueError("Select at least one candidate library")
        except ValueError as exc:
            messagebox.showerror("Invalid setup", str(exc), parent=self)
            return

        self.on_continue(
            {
                "filepath_obj": self.filepath_obj,
                "custom_paths": list(self.custom_paths),
                "green_threshold": green,
                "yellow_threshold": yellow,
                "skeleton_threshold": skeleton,
                "min_c": min_c,
                "max_c": max_c,
                "min_mw": min_mw,
                "max_mw": max_mw,
            }
        )


class SearchPage(ttk.Frame):
    def __init__(self, master, config, on_back):
        super().__init__(master)
        self.config_data = config
        self.on_back = on_back
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        ttk.Label(self, text="Spectrum input", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            self,
            text="Enter comma-separated 13C shifts. Leave unavailable DEPT fields empty.",
            style="Subtitle.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(42, 0))

        card = ttk.Frame(self, style="Surface.TFrame", padding=20)
        card.grid(row=1, column=0, sticky="nsew", pady=(24, 18))
        card.columnconfigure(1, weight=1)
        card.rowconfigure((0, 1, 2, 3), weight=1)
        self.shift_boxes = {}
        fields = (
            ("all", "All 13C signals"),
            ("up", "+DEPT-135"),
            ("down", "-DEPT-135"),
            ("dept90", "DEPT-90"),
        )
        for row, (key, label) in enumerate(fields):
            ttk.Label(card, text=label, style="Surface.TLabel").grid(
                row=row, column=0, sticky="nw", padx=(0, 18), pady=8
            )
            box = make_text(card, height=3, width=75)
            box.grid(row=row, column=1, sticky="nsew", pady=8)
            self.shift_boxes[key] = box

        actions = ttk.Frame(self)
        actions.grid(row=2, column=0, sticky="ew")
        ttk.Button(actions, text="Back", command=self.on_back).pack(side="left")
        self.search_button = ttk.Button(
            actions,
            text="Run analysis",
            style="Primary.TButton",
            command=self.search,
        )
        self.search_button.pack(side="right")

    def search(self):
        raw = {key: box.get("1.0", "end-1c").strip() for key, box in self.shift_boxes.items()}
        try:
            self._validate_shift_text(raw)
            self.config_data["filepath_obj"].select_user_path(
                self.config_data["custom_paths"]
            )
            all_paths = self.config_data["filepath_obj"].select_all_paths()
            if not all_paths:
                raise ValueError("No candidate library is available")

            self.search_button.configure(state="disabled")
            self.winfo_toplevel().configure(cursor="watch")
            self.update_idletasks()

            input_data = ProcesInputdata(raw["all"], raw["up"], raw["down"], raw["dept90"])
            if input_data.match_type_state is None:
                raise ValueError("The spectrum combination could not be interpreted")
            sdf_result = Process_sdf_files(
                all_paths,
                input_data.match_type_state,
                self.config_data["min_mw"],
                self.config_data["max_mw"],
                self.config_data["min_c"],
                self.config_data["max_c"],
            )
            compare_data = ProductBestCompare(
                input_data, sdf_result, self.config_data["green_threshold"]
            )
            score = ScoreInstitution(
                compare_data,
                self.config_data["yellow_threshold"],
                self.config_data["skeleton_threshold"],
            )
            result_dict = DrawCompound(all_paths, score).result
            if not result_dict:
                raise ValueError("No candidates satisfied the selected data and filters")
            ResultViewerApp(self.winfo_toplevel(), result_dict, callback=lambda idx: idx)
        except Exception as exc:
            messagebox.showerror("Analysis error", str(exc), parent=self)
        finally:
            self.search_button.configure(state="normal")
            self.winfo_toplevel().configure(cursor="")

    @staticmethod
    def _validate_shift_text(values):
        if not any(values.values()):
            raise ValueError("Enter at least one spectrum")
        for label, text in values.items():
            if not text:
                continue
            entries = [value.strip() for value in text.replace("，", ",").split(",")]
            if any(not value for value in entries):
                raise ValueError(f"{label} contains an empty comma-separated value")
            try:
                [float(value) for value in entries]
            except ValueError as exc:
                raise ValueError(f"{label} contains a non-numeric chemical shift") from exc


class DBPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.folder_path = ""
        self._build()

    def _build(self):
        self.columnconfigure((0, 1), weight=1)
        ttk.Label(self, text="Database tools", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            self,
            text="Validate source folders, generate SDF databases, or run structure searches.",
            style="Subtitle.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(42, 0))

        folder = ttk.LabelFrame(self, text="Database folder")
        folder.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(24, 18))
        folder.columnconfigure(0, weight=1)
        self.folder_label = ttk.Label(
            folder, text="No folder selected", style="Muted.TLabel", wraplength=760
        )
        self.folder_label.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))
        ttk.Button(folder, text="Select folder…", command=self.select_folder_path).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Button(folder, text="Clear", command=self.delete_folder_path).grid(
            row=1, column=1, sticky="w", padx=8
        )
        ttk.Button(folder, text="Check folder", command=self.check_folder).grid(
            row=1, column=2, sticky="e", padx=8
        )
        ttk.Button(
            folder,
            text="Generate database",
            style="Primary.TButton",
            command=self.generate_sdf,
        ).grid(row=1, column=3, sticky="e")

        similar = ttk.LabelFrame(self, text="Similarity search")
        similar.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
        similar.columnconfigure(1, weight=1)
        self.radius_var = tk.StringVar(value="3")
        self.similarity_var = tk.StringVar(value="0.8")
        self.smiles_var = tk.StringVar()
        self._db_entry(similar, 0, "Radius", self.radius_var)
        self._db_entry(similar, 1, "Similarity threshold", self.similarity_var)
        self._db_entry(similar, 2, "Target SMILES", self.smiles_var)
        ttk.Button(
            similar,
            text="Run similarity search",
            style="Primary.TButton",
            command=self.similar_search,
        ).grid(row=3, column=0, columnspan=2, sticky="e", pady=(16, 0))

        substructure = ttk.LabelFrame(self, text="Substructure search")
        substructure.grid(row=2, column=1, sticky="nsew", padx=(10, 0))
        substructure.columnconfigure(1, weight=1)
        self.sub_smiles_var = tk.StringVar()
        self._db_entry(substructure, 0, "Query SMILES", self.sub_smiles_var)
        ttk.Label(
            substructure,
            text="Returns database molecules containing the specified query structure.",
            style="Muted.TLabel",
            wraplength=360,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Button(
            substructure,
            text="Run substructure search",
            style="Primary.TButton",
            command=self.get_substructure_search,
        ).grid(row=2, column=0, columnspan=2, sticky="e", pady=(16, 0))

    @staticmethod
    def _db_entry(parent, row, label, variable):
        ttk.Label(parent, text=label, style="Surface.TLabel").grid(
            row=row, column=0, sticky="w", pady=6, padx=(0, 10)
        )
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=6)

    def select_folder_path(self):
        self.folder_path = Add_file_path().select_user_path(parent=self)
        self.folder_label.configure(text=self.folder_path or "No folder selected")

    def delete_folder_path(self):
        self.folder_path = ""
        self.folder_label.configure(text="No folder selected")

    def _require_folder(self):
        if not self.folder_path:
            raise ValueError("Select a database folder first")

    def check_folder(self):
        try:
            self._require_folder()
            organizer = DBOrganization(self.folder_path)
            organizer.check_sdf_excel()
            organizer.check_sdf_mol()
            messagebox.showinfo("Database tools", "Folder validation completed", parent=self)
        except Exception as exc:
            messagebox.showerror("Database error", str(exc), parent=self)

    def generate_sdf(self):
        try:
            self._require_folder()
            DBOrganization(self.folder_path).write_sdf()
            messagebox.showinfo("Database tools", "Database generation completed", parent=self)
        except Exception as exc:
            messagebox.showerror("Database error", str(exc), parent=self)

    def similar_search(self):
        try:
            self._require_folder()
            radius = int(self.radius_var.get())
            threshold = float(self.similarity_var.get())
            if radius <= 0 or not 0 <= threshold <= 1:
                raise ValueError("Radius must be positive and similarity must be in [0, 1]")
            if not self.smiles_var.get().strip():
                raise ValueError("Enter a target SMILES string")
            SimilarSearch(
                self.smiles_var.get().strip(), self.folder_path, threshold, radius
            ).get_similars()
        except Exception as exc:
            messagebox.showerror("Similarity search", str(exc), parent=self)

    def get_substructure_search(self):
        try:
            self._require_folder()
            smiles = self.sub_smiles_var.get().strip()
            if not smiles:
                raise ValueError("Enter a query SMILES string")
            GetSubstructures(smiles, self.folder_path).get_substructures_sdf()
        except Exception as exc:
            messagebox.showerror("Substructure search", str(exc), parent=self)


class AboutPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        card = ttk.Frame(self, style="Surface.TFrame", padding=(55, 45))
        card.grid(row=0, column=0, sticky="nsew", padx=100, pady=70)
        card.columnconfigure(0, weight=1)
        ttk.Label(
            card,
            text="ReCQC 1.0",
            style="Surface.TLabel",
            font=("Segoe UI Semibold", 28),
        ).grid(row=0, column=0, pady=(0, 10))
        ttk.Label(
            card,
            text="13C NMR mixture dereplication with maximum-core substructure scoring",
            style="Muted.TLabel",
        ).grid(row=1, column=0, pady=(0, 24))
        ttk.Label(
            card,
            text=(
                "Natural Products Research and Application Team\n"
                "Yunnan University\n\n"
                "Contact: 2638614209@qq.com"
            ),
            style="Surface.TLabel",
            justify="center",
        ).grid(row=2, column=0)


if __name__ == "__main__":
    root = tk.Tk()
    Main(root)
    root.mainloop()
