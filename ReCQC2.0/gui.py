import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

from PIL import ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PROCESSSDFFILES import SDFFileSelector
from CarbonScorerResult import *
from HSQCScorerResult import *
from CombineScorerResult import *
import json
import re

class ChemTheme:

    COLORS = {
        'background': '#2E5266',     # Deep blue background
        'foreground': '#E2E8E4',     # Off-white background
        'accent1': '#6B8F71',        # leaf green
        'accent2': '#A3C4BC',        # turquoise
        'highlight': '#FFD166'       # amber
    }

    FONTS = {
        'title': ('Helvetica', 24, 'bold'),
        'subtitle': ('Helvetica', 12),
        'label': ('Arial', 10),
        'input': ('Courier New', 10)
    }


    @classmethod
    def configure_style(cls):
        style = ttk.Style()
        style.theme_create('chem', parent='alt', settings={
            'TFrame': {'configure': {'background': cls.COLORS['background']}},
            'TLabel': {
                'configure': {
                    'foreground': cls.COLORS['foreground'],
                    'background': cls.COLORS['background'],
                    'font': cls.FONTS['label']
                }
            },
            'TButton': {
                'configure': {
                    'foreground': cls.COLORS['background'],
                    'background': cls.COLORS['accent1'],
                    'font': cls.FONTS['label']
                },
                'map': {
                    'background': [('active', cls.COLORS['highlight'])]
                }
            },
            'TEntry': {
                'configure': {
                    'fieldbackground': cls.COLORS['foreground'],
                    'font': cls.FONTS['input']
                }
            },
            'TCombobox': {
                'configure': {
                    'fieldbackground': cls.COLORS['foreground'],
                    'font': cls.FONTS['input']
                }
            }
        })
        style.theme_use('chem')

class BaseModeFrame(ttk.Frame):
    """Schema base class framework"""
    def __init__(self, master, mode_name):
        super().__init__(master)
        self.mode_name = mode_name
        self.file_selector = SDFFileSelector()
        self.create_widgets()
        self.db_path = 'chem_data.db'

    def create_widgets(self):
        file_frame = ttk.Frame(self)
        ttk.Button(file_frame, text="Select the SDF file", command=self.select_files).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT)
        file_frame.pack(pady=10, fill=tk.X)

        self.create_parameters()

        self.btn_frame = ttk.Frame(self)
        ttk.Button(self.btn_frame, text="Start the analysis", command=self.start_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.btn_frame, text="Reset parameter", command=self.reset_parameters).pack(side=tk.LEFT)
        self.btn_frame.pack(pady=10)

    def select_files(self):
        files = self.file_selector.select_files_via_gui()
        self.file_label.config(text="\n".join([Path(f).name for f in files]))

    def create_parameters(self):
        raise NotImplementedError("The parameter creation method must be implemented")

    def validate_input(self):
        raise NotImplementedError("An input validation method must be implemented")

    def start_analysis(self):
        if self.validate_input():
            self.run_analysis()

    def run_analysis(self):
        raise NotImplementedError("Analytical methods must be implemented")

    def reset_parameters(self):
        raise NotImplementedError("The reset method must be implemented")

    def show_results(self, results):
        pass


class CombinedModeFrame(BaseModeFrame):
    """Joint matching mode interface"""
    def __init__(self, master):
        super().__init__(master, "Joint Matching")

    def create_parameters(self):

        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas = tk.Canvas(main_container, height=300, highlightthickness=0, bd=0)
        vsb = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)


        vsb.pack(side="right", fill="y", padx=0)
        canvas.pack(side="left", fill="both", expand=True, padx=0)

        scroll_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        def _configure_canvas(e):
            canvas.itemconfig("all", width=e.width)
            canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.bind("<Configure>", _configure_canvas)
        carbon_frame = ttk.LabelFrame(scroll_frame, text="C parameters")
        self.create_carbon_params(carbon_frame)
        carbon_frame.pack(fill=tk.X, padx=5, pady=2, anchor=tk.NW)

        score_frame = ttk.LabelFrame(scroll_frame, text="Scoring parameters")
        self.create_score_params(score_frame)
        score_frame.pack(fill=tk.X, padx=5, pady=2, anchor=tk.NW)

        hmqc_frame = ttk.LabelFrame(scroll_frame, text="HSQC parameters")
        self.create_hmqc_params(hmqc_frame)
        hmqc_frame.pack(fill=tk.X, padx=5, pady=2, anchor=tk.NW)

        advanced_frame = ttk.LabelFrame(scroll_frame, text="C-HSQC weight")
        self.create_advanced_params(advanced_frame)
        advanced_frame.pack(fill=tk.X, padx=5, pady=2, anchor=tk.NW)


    def create_carbon_params(self, parent):
        mode_frame = ttk.Frame(parent)
        self.c_mode = tk.StringVar(value='typed')
        self.c_merge = tk.StringVar(value='N')
        ttk.Radiobutton(mode_frame, text="C_typed", variable=self.c_mode,
                        value='typed', command=self.toggle_c_params).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="C_untyped", variable=self.c_mode,
                        value='untyped', command=self.toggle_c_params).pack(side=tk.LEFT)
        mode_frame.pack(anchor=tk.W)
        ttk.Label(mode_frame, text=" C_merge:").pack(side=tk.LEFT, padx=(10,0))
        ttk.Radiobutton(mode_frame, text="Y", variable=self.c_merge,
                        value='Y').pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="N", variable=self.c_merge,
                        value='N').pack(side=tk.LEFT)

        self.param_container = ttk.Frame(parent)

        self.typed_params = {}
        typed_frame = ttk.Frame(self.param_container)
        labels = ['    All_C   ', '+Dept_135',' -Dept_135', '  Dept_90 ']
        for label in labels:
            row = ttk.Frame(typed_frame)
            ttk.Label(row, text=f"{label}:").pack(side=tk.LEFT)
            entry = tk.Text(row, height=2, width=60)
            entry.pack(side=tk.LEFT, padx=5)
            self.typed_params[label] = entry
            row.pack(anchor=tk.W)
        self.typed_frame = typed_frame

        untyped_frame = ttk.Frame(self.param_container)
        self.global_entry = tk.Text(untyped_frame, height=3, width=60)
        ttk.Label(untyped_frame, text="All_C:").pack(side=tk.LEFT)
        self.global_entry.pack(side=tk.LEFT, padx=5)
        self.untyped_frame = untyped_frame

        self.param_container.pack()
        self.toggle_c_params()

        range_frame = ttk.Frame(parent)
        ttk.Label(range_frame, text="C_num range:").pack(side=tk.LEFT)
        self.c_min = ttk.Entry(range_frame, width=5)
        self.c_min.insert(0, "5")
        self.c_min.pack(side=tk.LEFT)
        ttk.Label(range_frame, text="-").pack(side=tk.LEFT)
        self.c_max = ttk.Entry(range_frame, width=5)
        self.c_max.insert(0, "30")
        self.c_max.pack(side=tk.LEFT)

        ttk.Label(range_frame, text=" | ").pack(side=tk.LEFT, padx=5)

        ttk.Label(range_frame, text="MW range:").pack(side=tk.LEFT)
        self.m_min = ttk.Entry(range_frame, width=5)
        self.m_min.insert(0, "100")
        self.m_min.pack(side=tk.LEFT)
        ttk.Label(range_frame, text="-").pack(side=tk.LEFT)
        self.m_max = ttk.Entry(range_frame, width=5)
        self.m_max.insert(0, "500")
        self.m_max.pack(side=tk.LEFT)

        range_frame.pack(pady=5)

    def create_score_params(self, parent):
        env_frame = ttk.Frame(parent)
        ttk.Label(env_frame, text="Env atom level:").pack(side=tk.LEFT)
        self.env_level = ttk.Combobox(env_frame, values=[1, 2, 3], width=3)
        self.env_level.set(1)
        self.env_level.pack(side=tk.LEFT)

        # 权重设置
        ttk.Label(env_frame, text="Self weight:").pack(side=tk.LEFT, padx=(10,0))
        self.self_weight = ttk.Entry(env_frame, width=5)
        self.self_weight.insert(0, "0.7")
        self.self_weight.pack(side=tk.LEFT)

        ttk.Label(env_frame, text="Env weight:").pack(side=tk.LEFT, padx=(10,0))
        self.env_weight = ttk.Entry(env_frame, width=5, state='readonly')
        self.env_weight.config(state='normal')
        self.env_weight.insert(0, "0.3")
        self.env_weight.config(state='readonly')
        self.env_weight.pack(side=tk.LEFT)

        self.self_weight.bind("<KeyRelease>", self.update_env_weight)
        env_frame.pack(anchor=tk.W)

        # 评分模式
        self.score_mode = tk.StringVar(value='global')
        ttk.Radiobutton(parent, text="Global mode", variable=self.score_mode,
                        value='global', command=self.toggle_score_mode).pack(anchor=tk.W)
        ttk.Radiobutton(parent, text="Fine model", variable=self.score_mode,
                        value='fine', command=self.toggle_score_mode).pack(anchor=tk.W)

        self.global_frame = ttk.Frame(parent)
        ttk.Label(self.global_frame, text="Global threshold:").pack(side=tk.LEFT)
        self.green_thresh = ttk.Entry(self.global_frame, width=8)
        self.green_thresh.insert(0, "0.5")
        self.green_thresh.pack(side=tk.LEFT)
        ttk.Label(self.global_frame, text="-").pack(side=tk.LEFT)
        self.yellow_thresh = ttk.Entry(self.global_frame, width=8)
        self.yellow_thresh.insert(0, "2.0")
        self.yellow_thresh.pack(side=tk.LEFT)
        self.global_frame.pack(anchor=tk.W, fill=tk.X)
        self.fine_container = ttk.Frame(parent)
        self.canvas = tk.Canvas(self.fine_container, height=60)
        scrollbar = ttk.Scrollbar(self.fine_container, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)

        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.range_rows = []
        self.add_range_row((0, 220), 1, 3)

        btn_frame = ttk.Frame(self.fine_container)
        ttk.Button(btn_frame, text="+ Add range",
                   command=lambda: self.add_range_row((0, 0), 0, 0)).pack(pady=5)
        btn_frame.pack(fill=tk.X)

        self.fine_container.pack_forget()

    def update_env_weight(self, event=None):
        try:
            self_val = float(self.self_weight.get() or 0)  # Handle empty input
            env_val = max(0.0, min(1.0, 1 - self_val))  # Clamp between 0-1
            self.env_weight.config(state='normal')
            self.env_weight.delete(0, tk.END)
            self.env_weight.insert(0, f"{env_val:.2f}")
            self.env_weight.config(state='readonly')
        except ValueError:
            self.env_weight.config(state='normal')
            self.env_weight.delete(0, tk.END)
            self.env_weight.insert(0, "1.00")  # Default value
            self.env_weight.config(state='readonly')

    def toggle_score_mode(self):
        if self.score_mode.get() == 'global':
            self.global_frame.pack(anchor=tk.W, fill=tk.X)
            self.fine_container.pack_forget()
        else:
            self.global_frame.pack_forget()
            self.fine_container.pack(fill=tk.X)

    def add_range_row(self, carbon_range, green_val, yellow_val):
        if len(self.range_rows) >= 10:
            return

        frame = ttk.Frame(self.scroll_frame)
        ttk.Label(frame, text="C_range:").pack(side=tk.LEFT)
        start_entry = ttk.Entry(frame, width=6)
        start_entry.insert(0, str(carbon_range[0]))
        start_entry.pack(side=tk.LEFT)
        ttk.Label(frame, text="-").pack(side=tk.LEFT)
        end_entry = ttk.Entry(frame, width=6)
        end_entry.insert(0, str(carbon_range[1]))
        end_entry.pack(side=tk.LEFT)
        # 阈值
        ttk.Label(frame, text="Green threshold:").pack(side=tk.LEFT, padx=(10,0))
        green_entry = ttk.Entry(frame, width=6)
        green_entry.insert(0, str(green_val))
        green_entry.pack(side=tk.LEFT)
        ttk.Label(frame, text="Yellow threshold:").pack(side=tk.LEFT, padx=(10,0))
        yellow_entry = ttk.Entry(frame, width=6)
        yellow_entry.insert(0, str(yellow_val))
        yellow_entry.pack(side=tk.LEFT)
        # 删除按钮
        ttk.Button(frame, text="×", width=2,
                   command=lambda: self.remove_range_row(frame)).pack(side=tk.RIGHT, padx=5)
        self.range_rows.append({
            'frame': frame,
            'start': start_entry,
            'end': end_entry,
            'green': green_entry,
            'yellow': yellow_entry
        })
        frame.pack(fill=tk.X, pady=2)

    def remove_range_row(self, frame):
        for row in self.range_rows:
            if row['frame'] == frame:
                frame.destroy()
                self.range_rows.remove(row)
                break

    def toggle_c_params(self):
        """切换参数输入界面"""
        for widget in self.param_container.winfo_children():
            widget.pack_forget()

        if self.c_mode.get() == 'typed':
            self.typed_frame.pack(anchor=tk.W)
        else:
            self.untyped_frame.pack(anchor=tk.W)

    def create_hmqc_params(self, parent):
        mode_frame = ttk.Frame(parent)
        ttk.Label(mode_frame, text="Matching pattern:").pack(side=tk.LEFT)
        self.hmqc_mode = ttk.Combobox(mode_frame, values=[1,2,3], state='readonly')
        self.hmqc_mode.current(1)
        self.hmqc_mode.pack(side=tk.LEFT)
        self.hmqc_mode.bind("<<ComboboxSelected>>", lambda e: self.create_hmqc_input_groups())
        self.ch_merge = tk.StringVar(value='N')  # 默认值设为N
        ttk.Label(mode_frame, text=" | CH_merge:").pack(side=tk.LEFT, padx=(10,0))
        ttk.Radiobutton(mode_frame, text="Y", variable=self.ch_merge,
                        value='Y').pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="N", variable=self.ch_merge,
                        value='N').pack(side=tk.LEFT)
        mode_frame.pack(anchor=tk.W)

        self.hmqc_input_container = ttk.Frame(parent)
        self.hmqc_input_container.pack(fill=tk.X, padx=5, pady=2)
        self.point_groups = {}
        self.create_hmqc_input_groups()

        tol_frame = ttk.Frame(parent)
        ttk.Label(tol_frame, text="C tolerance:").pack(side=tk.LEFT)
        self.c_tol = ttk.Entry(tol_frame, width=8)
        self.c_tol.insert(0, "1.0")
        self.c_tol.pack(side=tk.LEFT)
        ttk.Label(tol_frame, text="H tolerance:").pack(side=tk.LEFT)
        self.h_tol = ttk.Entry(tol_frame, width=8)
        self.h_tol.insert(0, "0.2")
        self.h_tol.pack(side=tk.LEFT)
        tol_frame.pack(anchor=tk.W)

    def create_hmqc_input_groups(self):
        for widget in self.hmqc_input_container.winfo_children():
            widget.destroy()
        self.point_groups.clear()

        current_mode = int(self.hmqc_mode.get())
        types_config = {
            1: ['All_type'],
            2: ['type13', 'type2'],
            3: ['type1', 'type2', 'type3']
        }

        for t in types_config[current_mode]:
            group = ttk.LabelFrame(self.hmqc_input_container, text=t)
            entry = tk.Text(group, height=3, width=80)
            scroll = ttk.Scrollbar(group, command=entry.yview)
            entry.config(yscrollcommand=scroll.set)
            entry.pack(side=tk.LEFT)
            scroll.pack(side=tk.LEFT, fill=tk.Y)
            self.point_groups[t] = entry
            group.pack(fill=tk.X, padx=5, pady=2, side=tk.TOP)

    def create_advanced_params(self, parent):
        weight_frame = ttk.Frame(parent)
        ttk.Label(weight_frame, text="C weight:").pack(side=tk.LEFT)
        self.c_weight = ttk.Entry(weight_frame, width=8)
        self.c_weight.insert(0, "0.7")
        self.c_weight.pack(side=tk.LEFT)
        ttk.Label(weight_frame, text="CH weight:").pack(side=tk.LEFT)
        self.ch_weight = ttk.Entry(weight_frame, width=8)
        self.ch_weight.insert(0, "0.3")
        self.ch_weight.pack(side=tk.LEFT)
        weight_frame.pack(anchor=tk.W)

    def validate_input(self):
        try:
            if not self.file_selector.selected_files:
                raise ValueError("Please select the SDF file first")

            if self.c_mode.get() == 'typed':
                for label in ['    All_C   ', '+Dept_135',' -Dept_135', '  Dept_90 ']:
                    entry = self.typed_params[label]
                    text = entry.get("1.0", tk.END).strip() if isinstance(entry, tk.Text) else entry.get().strip()
                    if not text:
                        raise ValueError(f"{label.strip()} cannot be empty")
            else:
                text = self.global_entry.get("1.0", tk.END).strip().replace('，', ',')
                if not text:
                    raise ValueError("Global mode requires a list of displacements")

            if not (self.c_min.get().isdigit() and self.c_max.get().isdigit()):
                raise ValueError("The carbon number range must be an integer")

            current_mode = int(self.hmqc_mode.get())
            required_fields = {
                1: ['All_type'],
                2: ['type13', 'type2'],
                3: ['type1', 'type2', 'type3']
            }[current_mode]

            pattern = r'^\(\s*(?:\d+\.?\d*|\.\d+)\s*,\s*(?:\d+\.?\d*|\.\d+)\s*\)(?:[,\s]*\(\s*(?:\d+\.?\d*|\.\d+)\s*,\s*(?:\d+\.?\d*|\.\d+)\s*\))*$'

            for field in required_fields:
                text = self.point_groups[field].get("1.0", tk.END).strip()
                # 统一符号格式
                text = text.replace('，', ',').replace('（', '(').replace('）', ')')
                if not text:
                    raise ValueError(f"{field} cannot be empty")
                if not re.match(pattern, text):
                    raise ValueError(f"{field} format error, should be (numbers, numbers) list, support decimals, do not support negative numbers")
            return True
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return False

    def reset_parameters(self):

        self.c_mode.set('typed')
        self.toggle_c_params()

        for entry in self.typed_params.values():
            entry.delete("1.0", tk.END)

        self.global_entry.delete("1.0", tk.END)

        self.c_min.delete(0, tk.END)
        self.c_min.insert(0, "10")
        self.c_max.delete(0, tk.END)
        self.c_max.insert(0, "30")
        self.m_min.delete(0, tk.END)
        self.m_min.insert(0, "100")
        self.m_max.delete(0, tk.END)
        self.m_max.insert(0, "500")
        self.env_level.set(1)
        self.self_weight.delete(0, tk.END)
        self.self_weight.insert(0, "0.7")
        self.update_env_weight()
        self.score_mode.set('global')
        self.toggle_score_mode()
        self.green_thresh.delete(0, tk.END)
        self.green_thresh.insert(0, "0.5")
        self.yellow_thresh.delete(0, tk.END)
        self.yellow_thresh.insert(0, "2.0")

        for row in self.range_rows.copy():
            self.remove_range_row(row['frame'])
        self.add_range_row((0, 220), 1, 3)  # 添加默认范围

        self.hmqc_mode.current(1)
        self.create_hmqc_input_groups()
        self.c_tol.delete(0, tk.END)
        self.c_tol.insert(0, "1.0")
        self.h_tol.delete(0, tk.END)
        self.h_tol.insert(0, "0.2")

        self.c_weight.delete(0, tk.END)
        self.c_weight.insert(0, '0.7')
        self.ch_weight.delete(0, tk.END)
        self.ch_weight.insert(0, '0.3')

    def get_c_data(self):
        if self.c_mode.get() == 'typed':
            data = {
                label.strip(): [
                    float(x) for x in
                    (entry.get("1.0", tk.END).strip() if isinstance(entry, tk.Text) else entry.get().strip()
                     ).replace('，', ',')  # Replace Chinese commas with English commas
                    .split(',')]
                for label, entry in self.typed_params.items()
            }

            all_c = data['All_C']
            plus_dept = data['+Dept_135']
            minus_dept = data['-Dept_135']
            dept_90 = data['Dept_90']
            dept135 = plus_dept + minus_dept

            # Process Type 0 (All_C excluding matches with +Dept_135)
            exclude_all_c = set()
            for val in dept135:
                lower, upper = val - 0.1, val + 0.1
                for idx, c_val in enumerate(all_c):
                    if lower <= c_val <= upper and idx not in exclude_all_c:
                       exclude_all_c.add(idx)
                       break

            # Process Type 3 (+Dept_135 excluding matches with Dept_90)
            exclude_plus_dept = set()
            for val in dept_90:
                lower, upper = val - 0.1, val + 0.1
                for idx, p_val in enumerate(plus_dept):
                    if lower <= p_val <= upper and idx not in exclude_plus_dept:
                        exclude_plus_dept.add(idx)
                        break

            return {
                0: [v for i, v in enumerate(all_c) if i not in exclude_all_c],
                1: dept_90,
                2: minus_dept,
                3: [v for i, v in enumerate(plus_dept) if i not in exclude_plus_dept]
            }
        else:
            text = self.global_entry.get("1.0", tk.END).strip().replace('，', ',')  # 处理中文逗号
            values = [x.strip() for x in text.split(',') if x.strip()]  # 分割并过滤空值
            return [float(x) for x in values]

    def process_carbon_scoring_config(self):
        scoring_config = {}

        scoring_mode = str(self.score_mode.get())
        scoring_config['carbon_score_mode'] = scoring_mode

        if scoring_mode == 'global':
            scoring_config['carbon_global_thresholds'] = (
                float(self.green_thresh.get()),
                float(self.yellow_thresh.get())
            )
        elif scoring_mode == 'fine':
            fine_ranges = []
            for row in self.range_rows:
                try:
                    c_start = float(row['start'].get())
                    c_end = float(row['end'].get())
                    green_val = float(row['green'].get())
                    yellow_val = float(row['yellow'].get())
                    fine_ranges.append((
                        (c_start, c_end),
                        green_val,
                        yellow_val
                    ))
                except ValueError:
                    continue
            scoring_config['fine_ranges'] = fine_ranges

        return scoring_config

    def run_analysis(self):
        env_params = (
            int(self.env_level.get()),
            float(self.self_weight.get()),
            float(self.env_weight.get())
        )

        config = {
            'sdf_files': self.file_selector.selected_files,
            'c_mode': str(self.c_mode.get()),
             'C_merge':str(self.c_merge.get()),
            'c_data': self.get_c_data(),
            'c_range': tuple(map(int, (self.c_min.get(), self.c_max.get()))),
            'fw_range': tuple(map(float, (self.m_min.get(), self.m_max.get()))),
            'env_params': env_params,
            'ch_mode': int(self.hmqc_mode.get()),
            'CH_merge':str(self.ch_merge.get()),
            'ch_data': self.get_ch_data(),
            'ch_tolerances': tuple(map(float, (self.c_tol.get(), self.h_tol.get()))),
            'final_weights': tuple(map(float, (self.c_weight.get(), self.ch_weight.get())))
        }

        scoring_config = self.process_carbon_scoring_config()
        config.update(scoring_config)

        # try:
        pipeline = CombinedScorerGUI(**config)
        results = pipeline.execute()
        pipeline._generate_plots(output_dir="combined_results")

        viewer = CombinedResultViewer(
            db_path="chem_data.db",
            results=results,
            output_dir="combined_results"
        )
        viewer.create_result_window(self.master)
        #
        # except Exception as e:
        #     messagebox.showerror("Analysis error", f"An error occurred while performing the analysis:\n{str(e)}")
    def get_ch_data(self):

        ch_data = {}
        current_mode = int(self.hmqc_mode.get())
        types_config = {
            1: ['All_type'],
            2: ['type13', 'type2'],
            3: ['type1', 'type2', 'type3']
        }[current_mode]

        for t in types_config:
            text = self.point_groups[t].get("1.0", tk.END).strip()

            text = (text.replace('（', '(')
                    .replace('）', ')')
                    .replace('，', ','))

            matches = re.findall(r'\(([\d.]+),\s*([\d.]+)\)', text)
            ch_data[t] = [[float(m[0]), float(m[1])] for m in matches]

        return ch_data



from pathlib import Path
import sqlite3

class CarbonOnlyModeFrame(BaseModeFrame):

    def __init__(self, master):
        super().__init__(master, "C matching mode")

    def create_parameters(self):
        carbon_frame = ttk.LabelFrame(self, text="C parameters")
        mode_frame = ttk.Frame(carbon_frame)
        self.c_mode = tk.StringVar(value='typed')
        self.c_merge = tk.StringVar(value='N')
        ttk.Radiobutton(mode_frame, text="C_typed", variable=self.c_mode,
                        value='typed', command=self.toggle_c_params).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="C_untype", variable=self.c_mode,
                        value='untyped', command=self.toggle_c_params).pack(side=tk.LEFT)
        ttk.Label(mode_frame, text=" C_merge:").pack(side=tk.LEFT, padx=(10,0))
        ttk.Radiobutton(mode_frame, text="Y", variable=self.c_merge,
                        value='Y').pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="N", variable=self.c_merge,
                        value='N').pack(side=tk.LEFT)
        mode_frame.pack(anchor=tk.W)
        self.param_container = ttk.Frame(carbon_frame)
        self.typed_params = {}
        typed_frame = ttk.Frame(self.param_container)
        labels = ['   All_C    ', '+Dept_135','-Dept_135', '  Dept_90 ']
        for label in labels:
            row = ttk.Frame(typed_frame)
            ttk.Label(row, text=f"{label}:").pack(side=tk.LEFT)
            entry = tk.Text(row, height=2, width=60)
            entry.pack(side=tk.LEFT, padx=5)
            self.typed_params[label] = entry
            row.pack(anchor=tk.W)
        self.typed_frame = typed_frame
        untyped_frame = ttk.Frame(self.param_container)
        self.global_entry = tk.Text(untyped_frame, height=3, width=60)
        ttk.Label(untyped_frame, text="All_C:").pack(side=tk.LEFT)
        self.global_entry.pack(side=tk.LEFT, padx=5)
        self.untyped_frame = untyped_frame

        self.param_container.pack()
        self.toggle_c_params()
        range_frame = ttk.Frame(carbon_frame)
        ttk.Label(range_frame, text="C_num range:").pack(side=tk.LEFT)
        self.c_min = ttk.Entry(range_frame, width=5)
        self.c_min.insert(0, "5")
        self.c_min.pack(side=tk.LEFT)
        ttk.Label(range_frame, text="-").pack(side=tk.LEFT)
        self.c_max = ttk.Entry(range_frame, width=5)
        self.c_max.insert(0, "30")
        self.c_max.pack(side=tk.LEFT)

        ttk.Label(range_frame, text=" | ").pack(side=tk.LEFT, padx=5)
        ttk.Label(range_frame, text=":MW range").pack(side=tk.LEFT)
        self.m_min = ttk.Entry(range_frame, width=5)
        self.m_min.insert(0, "100")
        self.m_min.pack(side=tk.LEFT)
        ttk.Label(range_frame, text="-").pack(side=tk.LEFT)
        self.m_max = ttk.Entry(range_frame, width=5)
        self.m_max.insert(0, "500")
        self.m_max.pack(side=tk.LEFT)

        range_frame.pack(pady=5)

        carbon_frame.pack(fill=tk.X, padx=10, pady=5)


        score_frame = ttk.LabelFrame(self, text="Scoring parameters")


        env_frame = ttk.Frame(score_frame)
        ttk.Label(env_frame, text="Env atom level:").pack(side=tk.LEFT)
        self.env_level = ttk.Combobox(env_frame, values=[1, 2, 3], width=3)
        self.env_level.set(1)
        self.env_level.pack(side=tk.LEFT)

        # 权重设置
        ttk.Label(env_frame, text="Self weight:").pack(side=tk.LEFT, padx=(10,0))
        self.self_weight = ttk.Entry(env_frame, width=5)
        self.self_weight.insert(0, "0.7")
        self.self_weight.pack(side=tk.LEFT)

        ttk.Label(env_frame, text="Env weight:").pack(side=tk.LEFT, padx=(10,0))
        self.env_weight = ttk.Entry(env_frame, width=5, state='readonly')
        self.env_weight.config(state='normal')
        self.env_weight.insert(0, "0.3")
        self.env_weight.config(state='readonly')
        self.env_weight.pack(side=tk.LEFT)

        self.self_weight.bind("<KeyRelease>", self.update_env_weight)
        env_frame.pack(anchor=tk.W)
        self.score_mode = tk.StringVar(value='global')
        ttk.Radiobutton(score_frame, text="Global mode", variable=self.score_mode,
                        value='global', command=self.toggle_score_mode).pack(anchor=tk.W)
        ttk.Radiobutton(score_frame, text="Fine model", variable=self.score_mode,
                        value='fine', command=self.toggle_score_mode).pack(anchor=tk.W)


        self.global_frame = ttk.Frame(score_frame)
        ttk.Label(self.global_frame, text="Global threshold:").pack(side=tk.LEFT)
        self.green_thresh = ttk.Entry(self.global_frame, width=8)
        self.green_thresh.insert(0, "0.5")
        self.green_thresh.pack(side=tk.LEFT)
        ttk.Label(self.global_frame, text="-").pack(side=tk.LEFT)
        self.yellow_thresh = ttk.Entry(self.global_frame, width=8)
        self.yellow_thresh.insert(0, "2.0")
        self.yellow_thresh.pack(side=tk.LEFT)
        self.global_frame.pack(anchor=tk.W, fill=tk.X)

        self.fine_container = ttk.Frame(score_frame)

        self.canvas = tk.Canvas(self.fine_container, height=60)
        scrollbar = ttk.Scrollbar(self.fine_container, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)

        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.range_rows = []
        self.add_range_row((0, 220), 1, 3)  # 初始行

        btn_frame = ttk.Frame(self.fine_container)
        add_btn = ttk.Button(btn_frame, text="+ Add range",
                             command=lambda: self.add_range_row((0, 0), 0, 0))
        add_btn.pack(pady=5)
        btn_frame.pack(fill=tk.X)

        self.fine_container.pack_forget()
        score_frame.pack(fill=tk.X, padx=10, pady=5)

    def update_env_weight(self, event=None):
        try:
            self_val = float(self.self_weight.get() or 0)  # Handle empty input
            env_val = max(0.0, min(1.0, 1 - self_val))  # Clamp between 0-1
            self.env_weight.config(state='normal')
            self.env_weight.delete(0, tk.END)
            self.env_weight.insert(0, f"{env_val:.2f}")
            self.env_weight.config(state='readonly')
        except ValueError:
            self.env_weight.config(state='normal')
            self.env_weight.delete(0, tk.END)
            self.env_weight.insert(0, "1.00")  # Default value
            self.env_weight.config(state='readonly')


    def toggle_score_mode(self):
        if self.score_mode.get() == 'global':
            self.global_frame.pack(anchor=tk.W, fill=tk.X)
            self.fine_container.pack_forget()
        else:
            self.global_frame.pack_forget()
            self.fine_container.pack(fill=tk.X)

    def add_range_row(self, carbon_range, green_val, yellow_val):
        if len(self.range_rows) >= 10:
            return

        frame = ttk.Frame(self.scroll_frame)

        ttk.Label(frame, text="C_range:").pack(side=tk.LEFT)
        start_entry = ttk.Entry(frame, width=6)
        start_entry.insert(0, str(carbon_range[0]))
        start_entry.pack(side=tk.LEFT)
        ttk.Label(frame, text="-").pack(side=tk.LEFT)
        end_entry = ttk.Entry(frame, width=6)
        end_entry.insert(0, str(carbon_range[1]))
        end_entry.pack(side=tk.LEFT)

        # 阈值
        ttk.Label(frame, text="Green threshold:").pack(side=tk.LEFT, padx=(10,0))
        green_entry = ttk.Entry(frame, width=6)
        green_entry.insert(0, str(green_val))
        green_entry.pack(side=tk.LEFT)

        ttk.Label(frame, text="Yellow threshold:").pack(side=tk.LEFT, padx=(10,0))
        yellow_entry = ttk.Entry(frame, width=6)
        yellow_entry.insert(0, str(yellow_val))
        yellow_entry.pack(side=tk.LEFT)

        del_btn = ttk.Button(frame, text="×", width=2,
                             command=lambda: self.remove_range_row(frame))
        del_btn.pack(side=tk.RIGHT, padx=5)

        self.range_rows.append({
            'frame': frame,
            'start': start_entry,
            'end': end_entry,
            'green': green_entry,
            'yellow': yellow_entry
        })
        frame.pack(fill=tk.X, pady=2)

    def remove_range_row(self, frame):
        for row in self.range_rows:
            if row['frame'] == frame:
                frame.destroy()
                self.range_rows.remove(row)
                break
    def toggle_c_params(self):
        for widget in self.param_container.winfo_children():
            widget.pack_forget()

        if self.c_mode.get() == 'typed':
            self.typed_frame.pack(anchor=tk.W)
        else:
            self.untyped_frame.pack(anchor=tk.W)


    def validate_input(self):
        try:
            if not self.file_selector.selected_files:
                raise ValueError("Please select the SDF file first")

            if self.c_mode.get() == 'typed':
                for label in ['   All_C    ', '+Dept_135','-Dept_135', '  Dept_90 ']:
                    text = self.typed_params[label].get("1.0", tk.END).strip()
                    if not text:
                        raise ValueError(f"{label.strip()} cannot be empty")
            else:
                text = self.global_entry.get("1.0", tk.END).strip().replace('，', ',')
                if not text:
                    raise ValueError("Global mode requires a list of displacements")
            if not (self.c_min.get().isdigit() and self.c_max.get().isdigit()):
                raise ValueError("The carbon number range must be an integer")
            return True
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return False

    def process_carbon_scoring_config(self):
        scoring_config = {}

        scoring_mode = str(self.score_mode.get())
        scoring_config['score_mode'] = scoring_mode

        if scoring_mode == 'global':
            scoring_config['global_thresholds'] = (
                float(self.green_thresh.get()),
                float(self.yellow_thresh.get())
            )
        elif scoring_mode == 'fine':
            fine_ranges = []
            for row in self.range_rows:
                try:
                    c_start = float(row['start'].get())
                    c_end = float(row['end'].get())
                    green_val = float(row['green'].get())
                    yellow_val = float(row['yellow'].get())
                    fine_ranges.append((
                        (c_start, c_end),
                        green_val,
                        yellow_val
                    ))
                except ValueError:
                    continue
            scoring_config['fine_ranges'] = fine_ranges

        return scoring_config

    def run_analysis(self):

        config = {
            'sdf_files': self.file_selector.selected_files,
            'c_mode': str(self.c_mode.get()),
            'C_merge': str(self.c_merge.get()),
            'c_data': self.get_c_data(),
            'c_range': tuple(map(int, (self.c_min.get(), self.c_max.get()))),
            'fw_range': tuple(map(float, (self.m_min.get(), self.m_max.get()))),
            'env_level':int(self.env_level.get()),
            'self_weight':float(self.self_weight.get()),
            'env_weight': float(self.env_weight.get())
        }

        scoring_config = self.process_carbon_scoring_config()
        config.update(scoring_config)

        try:
            pipeline = CarbonOnlyScorerGUI(**config)
            results = pipeline.execute()
            visualizer = CarbonResultVisualizer(
                db_path="chem_data.db",
                top_results=results
            )
            visualizer.generate_plots(output_dir="molecule_plots")

            # 显示结果窗口
            ResultViewer("chem_data.db", results).create_result_window(self.master)

        except Exception as e:
            messagebox.showerror("Analysis error", str(e))

    def get_c_data(self):
        if self.c_mode.get() == 'typed':
            data = {
                label.strip(): [
                    float(x) for x in
                    (entry.get("1.0", tk.END).strip() if isinstance(entry, tk.Text) else entry.get().strip()
                     ).replace('，', ',')  # Replace Chinese commas with English commas
                    .split(',')]
                for label, entry in self.typed_params.items()
            }

            all_c = data['All_C']
            plus_dept = data['+Dept_135']
            minus_dept = data['-Dept_135']
            dept_90 = data['Dept_90']
            dept135 = plus_dept + minus_dept

            # Process Type 0 (All_C excluding matches with +Dept_135)
            exclude_all_c = set()
            for val in dept135:
                lower, upper = val - 0.1, val + 0.1
                for idx, c_val in enumerate(all_c):
                    if lower <= c_val <= upper and idx not in exclude_all_c:
                        exclude_all_c.add(idx)
                        break

            # Process Type 3 (+Dept_135 excluding matches with Dept_90)
            exclude_plus_dept = set()
            for val in dept_90:
                lower, upper = val - 0.1, val + 0.1
                for idx, p_val in enumerate(plus_dept):
                    if lower <= p_val <= upper and idx not in exclude_plus_dept:
                        exclude_plus_dept.add(idx)
                        break

            return {
                0: [v for i, v in enumerate(all_c) if i not in exclude_all_c],
                1: dept_90,
                2: minus_dept,
                3: [v for i, v in enumerate(plus_dept) if i not in exclude_plus_dept]
            }
        else:
            text = self.global_entry.get("1.0", tk.END).strip().replace('，', ',')  # 处理中文逗号
            values = [x.strip() for x in text.split(',') if x.strip()]  # 分割并过滤空值
            return [float(x) for x in values]
class CHMatchModeFrame(BaseModeFrame):

    def __init__(self, master):
        super().__init__(master, "CH matching mode")

    def create_parameters(self):
        hmqc_frame = ttk.LabelFrame(self, text="HSQC parameters")

        mode_frame = ttk.Frame(hmqc_frame)
        ttk.Label(mode_frame, text="Matching pattern:").pack(side=tk.LEFT)
        self.hmqc_mode = ttk.Combobox(mode_frame, values=[1,2,3], state='readonly')
        self.hmqc_mode.current(1)
        self.hmqc_mode.pack(side=tk.LEFT)
        self.hmqc_mode.bind("<<ComboboxSelected>>", lambda e: self.create_input_groups())
        self.ch_merge = tk.StringVar(value='N')  # 默认值设为N
        ttk.Label(mode_frame, text=" | CH_merge:").pack(side=tk.LEFT, padx=(10,0))
        ttk.Radiobutton(mode_frame, text="Y", variable=self.ch_merge,
                        value='Y').pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="N", variable=self.ch_merge,
                        value='N').pack(side=tk.LEFT)
        mode_frame.pack(anchor=tk.W)

        self.input_container = ttk.Frame(hmqc_frame)
        self.input_container.pack(fill=tk.X, padx=5, pady=2)

        self.point_groups = {}
        self.create_input_groups()

        tol_frame = ttk.Frame(hmqc_frame)
        ttk.Label(tol_frame, text="C tolerance:").pack(side=tk.LEFT)
        self.c_tol = ttk.Entry(tol_frame, width=8)
        self.c_tol.insert(0, "1.0")
        self.c_tol.pack(side=tk.LEFT)
        ttk.Label(tol_frame, text="H tolerance:").pack(side=tk.LEFT)
        self.h_tol = ttk.Entry(tol_frame, width=8)
        self.h_tol.insert(0, "0.2")
        self.h_tol.pack(side=tk.LEFT)
        tol_frame.pack(anchor=tk.W)

        hmqc_frame.pack(fill=tk.X, padx=10, pady=5)

    def create_input_groups(self):

        for widget in self.input_container.winfo_children():
            widget.destroy()
        self.point_groups.clear()

        current_mode = int(self.hmqc_mode.get())
        types_config = {
            1: ['All_type'],
            2: ['type13', 'type2'],
            3: ['type1', 'type2', 'type3']
        }

        for t in types_config[current_mode]:
            group = ttk.LabelFrame(self.input_container, text=t)
            entry = tk.Text(group, height=3, width=40)
            scroll = ttk.Scrollbar(group, command=entry.yview)
            entry.config(yscrollcommand=scroll.set)
            entry.pack(side=tk.LEFT)
            scroll.pack(side=tk.LEFT, fill=tk.Y)
            self.point_groups[t] = entry
            group.pack(fill=tk.X, padx=5, pady=2, side=tk.TOP)

    def validate_input(self):
        try:
            if not self.file_selector.selected_files:
                raise ValueError("Please select the SDF file first")

            current_mode = int(self.hmqc_mode.get())
            required_fields = {
                1: ['All_type'],
                2: ['type13', 'type2'],
                3: ['type1', 'type2', 'type3']
            }[current_mode]

            pattern = r'^\(\s*(?:\d+\.?\d*|\.\d+)\s*,\s*(?:\d+\.?\d*|\.\d+)\s*\)(?:[,\s]*\(\s*(?:\d+\.?\d*|\.\d+)\s*,\s*(?:\d+\.?\d*|\.\d+)\s*\))*$'

            for field in required_fields:
                text = self.point_groups[field].get("1.0", tk.END).strip()

                text = text.replace('，', ',').replace('（', '(').replace('）', ')')
                if not text:
                    raise ValueError(f"{field} cannot be empty")
                if not re.match(pattern, text):
                    raise ValueError(f"{field} format error, should be (numbers, numbers) list, support decimals, do not support negative numbers")
            return True
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return False

    def run_analysis(self):
            config = {
            'sdf_files': self.file_selector.selected_files,
            'ch_mode': int(self.hmqc_mode.get()),
                'CH_merge': str(self.ch_merge.get()),
                'ch_data': self.get_ch_data(),
            'tolerances': tuple(map(float, (self.c_tol.get(), self.h_tol.get()))),
        }

            try:

                pipeline = CHOnlyScorerGUI(**config)
                results = pipeline.execute()
                visualizer = CHMatchVisualizer(
                db_path="chem_data.db",
                top_results=results,
                mode=config['ch_mode']
                )
                results = pipeline.execute()
                viewer = CHResultViewer(db_path="chem_data.db", results=results)
                viewer.create_result_window(self.master)
                visualizer.generate_plots()
            except Exception as e:
                messagebox.showerror("Analysis error", str(e))

    def get_ch_data(self):
        ch_data = {}
        current_mode = int(self.hmqc_mode.get())
        types_config = {
            1: ['All_type'],
            2: ['type13', 'type2'],
            3: ['type1', 'type2', 'type3']
        }[current_mode]

        for t in types_config:
            text = self.point_groups[t].get("1.0", tk.END).strip()
            text = (text.replace('（', '(')
                    .replace('）', ')')
                    .replace('，', ','))
            matches = re.findall(r'\(([\d.]+),\s*([\d.]+)\)', text)
            ch_data[t] = [[float(m[0]), float(m[1])] for m in matches]

        return ch_data

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("ReCQC 2.0 - Natural product analysis based on C spectrum and HSQC")
        self.geometry("800x610")
        ChemTheme.configure_style()
        self.create_ui()

    def create_ui(self):

        header = ttk.Frame(self)
        ttk.Label(header, text="ReCQC 2.0", font=ChemTheme.FONTS['title'],
                  foreground=ChemTheme.COLORS['highlight']).pack(pady=5)
        ttk.Label(header, text="Analysis of natural product mixtures based on C spectrum and HSQC spectrum",
                  font=ChemTheme.FONTS['subtitle']).pack()
        header.pack(pady=20)

        self.mode_frames = {}
        nav_frame = ttk.Frame(self)
        modes = [
            ('Joint matching mode', CombinedModeFrame),
            ('C matching mode', CarbonOnlyModeFrame),
            ('CH matching mode', CHMatchModeFrame)
        ]

        for text, frame_class in modes:
            btn = ttk.Button(nav_frame, text=text,
                             command=lambda fc=frame_class: self.show_mode(fc))
            btn.pack(side=tk.LEFT, padx=10, ipadx=20, ipady=5)
            self.mode_frames[text] = frame_class(self)

        nav_frame.pack(pady=10)
        self.show_mode(CombinedModeFrame)

    def show_mode(self, frame_class):
        if hasattr(self, 'current_frame'):
            self.current_frame.pack_forget()

        for name, frame in self.mode_frames.items():
            if isinstance(frame, frame_class):
                self.current_frame = frame
                self.current_frame.pack(fill=tk.BOTH, expand=True)
                break

if __name__ == "__main__":
    app = App()
    app.mainloop()

