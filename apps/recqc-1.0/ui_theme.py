import tkinter as tk
from tkinter import ttk


COLORS = {
    "background": "#F4F7FB",
    "surface": "#FFFFFF",
    "primary": "#185ADB",
    "primary_hover": "#1247AD",
    "text": "#172033",
    "muted": "#667085",
    "border": "#D8E0EC",
    "success": "#16794A",
}


def configure_theme(root):
    """Configure the shared ReCQC 1.0 desktop visual system."""
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    root.configure(background=COLORS["background"])
    root.option_add("*Font", ("Segoe UI", 10))
    root.option_add("*tearOff", False)

    style.configure("TFrame", background=COLORS["background"])
    style.configure("Surface.TFrame", background=COLORS["surface"])
    style.configure("Header.TFrame", background="#102A56")
    style.configure("TLabel", background=COLORS["background"], foreground=COLORS["text"])
    style.configure("Surface.TLabel", background=COLORS["surface"], foreground=COLORS["text"])
    style.configure(
        "Header.TLabel",
        background="#102A56",
        foreground="#FFFFFF",
        font=("Segoe UI Semibold", 15),
    )
    style.configure(
        "Title.TLabel",
        background=COLORS["background"],
        foreground=COLORS["text"],
        font=("Segoe UI Semibold", 26),
    )
    style.configure(
        "Subtitle.TLabel",
        background=COLORS["background"],
        foreground=COLORS["muted"],
        font=("Segoe UI", 11),
    )
    style.configure(
        "Section.TLabel",
        background=COLORS["surface"],
        foreground=COLORS["text"],
        font=("Segoe UI Semibold", 13),
    )
    style.configure("Muted.TLabel", background=COLORS["surface"], foreground=COLORS["muted"])
    style.configure(
        "Score.TLabel",
        background=COLORS["surface"],
        foreground=COLORS["primary"],
        font=("Segoe UI Semibold", 12),
    )

    style.configure(
        "TButton",
        padding=(14, 8),
        background="#E8EEF8",
        foreground=COLORS["text"],
        borderwidth=0,
        focusthickness=0,
    )
    style.map(
        "TButton",
        background=[("active", "#DCE6F5"), ("disabled", "#EEF1F6")],
        foreground=[("disabled", "#98A2B3")],
    )
    style.configure(
        "Primary.TButton",
        background=COLORS["primary"],
        foreground="#FFFFFF",
        font=("Segoe UI Semibold", 10),
    )
    style.map(
        "Primary.TButton",
        background=[("active", COLORS["primary_hover"]), ("disabled", "#A8BDE8")],
        foreground=[("disabled", "#F2F4F7")],
    )
    style.configure(
        "Header.TButton",
        background="#102A56",
        foreground="#DDE7F8",
        padding=(12, 7),
        borderwidth=0,
    )
    style.map(
        "Header.TButton",
        background=[("active", "#1C3C70")],
        foreground=[("active", "#FFFFFF")],
    )
    style.configure(
        "TEntry",
        fieldbackground=COLORS["surface"],
        foreground=COLORS["text"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["border"],
        darkcolor=COLORS["border"],
        padding=7,
    )
    style.configure(
        "TCheckbutton",
        background=COLORS["surface"],
        foreground=COLORS["text"],
        padding=(2, 4),
    )
    style.map("TCheckbutton", background=[("active", COLORS["surface"])])
    style.configure("TRadiobutton", background=COLORS["surface"], foreground=COLORS["text"])
    style.map("TRadiobutton", background=[("active", COLORS["surface"])])
    style.configure(
        "TLabelframe",
        background=COLORS["surface"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["border"],
        darkcolor=COLORS["border"],
        borderwidth=1,
        relief="solid",
        padding=18,
    )
    style.configure(
        "TLabelframe.Label",
        background=COLORS["surface"],
        foreground=COLORS["text"],
        font=("Segoe UI Semibold", 12),
    )
    style.configure("TNotebook", background=COLORS["background"], borderwidth=0)
    style.configure("TNotebook.Tab", padding=(14, 8))
    return style


def make_text(parent, height=5, width=60):
    return tk.Text(
        parent,
        height=height,
        width=width,
        wrap="word",
        undo=True,
        font=("Consolas", 10),
        background=COLORS["surface"],
        foreground=COLORS["text"],
        insertbackground=COLORS["text"],
        highlightthickness=1,
        highlightbackground=COLORS["border"],
        highlightcolor=COLORS["primary"],
        relief="flat",
        padx=8,
        pady=7,
    )
