import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import ColorIdentify as colIde


class MyGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Image Viewer")
        self.label_len = 0
        self.input = None
        self.filename = None
        self.square = None
        self.original = None
        self.original_label = None
        self.roi = None
        self.label_roi = None

        self.fig, self.axs = plt.subplots(1, 3)
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a menu bar with a "File" menu
        menubar = tk.Menu(self.window)
        self.window.config(menu=menubar)
        filemenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="image", command=self.load_image)

        right_frame = tk.Frame(self.window)
        right_frame.pack(side="right")
        # Create scrollable frame
        scroll_frame = tk.Frame(right_frame)
        scroll_frame.pack(fill="both", expand=True)

        self.canvas1 = tk.Canvas(scroll_frame)
        self.scrollbar = tk.Scrollbar(scroll_frame, orient="vertical", command=self.canvas1.yview)
        self.scrollable_frame = tk.Frame(self.canvas1)

    def load_image(self):
        self.axs[0].clear()
        self.axs[1].clear()
        self.axs[2].clear()
        self.filename = filedialog.askopenfilename(title="Select Image", filetypes=(
            ("jpeg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")))
        if not self.filename:
            return
        else:
            self.original = plt.imread(self.filename)
            self.square = self.original.copy()
            self.input = colIde.Input(imgPath=self.filename)
            self.label_len = self.input.label_len
        self.axs[0].imshow(self.original)
        self.canvas.draw()

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas1.configure(
                scrollregion=self.canvas1.bbox("all")
            )
        )

        self.canvas1.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas1.configure(yscrollcommand=self.scrollbar.set,width=150, height=700)

        # Add buttons to scrollable frame
        for i in range(self.label_len-1):
            tk.Button(self.scrollable_frame, text=f"region {i + 1}", width=10, height=2, font=("Arial", 14),
                      command=lambda idx=i + 1: self.local_identify(idx)).pack(pady=5, padx=10)

        # Pack scrollbar and canvas
        self.scrollbar.pack(side="right", fill="y")
        self.canvas1.pack(side="left", fill="both", expand=True)

    def local_identify(self, idx):
        self.axs[1].clear()
        self.roi, self.label_roi, square = self.input.process(idx)
        self.axs[0].imshow(square)
        self.axs[0].set_title("Whole image")
        self.axs[1].imshow(self.roi)
        self.axs[1].set_title(f"region of interest: region {idx}")
        self.axs[2].imshow(self.label_roi)
        self.axs[2].set_title("region of interest segment")
        self.canvas.draw()
