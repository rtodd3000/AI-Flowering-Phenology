import os
import csv
import customtkinter
from tkinter import ttk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from plot import FloweringPlot
from predict import predict_folder, predict_single

INTENSITY_CLASSES = [
    "0 - No Flowers",
    "1 - Few Bunches",
    "2 - Transition",
    "3 - Full Bloom"
]

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.file_path = ""
        self.images = []
        self.predicted_flower = ""
        self.mode = "Directory"
        self.data = []

        self.title("Flowering Analyzer")
        self.geometry("1200x600")

        # Header Frame
        self.header_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(pady=5, padx=20, fill="x")
        self.welcome_label = customtkinter.CTkLabel(self.header_frame, text="Welcome to the Flowering Analyzer!", height=20, font=("Roboto", 25, "bold"))
        self.welcome_label.grid(row=0, column=0, sticky="w")
        self.author_label = customtkinter.CTkLabel(self.header_frame, text="Made By Ryan Todd, Corvey Lee, Eden Parungao, Javin Solmirin", height=10, font=("Roboto", 15, "bold"))
        self.author_label.grid(row=1, column=0, sticky="w", padx=10, pady=3)

        # Body Frame
        self.body_frame = customtkinter.CTkFrame(self)
        self.body_frame.pack(pady=(5, 0), padx=20, fill="x")
        self.body_frame.grid_columnconfigure(1, weight=1)
         # Segmented Button
        self.select_segment = customtkinter.CTkSegmentedButton(master=self.body_frame, values=["Directory", "Image", "CSV"], command=self.select_segment_action, corner_radius=20)
        self.select_segment.set("Directory")
        self.select_segment.grid(row=0, column=0)
        
         # File Selection
        self.user_image_label = customtkinter.CTkLabel(self.body_frame, text="Your selected directory: ", justify="left")
        self.user_image_label.grid(row=1, column=0, sticky="w", padx=(10, 0), pady=(5, 0))
        self.path_label = customtkinter.CTkLabel(self.body_frame, width=300, text="No directory selected", anchor="w")
        self.path_label.grid(row=1, column=1, padx=10, pady=(5, 0), sticky="w")
        self.select_file_button = customtkinter.CTkButton(self.body_frame, text="Select Directory", height = 20, command=self.select_file)
        self.select_file_button.grid(row=1, column=2, padx=10, pady=(5, 0), sticky="e")

         # Confirmation 
        self.confirm_text = customtkinter.CTkLabel(self.body_frame, text="0 Files Found")
        self.confirm_text.grid(row=2, column=1, padx=10, sticky="w")
        self.confirm_button = customtkinter.CTkButton(self.body_frame, text="Start Analysis", height= 20, fg_color="green", hover_color="darkgreen", font=("Roboto", 10, "bold"), command=self.confirm_selection)
        self.confirm_button.grid(row=2, column=2, padx=10, sticky="e")
        self.confirm_button.grid_remove() # Hide the button until a file is selected
        self.confirm_text.grid_remove() # Hide the text until a file is selected

         # Prediction
        self.prediction_frame = customtkinter.CTkFrame(self)
        self.prediction_frame.pack(padx=20,pady=(5, 0), fill="x")
        self.prediction_label = customtkinter.CTkLabel(self.prediction_frame, text="Predicted Flower: ", font=("Roboto", 20, "bold"))
        self.prediction_label.pack(pady=10)
        self.prediction_frame.pack_forget()

        # Body Frame - Data
        self.body_data_frame = customtkinter.CTkFrame(self)
        self.body_data_frame.pack(fill="both", expand=True, padx=20, pady=(5, 0))

        # Body Loading
        self.loading = customtkinter.CTkLabel(self.body_data_frame, text="Loading...\nThis may take up to 5 minutes.\nPlease ignore any not responding.", font=("Helvetica", 30), fg_color="transparent")

        # Body Data
        self.body_data = customtkinter.CTkFrame(self.body_data_frame, fg_color="transparent")
        self.body_data.grid_columnconfigure(0, weight=1)  # left frame gets 1 part
        self.body_data.grid_columnconfigure(1, weight=3)  # right frame gets 3 parts
        self.body_data.grid_rowconfigure(0, weight=1)
        self.body_data.pack(fill="both", expand=True)
        self.body_data.pack_forget()    

         # Table Frame
        self.table_frame = customtkinter.CTkFrame(self.body_data, fg_color="transparent")
        self.table_frame.grid(row=0, column=0, sticky="nsew")

           # Scrollbar
        self.scrollbar = customtkinter.CTkScrollbar(self.table_frame)
        self.scrollbar.pack(side="right", fill="y")

           # Treeview Table
        self.data_table = ttk.Treeview(self.table_frame, columns=("Image Name", "Date", "Intensity", "Confidence", "Site", "Flower Type"), show="headings", selectmode="browse", yscrollcommand= self.scrollbar.set)
        self.data_table.bind("<<TreeviewSelect>>", self.table_selection)
        self.data_table.column("Image Name", anchor="center", width=100)
        self.data_table.column("Date", anchor="center", width=75)
        self.data_table.column("Intensity", anchor="center", width=75)
        self.data_table.column("Confidence", anchor="center", width=75)
        self.data_table.column("Site", anchor="center", width=75)
        self.data_table.column("Flower Type", anchor="center", width=75)
        self.data_table.heading("Image Name", text="Image Name")
        self.data_table.heading("Date", text="Date")
        self.data_table.heading("Intensity", text="Intensity")
        self.data_table.heading("Confidence", text="Confidence")
        self.data_table.heading("Site", text="Site")
        self.data_table.heading("Flower Type", text="Flower Type")
        self.data_table.pack(fill="both", expand=True, padx=5, pady=(10,0))
        self.scrollbar.configure(command=self.data_table.yview)


          # Modify Bar
        self.modify_bar = customtkinter.CTkFrame(self.table_frame)
        self.modify_bar.pack(fill="x", padx=5)
        self.change_intensity_label = customtkinter.CTkLabel(self.modify_bar, text="Change Intensity:")
        self.change_intensity_label.grid(row=0, column=0, padx=(10, 5))
        self.change_intensity_entry = customtkinter.CTkEntry(self.modify_bar, width=50, height=15)
        self.change_intensity_entry.grid(row=0, column=1)
        self.change_intensity_confirm = customtkinter.CTkButton(self.modify_bar, width=50, height=15, text="Confirm", fg_color="green", command=self.confirm_intensity_change)
        self.change_intensity_confirm.grid(row=0, column=2, padx=10)
        self.label_warning = customtkinter.CTkLabel(self.modify_bar, text="Error: 0-3 Only", text_color="red")
        self.label_warning.grid(row=0, column=3, padx=3)
        self.label_warning.grid_forget()

         # Visual Frame
        self.visual_frame = customtkinter.CTkFrame(self.body_data, fg_color="transparent")
        self.visual_frame.grid(row=0, column=1, sticky="nsew")

          # Export buttons
        self.export_hotbar = customtkinter.CTkFrame(self.visual_frame, fg_color="transparent")
        self.export_hotbar.pack(side = "bottom", fill="x")
        self.export_hotbar.columnconfigure(0, weight=1)
        self.export_csv_button = customtkinter.CTkButton(self.export_hotbar, text="Export CSV", width=50, height=15, command=self.export_csv)
        self.export_csv_button.grid(row=0, column=0, sticky="e")
        self.export_graph_button = customtkinter.CTkButton(self.export_hotbar, text="Export Graph", width=50, height=15, command=self.export_graph)
        self.export_graph_button.grid(row=0, column=1, pady=4, sticky="e", padx=5) 

          # Tabviewer
        self.tab_viewer = customtkinter.CTkTabview(self.visual_frame, fg_color="transparent")
        self.tab_viewer.pack(fill="both", expand=True)

          # Image Tab
        self.image_tab = self.tab_viewer.add("Image View")
        self.image_label = customtkinter.CTkLabel(self.image_tab, image=None, text="")
        self.image_label.pack(pady=5)
        self.visual_frame.pack_propagate(False)
        
          # Graph Tab
        self.graph_tab = self.tab_viewer.add("Graph View")
        self.tab_viewer.set("Graph View")
        self.flowering_plot = FloweringPlot()
        self.graph_ui = FigureCanvasTkAgg(self.flowering_plot.get_figure(), master=self.graph_tab)
        self.graph_ui.draw()
        self.graph_ui.get_tk_widget().pack(expand=True, fill="both")

        # Footer Frame
        self.console_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.console_frame.pack(pady=(0,10), padx=20, fill="x")
        self.console_frame.grid_columnconfigure(0, weight=1)
        self.console_label = customtkinter.CTkLabel(self.console_frame, width=100, text="Console", anchor="w")
        self.console_label.grid(row=0, column=0, sticky="w")
        self.console_text = customtkinter.CTkTextbox(self.console_frame, width=750, height=100)
        self.console_text.grid(row=1, column=0, sticky="ew")
    
    def select_segment_action(self, value):
        self.mode = value
        self.reset()
        if value == "Directory":
            self.export_graph_button.grid()
            self.tab_viewer._segmented_button._buttons_dict["Image View"].grid()
            self.tab_viewer._segmented_button._buttons_dict["Graph View"].grid()
            self.tab_viewer.set("Graph View")
        elif value == "Image":
            self.export_graph_button.grid_remove()
            self.tab_viewer._segmented_button._buttons_dict["Image View"].grid()
            self.tab_viewer._segmented_button._buttons_dict["Graph View"].grid_remove()
            self.tab_viewer.set("Image View")
        else:
            self.export_graph_button.grid()
            self.tab_viewer._segmented_button._buttons_dict["Image View"].grid_remove()
            self.tab_viewer._segmented_button._buttons_dict["Graph View"].grid()
            self.tab_viewer.set("Graph View")

        self.console_log(f"Switched to {self.mode.lower()} mode")
            

    def select_file(self):
        if self.mode == "Directory":
            self.file_path = customtkinter.filedialog.askdirectory()
        elif self.mode == "Image":
            self.file_path = customtkinter.filedialog.askopenfilename(filetypes=[(".heif .heic", "*.heif;*.heic")])
        else:
            self.file_path = customtkinter.filedialog.askopenfilename(filetypes=[(".csv", "*.csv")])
        
        if self.file_path:
            self.confirm_button.grid_remove()
            self.confirm_button.configure(state="enabled", fg_color="green", hover_color="darkgreen")
            self.path_label.configure(text=self.file_path)   
            if (self.mode == "CSV" or self.check_files()):
                self.confirm_button.grid()

            if (self.mode != "CSV"):
                self.confirm_text.grid()

    def check_files(self):
        if self.mode == "Directory":
            self.images = [f for f in os.listdir(self.file_path) if f.endswith(('.heic', '.heif', '.HEIF', '.HEIC'))]
        else:
            self.images = [self.file_path] if self.file_path.endswith(('.heic', '.heif', '.HEIF', '.HEIC')) else []
        if len(self.images) == 0:
            self.console_log(f"Found 0 .heif / .heic files.")
            self.confirm_text.configure(text="0 image files found, please try again.")
            return False
        self.confirm_text.configure(text=f"{len(self.images)} File(s) Found. Click to start analysis.")
        self.console_log(f"Found {len(self.images)} image file(s). Click 'Start Analysis' to begin.")
        return bool(self.images)

    def confirm_selection(self):
        if (self.mode == "CSV"):
            self.console_log("Starting analysis on CSV")
        else:
            self.console_log(f"Starting analysis on {len(self.images)} file(s)...")
        self.confirm_button.grid_remove()
        self.confirm_text.grid_remove()

        self.body_data.pack_forget()
        self.loading.pack(fill="both", expand=True)
        self.loading.configure(text = "Loading...\nThis may take up to 5 minutes.\nPlease ignore any not responding.F")

        if (self.mode == "CSV"):
            self.load_CSV()
        else:
            self.start_AI()

        if not self.data:
            self.loading.configure(text = "Error while reading file.\nPlease ensure file is in the correct formats.")
            return
        
        self.loading.pack_forget()
        self.prediction_frame.pack(before=self.body_data_frame, padx=20,pady=(5, 0), fill="x")
        self.prediction_label.configure(text=f"Predicted Flower: {self.predicted_flower}")
        # Reset Data
        for record in self.data_table.get_children():
            self.data_table.delete(record)

        # Populate table
        for row in self.data:
            self.data_table.insert("", "end", values=[os.path.basename(row[0]), row[1], INTENSITY_CLASSES[int(row[2])], row[3], row[4], row[5]]) 

        # Make Flowering Plot
        self.flowering_plot.plot(self.data)
        self.graph_ui.draw()

        self.body_data.pack(fill="both", expand=True)

    def start_AI(self):
        self.data.clear()
        self.update()
        # Call Prediction AI
        if self.mode == "Directory":
            site = os.path.basename(os.path.normpath(self.file_path))
            results, console_string = predict_folder(self.file_path)
        else:
            site = os.path.basename(os.path.dirname(self.file_path))  
            result, console_string = predict_single(self.file_path)
            results = [result]
        self.console_log(console_string)
        results.sort(key = lambda x: x["date_created"])
        for result in results:
            self.data.append([result["image_path"], result["date_created"], result["intensity_pred"], result["intensity_conf"], site, result["flower_pred"]])
        flower_counts = {}
        for r in results:
            flower_counts[r["flower_pred"]] = flower_counts.get(r["flower_pred"], 0) + 1

        most_common_name = max(flower_counts, key=flower_counts.get)
        self.predicted_flower  = f"{most_common_name} ({flower_counts[most_common_name]} of {len(results)} detected)"

    def load_CSV(self):
        flower_counts = {}
        self.data.clear()
        self.update()
        with open(self.file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader) # Skip the header row
            for row in reader:
                try:
                    parts = row[1].split("-")
                    if len(parts) == 3 and all(p.isdigit() for p in parts):
                        int(row[2])
                        self.data.append(row)
                        flower_counts[row[5]] = flower_counts.get(row[5], 0) + 1
                except (ValueError, IndexError):
                    continue

        if flower_counts:
            self.predicted_flower = max(flower_counts, key=flower_counts.get)
        else:
            self.predicted_flower = "Unknown"

    def table_selection(self, e):
        selected_id = self.data_table.focus()
        row = self.data_table.item(selected_id, "values")
        self.label_warning.grid_forget()

        if not row:
            return

        index = self.data_table.index(selected_id)
        full_path = self.data[index][0]

        self.change_intensity_entry.delete(0, "end")
        self.change_intensity_entry.insert(0, row[2][0])

        self.set_marker(row[1], row[2][0])

        self.image = customtkinter.CTkImage(size=(800, 600), light_image=Image.open(full_path))
        self.image_label.configure(image=self.image)
        
    def confirm_intensity_change(self):
        self.label_warning.grid_forget()
        selected_id = self.data_table.focus()
        if selected_id:
            row = self.data_table.item(selected_id, "values")
            new_intensity = self.change_intensity_entry.get()
            if new_intensity in ["0", "1", "2", "3"]:

                # Update treeview
                self.data_table.set(selected_id, column="Intensity", value=INTENSITY_CLASSES[int(new_intensity)])
                self.data_table.set(selected_id, column="Confidence", value="User Changed")

                # Update data list
                index = self.data_table.index(selected_id)
                self.data[index][2] = new_intensity

                # Update graph
                self.flowering_plot.update_point(row[1], new_intensity)
                self.graph_ui.draw()

                # Update plot marker
                self.set_marker(self.data[index][1], new_intensity)

                self.console_log(f"Intensity change confirmed: {new_intensity}")
            else:
                self.label_warning.grid(row=0, column=3, padx=3)
        else:
            self.console_log("ERROR: No row selected for intensity change")

    def set_marker(self, date, intensity):
        xpos = (int(date.split("-")[1]) - 1) * 4 + int(date.split("-")[2]) // 7
        self.flowering_plot.set_marker(xpos, intensity)
        self.graph_ui.draw()

    def export_graph(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        if file_path:
            self.flowering_plot.save(file_path)
        self.console_log(f"Exporting Graph to {file_path}")

    def export_csv(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Image Name", "Date", "Intensity", "Confidence", "Site", "Flower Type"])
                writer.writerows(self.data)
        self.console_log(f"Exporting CSV to {file_path}")

    def reset(self):
        self.file_path = ""
        self.user_image_label.configure(text=f"Your selected {self.mode}: ")
        self.path_label.configure(text="No directory selected")
        self.select_file_button.configure(text=f"Select {self.mode}")
        self.path_label.configure(text=f"No {self.mode} Selected")
        self.confirm_button.grid_remove()
        self.confirm_text.grid_remove()
        self.confirm_button.configure(state="enabled", fg_color="green", hover_color="darkgreen")
        self.body_data.pack_forget()
        self.prediction_frame.pack_forget()
        self.loading.pack_forget()

    def console_log(self, message):
        self.console_text.insert("end", f"[{datetime.now().strftime('%H:%M')}] - {message}\n")
        self.console_text.see("end")


if __name__ == "__main__":
    app = App()
    app.mainloop()