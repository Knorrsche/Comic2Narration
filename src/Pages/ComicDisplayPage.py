import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
from Classes import Page
from Utils import IOUtils as io
from pygame import mixer
import tkinter.font as fnt
import pygame
import pyttsx3
import os


class ComicDisplayPage:
    def __init__(self, parent, comic, filepath):
        self.parent = parent
        self.comic = comic
        self.current_page_pair_index = 0
        self.last_window_height = self.parent.root.winfo_height()
        self.last_window_width = self.parent.root.winfo_width()
        self.show_speech_bubbles = True
        self.show_panels = True
        self.show_entities = True
        self.show_text = False
        self.import_path = filepath

        self.engine = pyttsx3.init()
        self.buttons = []

        self.menu = tk.Menu(self.parent.root)
        self.parent.root.config(menu=self.menu)
        self.create_menu()

        self.frame_left = tk.Frame(self.parent.root)
        self.frame_left.pack(side='left', expand=True, fill='both')

        self.frame_right = tk.Frame(self.parent.root)
        self.frame_right.pack(side='right', expand=True, fill='both')

        self.label_left = tk.Label(self.frame_left)
        self.label_left.pack(expand=True, fill='both')

        self.label_right = tk.Label(self.frame_right)
        self.label_right.pack(expand=True, fill='both')

        self.left_image = None
        self.right_image = None
        self.left_photo_image = None
        self.right_photo_image = None

        self.resize_pending = None
        self.resize_delay = 10  # milliseconds

        self.parent.root.bind('<Configure>', self.on_window_resize)
        self.parent.root.bind('<Left>', self.show_previous_page_pair)
        self.parent.root.bind('<Right>', self.show_next_page_pair)

        self.display_images()

    def create_menu(self):
        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_comic)
        file_menu.add_command(label="Save", command=self.save_comic)
        file_menu.add_command(label="Match Entities", command=self.match_entities)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.parent.root.quit)

        export_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Export", menu=export_menu)
        export_menu.add_command(label="Export XML", command=self.export_as_xml)
        export_menu.add_command(label="Export Annotated PDF", command=self.export_as_annotated_pdf)
        export_menu.add_command(label="Export Script", command=self.export_as_script)
        export_menu.add_command(label="Export MP3", command=self.export_as_mp3)

        display_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Display", menu=display_menu)
        display_menu.add_command(label="Toggle Panels", command=self.toggle_panels)
        display_menu.add_command(label="Toggle Speech Bubbles", command=self.toggle_speech_bubbles)
        display_menu.add_command(label="Toggle Entities", command=self.toggle_entities)
        display_menu.add_command(label="Toggle Text",command=self.toggle_text)

    def display_images(self):
        if self.current_page_pair_index < 0 or self.current_page_pair_index >= len(self.comic.page_pairs):
            return

        left_page: Page = self.comic.page_pairs[self.current_page_pair_index][0]
        right_page: Page = self.comic.page_pairs[self.current_page_pair_index][1]

        left_image_array = left_page.annotated_image(self.show_panels,
                                                     self.show_speech_bubbles,
                                                     self.show_entities) if left_page is not None else self.create_blank_image()
        right_image_array = right_page.annotated_image(self.show_panels,
                                                       self.show_speech_bubbles,
                                                       self.show_entities) if right_page is not None else self.create_blank_image()

        self.left_image = Image.fromarray(left_image_array)
        self.right_image = Image.fromarray(right_image_array)

        self.resize_and_update_images()

    def resize_and_update_images(self):
        if self.left_image is None or self.right_image is None:
            return

        original_width, original_height = self.left_image.size

        max_width = self.parent.root.winfo_width() // 2
        max_height = self.parent.root.winfo_height()

        resized_left_image = self.left_image.resize((max_width, max_height), Image.LANCZOS)
        resized_right_image = self.right_image.resize((max_width, max_height), Image.LANCZOS)

        self.left_photo_image = ImageTk.PhotoImage(resized_left_image)
        self.right_photo_image = ImageTk.PhotoImage(resized_right_image)

        self.label_left.config(image=self.left_photo_image)
        self.label_right.config(image=self.right_photo_image)

        self.frame_left.config(width=max_width)
        self.frame_right.config(width=max_width)

        self.label_left.bind('<Button-1>', self.show_previous_page_pair)
        self.label_right.bind('<Button-1>', self.show_next_page_pair)

        width_scale = max_width / original_width
        height_scale = max_height / original_height

        if self.show_text:
            self.create_interactive_buttons(width_scale, height_scale)

    def create_blank_image(self):
        width, height = 800, 600
        blank_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        return blank_image

    def on_window_resize(self, event):
        if event.width == self.last_window_width and event.height == self.last_window_height:
            return

        if event.widget == self.parent.root:
            if self.resize_pending:
                self.parent.root.after_cancel(self.resize_pending)
            self.last_window_width = event.width
            self.last_window_height = event.height
            self.resize_pending = self.parent.root.after(self.resize_delay, self.handle_delayed_resize)

    def handle_delayed_resize(self):
        self.resize_and_update_images()
        self.resize_pending = None

    def show_previous_page_pair(self, event=None):
        if self.current_page_pair_index > 0:
            self.current_page_pair_index -= 1
            self.display_images()

    def show_next_page_pair(self, event=None):
        if self.current_page_pair_index < len(self.comic.page_pairs) - 1:
            self.current_page_pair_index += 1
            self.display_images()

    def toggle_panels(self):
        self.show_panels = not self.show_panels
        self.display_images()

    def toggle_speech_bubbles(self):
        self.show_speech_bubbles = not self.show_speech_bubbles
        self.display_images()

    def toggle_entities(self):
        self.show_entities = not self.show_entities
        self.display_images()

    def toggle_text(self):
        self.show_text = not self.show_text

        if not self.show_text:
            for button in self.buttons:
                button.destroy()
            self.buttons = []

        self.display_images()

    def export_as_xml(self):
        export_path = filedialog.asksaveasfilename(defaultextension=".xml",
                                                   filetypes=[("XML Files", "*.xml")],
                                                   title="Select where to save the XML")
        if export_path:
            xml = self.comic.to_xml()
            xml_str = io.prettify_xml(xml)
            io.save_xml_to_file(export_path, xml_str)

            messagebox.showinfo("Export Successful", "The XML was exported successfully")
        else:
            messagebox.showerror("Error", "Error while trying to export, please try again")

    def export_as_annotated_pdf(self):
        export_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                   filetypes=[("PDF files", "*.pdf")],
                                                   title="Select where to save the annotated PDF")

        if export_path:
            xml = self.comic.to_xml()
            xml_str = io.prettify_xml(xml)
            io.add_annotation_to_pdf(self.import_path, xml_str, export_path)

            messagebox.showinfo("Export Successfully", "The annotated PDF was exported successfully")
        else:
            messagebox.showerror("Error", "Error while trying to export, please try again")

    def export_as_script(self):
        export_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                   filetypes=[("Text files", "*.txt")],
                                                   title="Select where to save the Script")

        if export_path:
            io.save_script_as_txt(export_path, self.comic.to_narrative())

            messagebox.showinfo("Export Successfully", "The Script was exported successfully")
        else:
            messagebox.showerror("Error", "Error while trying to export, please try again")

    def export_as_mp3(self):
        export_path = filedialog.asksaveasfilename(defaultextension=".mp3",
                                                   filetypes=[("MP3 files", "*.mp3")],
                                                   title="Select where to save the MP3")

        if export_path:
            io.save_script_as_mp3(export_path, self.comic.to_narrative())

            messagebox.showinfo("Export Successfully", "The MP3 was exported successfully")
        else:
            messagebox.showerror("Error", "Error while trying to export, please try again")

    # TODO: Add load comic
    def open_comic(self):
        pass

    # TODO: Add save comic
    def save_comic(self):
        pass

    def match_entities(self):
        input_window = tk.Toplevel(self.parent.root)
        input_window.title("Input Cluster Sizes and Options")

        num_entities = len(self.comic.scenes)

        self.create_input_fields(input_window, num_entities)

    def create_input_fields(self, input_window, num_entities):
        for widget in input_window.winfo_children():
            widget.destroy()

        label = tk.Label(input_window, text="Enter Cluster Sizes:")
        label.pack(pady=10)

        entity_entries = []
        if num_entities == 0:
            entry_label = tk.Label(input_window, text=f"Scene {1}:")
            entry_label.pack()
            entry = tk.Entry(input_window)
            entry.pack(pady=5)
            entity_entries.append(entry)
        else:
            for i in range(num_entities):
                entry_label = tk.Label(input_window, text=f"Scene {i + 1}:")
                entry_label.pack()
                entry = tk.Entry(input_window)
                entry.pack(pady=5)
                entity_entries.append(entry)

        confidence_label = tk.Label(input_window, text="Enter Confidence Limit (0-1):")
        confidence_label.pack(pady=10)
        confidence_entry = tk.Entry(input_window)
        confidence_entry.pack(pady=5)

        algorithm_label = tk.Label(input_window, text="Select Clustering Algorithm:")
        algorithm_label.pack(pady=10)

        algorithm_options = ['KMeans', 'DBSCAN', 'Agglomerative', 'Gaussian Mixture', 'Birch']
        algorithm_var = tk.StringVar(input_window)
        algorithm_var.set(algorithm_options[0])
        algorithm_menu = tk.OptionMenu(input_window, algorithm_var, *algorithm_options)
        algorithm_menu.pack(pady=5)

        input_type_label = tk.Label(input_window, text="Select Input Encoding Type:")
        input_type_label.pack(pady=10)

        input_type_options = ['One-Hot Encoding', 'TF-IDF-scaled One-Hot', 'Word2Vec', 'TF-IDF-scaled Word2Vec']
        input_type_var = tk.StringVar(input_window)
        input_type_var.set(input_type_options[0])
        input_type_menu = tk.OptionMenu(input_window, input_type_var, *input_type_options)
        input_type_menu.pack(pady=5)

        submit_button = tk.Button(input_window, text="Submit",
                                  command=lambda: self.submit_entities(input_window, entity_entries, algorithm_var, input_type_var, confidence_entry))
        submit_button.pack(pady=10)

    def submit_entities(self, input_window, entity_entries, algorithm_var, input_type_var, confidence_entry):
        cluster_list = []
        for entry in entity_entries:
            try:
                entity_value = int(entry.get())
                cluster_list.append(entity_value)
            except ValueError:
                messagebox.showerror("Error", "Please enter valid cluster sizes")
                return

        algorithm = algorithm_var.get()
        input_type = input_type_var.get()

        try:
            confidence_value = float(confidence_entry.get())
            if confidence_value < 0 or confidence_value > 1:
                raise ValueError("Confidence limit must be between 0 and 1.")
        except ValueError as ve:
            messagebox.showerror("Error", str(ve))
            return

        self.comic.match_entities(cluster_list, algorithm=algorithm, input_type=input_type, confidence=confidence_value, debug=True)

        self.display_images()

        input_window.destroy()

    def create_interactive_buttons(self, width_scale, height_scale):
        for button in self.buttons:
            button.destroy()
        self.buttons = []

        left_page = self.comic.page_pairs[self.current_page_pair_index][0]
        if left_page:
            self.create_buttons_for_page(left_page, self.frame_left, width_scale, height_scale)

        right_page = self.comic.page_pairs[self.current_page_pair_index][1]
        if right_page:
            self.create_buttons_for_page(right_page, self.frame_right, width_scale, height_scale)

    def create_buttons_for_page(self, page, frame, width_scale, height_scale):
        for panel in page.panels:
            if self.show_panels:
                bbox = panel.bounding_box

                center_x = bbox['x'] * width_scale
                center_y = bbox['y'] * height_scale

                w = (bbox['width'] * width_scale)/10
                h = (bbox['height'] * height_scale)/10

                button_x = center_x - (w / 2)
                button_y = center_y - (h / 2)

                borderwidth = 0

                if panel.starting_tag:
                    borderwidth = w/10

                button = tk.Button(
                    frame,
                    bg='green',
                    text=panel.description,
                    font=fnt.Font(size=6),
                    borderwidth=borderwidth,
                    highlightthickness=0,
                    command=lambda e=panel: self.on_panel_click(e)
                )

                button.place(x=button_x, y=button_y, width=w, height=h)

                button.bind("<Button-3>", lambda event, e=panel: self.on_panel_right_click(e))

                self.buttons.append(button)

            if self.show_speech_bubbles:
                for speech_bubble in panel.speech_bubbles:
                    bbox = speech_bubble.bounding_box

                    center_x = bbox['x'] * width_scale
                    center_y = bbox['y'] * height_scale
                    w = bbox['width'] * width_scale
                    h = bbox['height'] * height_scale

                    button = tk.Button(
                        frame,
                        bg="red",
                        text=speech_bubble.text,
                        font=fnt.Font(size=6),
                        borderwidth=0,
                        highlightthickness=0,
                        command=lambda e=speech_bubble: self.on_speech_bubble_click(e)
                    )

                    button_x = center_x - (w / 2)
                    button_y = center_y - (h / 2)

                    button.place(x=button_x, y=button_y, width=w, height=h)
                    self.buttons.append(button)

            if self.show_entities:
                for entity in panel.entities:
                    bbox = entity.bounding_box

                    center_x = bbox['x'] * width_scale
                    center_y = bbox['y'] * height_scale
                    w = bbox['width'] * width_scale
                    h = bbox['height'] * height_scale

                    button = tk.Button(
                        frame,
                        bg="blue",
                        text=speech_bubble.text,
                        font=fnt.Font(size=6),
                        borderwidth=0,
                        highlightthickness=0,
                        command=lambda e=entity: self.on_entity_click(e)
                    )

                    button_x = center_x - (w / 2)
                    button_y = center_y - (h / 2)

                    button.place(x=button_x, y=button_y, width=w, height=h)
                    self.buttons.append(button)

    #TODO: refactor with speechbubble
    def on_panel_click(self,panel):
        text = panel.description

        if text == '':
            return

        text = text.replace("\n", " ")
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('volume', 1.0)
        self.engine.setProperty('voice', voices[1].id)
        self.engine.setProperty('rate', 200)
        outfile = "temp.wav"
        self.engine.save_to_file(text, outfile)
        self.engine.runAndWait()

        mixer.init()
        mixer.music.load("temp.wav")
        mixer.music.play()

        while mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        mixer.music.stop()
        mixer.quit()
        if os.path.isfile(outfile):
            os.remove(outfile)

    def on_speech_bubble_click(self, speech_bubble):

        text = speech_bubble.text

        if text == '':
            return

        text = text.replace("\n", " ")
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('volume', 1.0)
        self.engine.setProperty('voice', voices[1].id)
        self.engine.setProperty('rate', 200)
        outfile = "temp.wav"
        self.engine.save_to_file(text, outfile)
        self.engine.runAndWait()

        mixer.init()
        mixer.music.load("temp.wav")
        mixer.music.play()

        while mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        mixer.music.stop()
        mixer.quit()
        if os.path.isfile(outfile):
            os.remove(outfile)

    # TODO: Add more efficent scene algorithm
    def on_panel_right_click(self, panel):
        panel.starting_tag = not panel.starting_tag
        self.comic.update_scenes()
        self.display_images()

    #TODO: show matching tags with other entities in scene or show counter of tag in scene entities
    def on_entity_click(self, entity):
        tags_window = tk.Toplevel(self.parent.root)
        tags_window.title("Entity Tags and Confidence")

        confidence_label = tk.Label(tags_window, text="Filter tags by confidence:", font=("Helvetica", 12, "bold"))
        confidence_label.pack(pady=10)

        confidence_scale = tk.Scale(
            tags_window,
            from_=0,
            to=1,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            label="Confidence Threshold",
            length=300
        )
        confidence_scale.set(0.1)
        confidence_scale.pack(padx=10, pady=5)

        total_count_label = tk.Label(tags_window, text="", font=("Helvetica", 12))
        total_count_label.pack(pady=5)

        tags_frame = tk.Frame(tags_window)
        tags_frame.pack(pady=10)

        def update_displayed_tags():
            for widget in tags_frame.winfo_children():
                widget.destroy()

            threshold = confidence_scale.get()

            filtered_tags = [tag for tag in entity.tags if
                             tag[1] >= threshold]

            total_count_label.config(text=f"Total Tags Meeting Confidence ({threshold:.2f}): {len(filtered_tags)}")

            for tag in filtered_tags:
                tag_label = tk.Label(tags_frame, text=f"{tag[0]} (Confidence: {tag[1]:.2f})")
                tag_label.pack(anchor='w', padx=10)

        update_displayed_tags()

        confidence_scale.bind("<Motion>", lambda event: update_displayed_tags())

        close_button = tk.Button(tags_window, text="Close", command=tags_window.destroy)
        close_button.pack(pady=10)