import tkinter as tk
from tkinter import IntVar
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from Classes.Comic import Comic

class ComicDisplayPage:
    def __init__(self, parent, comic: Comic):
        self.parent = parent
        self.comic = comic
        self.current_page_index = 0
        self.current_panel_index = 0
        self.current_speech_bubble_index = 0
        self.highlight_current_box = IntVar()
        self.highlight_current_box.set(1)
        self.show_all_boxes = IntVar()
        self.show_all_boxes.set(1)

        self.frame = tk.Frame(parent.root)
        self.frame.pack(padx=20, pady=20)

        self.label = tk.Label(self.frame, text="Comic Display", font=("Arial", 14, "bold"))
        self.label.grid(row=0, column=0, pady=10, columnspan=2)

        self.info_label = tk.Label(self.frame, text=f"Comic Info: {comic.name}, Volume: {comic.volume}")
        self.info_label.grid(row=1, column=0, pady=10, columnspan=2)

        self.page_panel_frame = tk.Frame(self.frame)
        self.page_panel_frame.grid(row=2, column=0, padx=10, pady=10)

        self.page_image_label = tk.Label(self.page_panel_frame)
        self.page_image_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.panel_image_label = tk.Label(self.page_panel_frame)
        self.panel_image_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.speech_bubble_label = tk.Label(self.page_panel_frame)
        self.speech_bubble_label.pack(side=tk.RIGHT, padx=10, pady=10)

        self.controls_frame = tk.Frame(self.frame)
        self.controls_frame.grid(row=3, column=0, pady=10, columnspan=2)

        self.page_number_label = tk.Label(self.controls_frame, text=f"Page {self.current_page_index + 1} of {len(self.comic.pages)}")
        self.page_number_label.grid(row=0, column=0, columnspan=2, padx=10)

        self.panel_number_label = tk.Label(self.controls_frame, text=f"Panel {self.current_panel_index + 1}")
        self.panel_number_label.grid(row=0, column=2, columnspan=2, padx=10)

        self.speech_bubble_number_label = tk.Label(self.controls_frame, text=f"Speech Bubble {self.current_speech_bubble_index + 1}")
        self.speech_bubble_number_label.grid(row=0, column=4, columnspan=2, padx=10)

        self.prev_page_button = tk.Button(self.controls_frame, text="Previous Page", command=self.prev_page)
        self.prev_page_button.grid(row=1, column=0, padx=10, pady=10)

        self.next_page_button = tk.Button(self.controls_frame, text="Next Page", command=self.next_page)
        self.next_page_button.grid(row=1, column=1, padx=10, pady=10)

        self.prev_panel_button = tk.Button(self.controls_frame, text="Previous Panel", command=self.prev_panel)
        self.prev_panel_button.grid(row=1, column=2, padx=10, pady=10)

        self.next_panel_button = tk.Button(self.controls_frame, text="Next Panel", command=self.next_panel)
        self.next_panel_button.grid(row=1, column=3, padx=10, pady=10)

        self.prev_speech_bubble_button = tk.Button(self.controls_frame, text="Previous Speech Bubble", command=self.prev_speech_bubble)
        self.prev_speech_bubble_button.grid(row=1, column=4, padx=10, pady=10)

        self.next_speech_bubble_button = tk.Button(self.controls_frame, text="Next Speech Bubble", command=self.next_speech_bubble)
        self.next_speech_bubble_button.grid(row=1, column=5, padx=10, pady=10)

        self.highlight_current_box_checkbox = tk.Checkbutton(self.controls_frame, text="Highlight Current Bounding Box", variable=self.highlight_current_box, command=self.toggle_boxes)
        self.highlight_current_box_checkbox.grid(row=2, column=2, columnspan=2, padx=10, pady=10)

        self.show_all_boxes_checkbox = tk.Checkbutton(self.controls_frame, text="Show All Bounding Boxes", variable=self.show_all_boxes, command=self.toggle_boxes)
        self.show_all_boxes_checkbox.grid(row=2, column=4, columnspan=2, padx=10, pady=10)

        self.load_comic_page()
    
    def load_comic_page(self):
        if 0 <= self.current_page_index < len(self.comic.pages):
            self.current_panel_index = 0
            self.current_speech_bubble_index = 0
            page = self.comic.pages[self.current_page_index]
            self.load_panel(page)
            self.page_number_label.config(text=f"Page {self.current_page_index + 1} of {len(self.comic.pages)}")

    def load_panel(self, page):
        if 0 <= self.current_panel_index < len(page.panels):
            panel = page.panels[self.current_panel_index]

            panel_img = Image.fromarray(panel.image)
            panel_img = panel_img.resize((200, 300))
            panel_img_tk = ImageTk.PhotoImage(panel_img)
            self.panel_image_label.configure(image=panel_img_tk)
            self.panel_image_label.image = panel_img_tk

            self.load_speech_bubble(panel)
            self.draw_page_with_panel(page, panel)

            self.panel_number_label.config(text=f"Panel {self.current_panel_index + 1} of {len(page.panels)}")

    def load_speech_bubble(self, panel):
        if 0 <= self.current_speech_bubble_index < len(panel.speech_bubbles):
            speech_bubble = panel.speech_bubbles[self.current_speech_bubble_index]

            speech_bubble_img = Image.fromarray(speech_bubble.image)
            speech_bubble_img = speech_bubble_img.resize((200, 100))
            speech_bubble_img_tk = ImageTk.PhotoImage(speech_bubble_img)
            self.speech_bubble_label.configure(image=speech_bubble_img_tk)
            self.speech_bubble_label.image = speech_bubble_img_tk

            self.speech_bubble_number_label.config(text=f"Speech Bubble {self.current_speech_bubble_index + 1} of {len(panel.speech_bubbles)}")

    def draw_page_with_panel(self, page, panel):
        img = Image.fromarray(page.page_image)
        draw = ImageDraw.Draw(img)

        if self.show_all_boxes.get() == 1:
            for p in page.panels:
                bbox = p.bounding_box
                panel_x1 = int(bbox['x'] - bbox['width'] / 2)
                panel_y1 = int(bbox['y'] - bbox['height'] / 2)
                panel_x2 = panel_x1 + int(bbox['width'])
                panel_y2 = panel_y1 + int(bbox['height'])
                draw.rectangle([panel_x1, panel_y1, panel_x2, panel_y2], outline="red", width=8)

                for idx, sb in enumerate(p.speech_bubbles):
                    bbox = sb.bounding_box
                    x1 = panel_x1 + int(bbox['x'] - bbox['width'] / 2)
                    y1 = panel_y1+ int(bbox['y'] - bbox['height'] / 2)
                    x2 = x1 + int(bbox['width'])
                    y2 = y1 + int(bbox['height'])
                    draw.rectangle([x1,y1, x2, y2], outline="blue", width=8)

        if self.highlight_current_box.get() == 1:
            bbox = panel.bounding_box
            x1 = int(bbox['x'] - bbox['width'] / 2)
            y1 = int(bbox['y'] - bbox['height'] / 2)
            x2 = x1 + int(bbox['width'])
            y2 = y1 + int(bbox['height'])
            draw.rectangle([x1, y1, x2, y2], outline="green", width=8)

        img = img.resize((400, 600))
        img_tk = ImageTk.PhotoImage(img)
        self.page_image_label.configure(image=img_tk)
        self.page_image_label.image = img_tk

    def toggle_boxes(self):
        page = self.comic.pages[self.current_page_index]
        panel = page.panels[self.current_panel_index]
        self.draw_page_with_panel(page, panel)

    def prev_page(self):
        if self.current_page_index > 0:
            self.current_page_index -= 1
            self.load_comic_page()

    def next_page(self):
        if self.current_page_index < len(self.comic.pages) - 1:
            self.current_page_index += 1
            self.load_comic_page()

    def prev_panel(self):
        if self.current_panel_index > 0:
            self.current_panel_index -= 1
            self.current_speech_bubble_index = 0
            page = self.comic.pages[self.current_page_index]
            self.load_panel(page)

    def next_panel(self):
        page = self.comic.pages[self.current_page_index]
        if self.current_panel_index < len(page.panels) - 1:
            self.current_panel_index += 1
            self.current_speech_bubble_index = 0
            self.load_panel(page)

    def prev_speech_bubble(self):
        if self.current_speech_bubble_index > 0:
            self.current_speech_bubble_index -= 1
            page = self.comic.pages[self.current_page_index]
            panel = page.panels[self.current_panel_index]
            self.load_speech_bubble(panel)

    def next_speech_bubble(self):
        page = self.comic.pages[self.current_page_index]
        panel = page.panels[self.current_panel_index]
        if self.current_speech_bubble_index < len(panel.speech_bubbles) - 1:
            self.current_speech_bubble_index += 1
            self.load_speech_bubble(panel)
