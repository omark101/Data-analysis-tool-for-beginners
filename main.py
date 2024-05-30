import customtkinter 
from PIL import Image, ImageTk
import tkinter as tk
from serpapi import GoogleSearch
from tkinter import filedialog
from tkinter import scrolledtext
from io import StringIO
import sys
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urlparse
import cv2
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns




Main_window_photo = Image.open("C:/Users/omera/Downloads/Add a heading (1).png") 
search_engine_window = Image.open("C:/Users/omera/.spyder-py3/1 (1).png")   

class App:
    def __init__(self):
        self.app = customtkinter.CTk()
        self.app.geometry("1120x700")
        self.file_path = None 
        self.excel_data = None  
        self.web_url = None
        self.image_urls = []
        self.photos = []
        self.setup_main_window()
        
    def create_button(self, master, text, x, y, command=None):
        button = customtkinter.CTkButton(master=master, text=text, state=customtkinter.NORMAL, width=150, height=35, command=command)
        button.configure(bg_color="black")
        button.place(x=x, y=y)
        return button
    
    def create_entry(self, master, placeholder, x, y):
        entry = customtkinter.CTkEntry(master=master, placeholder_text=placeholder, placeholder_text_color="lightgrey", border_color="deepskyblue", width=290)
        entry.place(x=x, y=y)
        return entry

    def setup_main_window(self):
        canvas_width, canvas_height = Main_window_photo.size
        canvas = tk.Canvas(self.app, width=canvas_width, height=canvas_height)
        canvas.pack(fill="both", expand=True)
        bg = ImageTk.PhotoImage(Main_window_photo)    
        canvas.Main_window_photo = bg    
        canvas.create_image(0, 0, image=bg, anchor="nw")
        
        face_deticton = self.create_button(canvas, "face detectiom", 150, 250, self.face_detection_window)
        #models_button = self.create_button(canvas, "3D Models", 150, 300)
        #photo_button = self.create_button(canvas, "Photos section", 150, 350)
        excel_button = self.create_button(canvas, "Excel file section", 150, 400, self.Excel_analysis)
        search_button = self.create_button(canvas, "Search engines", 150, 450, self.open_search_engines)
        web_scraping_button = self.create_button(canvas, "Web scraping", 150, 500,self.Web_scrap_window_setup)


    def open_search_engines(self):
        search_window = customtkinter.CTkToplevel()
        search_window.geometry("1040x690")
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")
        canvas_width, canvas_height = search_engine_window.size
        canvas = tk.Canvas(search_window, width=canvas_width, height=canvas_height)
        canvas.pack(fill="both", expand=True)
        bg = ImageTk.PhotoImage(search_engine_window)    
        canvas.search_engine_window = bg    
        canvas.create_image(0, 0, image=bg, anchor="nw")
        
        self.create_button(canvas, "Youtube", 150, 400, self.youtube_search_start)
        self.create_button(canvas, "Google", 150, 350, self.google_button_clicked)
        

    def Excel_analysis(self):
        excel_window = customtkinter.CTkToplevel()
        excel_window.geometry("1100x700")  

        frame = tk.Frame(excel_window, width=200, height=200)
        frame.pack(side="top", padx=20, pady=20)

        self.text_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=150, height=15)  
        self.text_area.pack(expand=True, fill="both")  

        self.stdout = sys.stdout
        sys.stdout = StringIO()
        self.text_area.config(state=tk.DISABLED)
        self.text_area.bind("<Key>", lambda e: "break")
        self.text_area.bind("<Button-1>", lambda e: "break")
        
        
        self.create_button(excel_window,"Upload File",500 ,300,self.upload_file)
        self.create_button(excel_window, "all coloumns names",500, 350, lambda: self.coloumns(self.file_path))
        self.create_button(excel_window, "all rows names",500, 400, lambda: self.rows(self.file_path))
        self.search_entry = self.create_entry(excel_window, "search the occurrence times of word", 450, 450)
        self.create_button(excel_window, "search", 500, 500,self.search_word_occurrences)
        
        
    def Web_scrap_window_setup(self):
        web_scrap_window = customtkinter.CTkToplevel()
        web_scrap_window.geometry("1100x700")  
        web_scrap_window.title("Web Scrap Window")

        frame = tk.Frame(web_scrap_window, width=200, height=200)
        frame.pack(side="top", padx=20, pady=20)

        self.text_area_for_scrap = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=150, height=15)  
        self.text_area_for_scrap.pack(expand=True, fill="both")  

        self.stdout = sys.stdout
        sys.stdout = StringIO()
        self.text_area_for_scrap.config(state=tk.DISABLED)
        self.text_area_for_scrap.bind("<Key>", lambda e: "break")
        self.text_area_for_scrap.bind("<Button-1>", lambda e: "break")

        self.print_to_text_area_scrap_window("Hello")
        self.print_to_text_area_scrap_window("Please enter a web page you want to scrap and make sure it's legal")

        web_url_entry = self.create_entry(web_scrap_window, "Enter the Web page", 450, 350)
        
        def download_images():
            self.download_images()
            
        self.create_button(web_scrap_window, "continue", 520, 400, lambda: self.scrap_function(web_url_entry))
        self.create_button(web_scrap_window, "Download the scraped photos", 520, 450, download_images)



    def face_detection_window(self):
        face_window = customtkinter.CTkToplevel()
        face_window.geometry("1100x700")  
        face_window.title("face detection")
        
        self.create_button(face_window, "upload photos", 450, 300 , self.upload_photos)
        self.create_button(face_window, "continue",450,350,self.generate_graph)
        self.create_button(face_window, "heat map",450,450,self.generate_heatmap)
        
        frame = tk.Frame(face_window, width=200, height=200)
        frame.pack(side="top", padx=20, pady=20)

        self.text_area_photos_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=150, height=15)  
        self.text_area_photos_area.pack(expand=True, fill="both")  

        self.stdout = sys.stdout
        sys.stdout = StringIO()
        self.text_area_photos_area.config(state=tk.DISABLED)
        self.text_area_photos_area.bind("<Key>", lambda e: "break")
        self.text_area_photos_area.bind("<Button-1>", lambda e: "break")
        
        self.print_to_text_area_photos_window("heloo it work")
        
        
        

    def scrap_function(self, web_url_entry):
        url = web_url_entry.get()
        response = requests.get(url)
        if response.status_code == 200:
            self.print_to_text_area_scrap_window("You're allowed to continue")
        
            soup = BeautifulSoup(response.content, "html5lib")
            img_tags = soup.find_all('img')
            self.image_urls.extend(img['src'] for img in img_tags if 'src' in img.attrs)
            
            self.print_to_text_area_scrap_window("Image URLs:")
            for img_url in self.image_urls:
                self.print_to_text_area_scrap_window(img_url)
        else:
            self.print_to_text_area_scrap_window("Failed to access the website")

            

 
    def download_images(self):        
        try:
            for img_url in self.image_urls:
                response = requests.get(img_url, stream=True)
                if response.status_code == 200:
                    save_directory = os.path.join(os.path.expanduser('~'), 'Downloads', 'Images')
                    os.makedirs(save_directory, exist_ok=True)
                    filename = os.path.basename(urlparse(img_url).path)
                    with open(os.path.join(save_directory, filename), 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                self.print_to_text_area_scrap_window(f"Image '{filename}' downloaded successfully.")
            else:
                 self.print_to_text_area_scrap_window(f"Failed to download image: {img_url}")
        except Exception as e:
          self.print_to_text_area_scrap_window(f"An error occurred: {e}")
  

    def upload_photos(self):
        directory_path = filedialog.askdirectory()
        if directory_path:
            self.print_to_text_area_photos_window(f"Directory selected: {directory_path}")

            num_faces_per_photo = []  

            
            for filename in os.listdir(directory_path):
            
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(directory_path, filename)

                
                    image = cv2.imread(file_path)
                    if image is not None:
                        
                        self.print_to_text_area_photos_window(f"Image shape: {image.shape}")
                        gray = image
                        self.print_to_text_area_photos_window(f"Grayscale image shape: {gray.shape}")
                        num_faces = self.detect_faces(gray)
                        num_faces_per_photo.append(num_faces)  
                        self.photos.append((image, num_faces))
                        self.print_to_text_area_photos_window(f"Photo loaded: {file_path}, Faces detected: {num_faces}")
                    else:
                        self.print_to_text_area_photos_window(f"Failed to load photo: {file_path}")
                else:
                    self.print_to_text_area_photos_window(f"Ignored non-image file: {filename}")
            self.num_faces_per_photo = num_faces_per_photo
        else:
            self.print_to_text_area_photos_window("No directory selected.")

    def generate_heatmap(self):
        if self.num_faces_per_photo:
            plt.figure(figsize=(8, 6))
            sns.heatmap([self.num_faces_per_photo], cmap='YlGnBu', annot=True, fmt='d')
            plt.title('Number of Faces Detected in Each Photo')
            plt.xlabel('Photo Index')
            plt.ylabel('Number of Faces')
            plt.show()
        else:
            self.print_to_text_area_photos_window("Please upload photos and detect faces first.")




    def detect_faces(self, image):
        print(f"Input image shape: {image.shape}")
        gray = image
        print(f"Grayscale image shape: {gray.shape}")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        num_faces = len(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('Detected Faces', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return num_faces


    
    
    
    
    def generate_graph(self):
        photos_with_faces = sum(1 for _, num_faces in self.photos if num_faces > 0)
        photos_without_faces = sum(1 for _, num_faces in self.photos if num_faces == 0)
        total_photos = len(self.photos)

        labels = ['With Faces', 'Without Faces']
        counts = [photos_with_faces, photos_without_faces]

        plt.figure(figsize=(8, 6))
        plt.bar(labels, counts, color=['blue', 'orange'])
        plt.title('Photos with and without Faces')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.yticks(range(total_photos + 1))
        for i, count in enumerate(counts):
            plt.text(i, count, str(count), ha='center', va='bottom')

        plt.show()  



        
    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls"), ("CSV files", "*.csv")])
        if file_path:
            self.file_path = file_path  
            self.print_to_text_area(f"File uploaded: {file_path}")
            try:
                self.excel_data = pd.read_csv(file_path)
            except pd.errors.ParserError:
                try:
                    self.excel_data = pd.read_excel(file_path)
                except Exception as e:
                    self.print_to_text_area(f"Error reading file: {e}")

            


    def coloumns(self, file_path):
        try:
            csv_file = pd.read_csv(file_path)
        
            self.print_to_text_area("Column/Row names:")
            for col_name in csv_file.columns:
                self.print_to_text_area(col_name)
            
        except Exception as e:
            self.print_to_text_area(f"Error reading file: {e}")
            
            
            
    def rows(self, file_path):
        try:
            csv_file = pd.read_csv(file_path)
            self.print_to_text_area("Row names:")
            for index, row in csv_file.iterrows():
                self.print_to_text_area(f"Row {index + 1}: {row.values}")
            
        except Exception as e:
            self.print_to_text_area(f"Error reading file: {e}")
            
            
    def search_word_occurrences(self):
        word_to_search = self.search_entry.get()
        
        if self.excel_data is None:
            self.print_to_text_area("Please upload an Excel file first.")
            return
        
        occurrences = self.excel_data.apply(lambda row: row.astype(str).str.lower().str.count(word_to_search.lower())).sum().sum()        
        self.print_to_text_area(f"The word '{word_to_search}' appears {occurrences} times in the uploaded file")


        

    def google_button_clicked(self):
        google_engine = customtkinter.CTkToplevel()
        google_engine.geometry("1100x700")
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")
        
        frame = tk.Frame(google_engine, width=200, height=200)
        frame.pack(side="top", padx=20, pady=20)

        self.text_area_for_scrap_api = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=150, height=15)  
        self.text_area_for_scrap_api.pack(expand=True, fill="both")  

        self.stdout = sys.stdout
        sys.stdout = StringIO()
        self.text_area_for_scrap_api.config(state=tk.DISABLED)
        self.text_area_for_scrap_api.bind("<Key>", lambda e: "break")
        self.text_area_for_scrap_api.bind("<Button-1>", lambda e: "break")
        self.print_to_text_area_scrap_api_google("hello \n please fill the required data")


        query_entry_google = self.create_entry(google_engine, "Enter the title you want to search.....", 450, 350)
        country_entry_google = self.create_entry(google_engine, "Enter country abbreviation to search in.....", 450, 400)
        api_entry_google = self.create_entry(google_engine, "Enter your API key.....", 450, 450)
        self.create_button(google_engine, "Continue", 450, 500, lambda: self.process_for_google(query_entry_google.get(), country_entry_google.get(), api_entry_google.get()))
        
        
        
        
    def youtube_search_start(self):
        youtube_engine = customtkinter.CTkToplevel()
        youtube_engine.geometry("1100x700")
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")
        
        
        frame = tk.Frame(youtube_engine, width=200, height=200)
        frame.pack(side="top", padx=20, pady=20)

        self.text_area_for_scrap_api_youtube = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=150, height=15)  
        self.text_area_for_scrap_api_youtube.pack(expand=True, fill="both")  

        self.stdout = sys.stdout
        sys.stdout = StringIO()
        self.text_area_for_scrap_api_youtube.config(state=tk.DISABLED)
        self.text_area_for_scrap_api_youtube.bind("<Key>", lambda e: "break")
        self.text_area_for_scrap_api_youtube.bind("<Button-1>", lambda e: "break")

        query_entry = self.create_entry(youtube_engine, "Enter the title you want to search.....", 450, 350)
        country_entry = self.create_entry(youtube_engine, "Enter country abbreviation to search in.....", 450, 400)
        api_entry = self.create_entry(youtube_engine, "Enter your API key.....", 450, 450)

        self.create_button(youtube_engine, "Continue", 450, 500, lambda: self.process(query_entry.get(), country_entry.get(), api_entry.get()))
        

    def process_for_google(self, query, country, api_key):
        params = {
            "q": query,
            "engine": "google",
            "api_key": api_key,
            "gl": country
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        self.print_to_text_area_scrap_api_google(results)

    def process(self, query, country, api_key):
        params = {
            "search_query": query,
            "engine": "youtube",
            "api_key": api_key,
            "gl": country
        }
        search = GoogleSearch(params)
        results = search.get_dict()    
        final_result = results["video_results"]
        for i, video in enumerate(final_result):
            self.print_to_text_area_scrap_api_youtube(f"Video link {i+1}: {video['link']} \t Its channel: {video['channel']['name']}")
            
        G = nx.DiGraph()

        search_results = []

        for i, video in enumerate(final_result):
            search_results.append({"video_link": f"Video Link {i+1}", "channel": video['channel']['name']})
            G.add_node(video['channel']['name'])  
            G.add_edge(f"Video Link {i+1}", video['channel']['name'])  

        nx.draw(G, with_labels=True, node_size=1000, node_color="skyblue", font_size=10)
        plt.show()

            
            
    def print_to_text_area(self, message):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, message + "\n")
        self.text_area.see(tk.END)
        self.text_area.config(state=tk.DISABLED)
        
        
    def print_to_text_area_photos_window(self, message):
        self.text_area_photos_area.config(state=tk.NORMAL)
        self.text_area_photos_area.insert(tk.END, message + "\n")
        self.text_area_photos_area.see(tk.END)
        self.text_area_photos_area.config(state=tk.DISABLED)
        
        
    def print_to_text_area_scrap_window(self, message):
        self.text_area_for_scrap.config(state=tk.NORMAL)
        self.text_area_for_scrap.insert(tk.END, message + "\n")
        self.text_area_for_scrap.see(tk.END)
        self.text_area_for_scrap.config(state=tk.DISABLED)
        
        
    def print_to_text_area_scrap_api_google(self, results):
        message = str(results)
        self.text_area_for_scrap_api.config(state=tk.NORMAL)
        self.text_area_for_scrap_api.insert(tk.END, message + "\n")
        self.text_area_for_scrap_api.see(tk.END)
        self.text_area_for_scrap_api.config(state=tk.DISABLED)
    
    def print_to_text_area_scrap_api_youtube(self, message):
        self.text_area_for_scrap_api_youtube.config(state=tk.NORMAL)
        self.text_area_for_scrap_api_youtube.insert(tk.END, message + "\n")
        self.text_area_for_scrap_api_youtube.see(tk.END)
        self.text_area_for_scrap_api_youtube.config(state=tk.DISABLED)
        
            
if __name__ == "__main__":
    app = App()
    app.app.mainloop()
