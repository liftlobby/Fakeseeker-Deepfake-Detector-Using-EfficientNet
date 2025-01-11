import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime
import numpy as np
from deepfake_detector import DeepfakeDetector
from face_extractor import FaceExtractor
import json
import shutil
from mss import mss

class FakeSeekerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FakeSeeker - Deepfake Detection Tool")
        self.root.geometry("1600x900")  # Adjusted for better resolution
        self.root.resizable(True, True)  # Allow resizing if needed

        # Set up styles
        self.setup_styles()
        
        # Initialize variables
        self.selected_file = None
        self.extracted_faces = None
        self.scan_history = []
        self.monitoring = False
        self.screenshot_interval = 5000  # 5 seconds
        
        # Set base directory and create necessary directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.reports_dir = os.path.join(self.base_dir, 'reports')
        self.thumbnails_dir = os.path.join(self.reports_dir, 'thumbnails')
        self.history_file = os.path.join(self.reports_dir, 'scan_history.json')
        self.images_dir = os.path.join(self.base_dir, 'images')
        
        # Create all necessary directories
        for directory in [self.models_dir, self.reports_dir, 
                         self.thumbnails_dir, self.images_dir]:
            try:
                os.makedirs(directory, exist_ok=True)
                # Test write permissions
                test_file = os.path.join(directory, '.test')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except Exception as e:
                messagebox.showerror("Error", 
                    f"Failed to create or access directory {directory}: {str(e)}\n"
                    "Please check permissions and try again.")
                self.root.destroy()
                return

        # Initialize components
        try:
            self.face_extractor = FaceExtractor()
            self.detector = DeepfakeDetector()
        except Exception as e:
            messagebox.showerror("Error", 
                f"Failed to initialize components: {str(e)}\n"
                "Please check if a trained model exists in the models directory.")
            self.root.destroy()
            return

        # Load icons
        icon_size = (64, 64)
        try:
            # Create default icons using PIL
            default_icon = Image.new('RGB', icon_size, color='lightgray')
            self.upload_icon = ImageTk.PhotoImage(default_icon)
            self.realtime_icon = ImageTk.PhotoImage(default_icon)
            self.results_icon = ImageTk.PhotoImage(default_icon)
            
            # Try to load actual icons if they exist
            icons_dir = os.path.join(self.base_dir, 'images')
            if os.path.exists(icons_dir):
                for icon_name, attr_name in [
                    ('upload.png', 'upload_icon'),
                    ('realtime.png', 'realtime_icon'),
                    ('results.png', 'results_icon')
                ]:
                    icon_path = os.path.join(icons_dir, icon_name)
                    if os.path.exists(icon_path):
                        icon = Image.open(icon_path)
                        icon = icon.resize(icon_size, Image.Resampling.LANCZOS)
                        setattr(self, attr_name, ImageTk.PhotoImage(icon))
        except Exception as e:
            print(f"Warning: Could not load some icons: {e}")
        
        # Load existing scan history
        self.load_scan_history()
        
        # Create main interface
        self.create_home_page()

    def setup_styles(self):
        """Set up custom styles for the application."""
        style = ttk.Style()
        
        # Configure common styles
        style.configure("TButton", padding=5, font=('Helvetica', 15))
        style.configure("TLabel", font=('Helvetica', 15))
        style.configure("Header.TLabel", font=('Helvetica', 28, 'bold'))
        style.configure("Title.TLabel", font=('Helvetica', 25, 'bold'))
        style.configure("Bold.TLabel", font=('Helvetica', 15, 'bold'))
        style.configure("Small.TLabel", font=('Helvetica', 12))
        style.configure("Card.TFrame", background='white', borderwidth=2, relief="solid")
        
        # Configure status label styles with colors
        style.configure("Real.TLabel", foreground='green', font=('Helvetica', 15, 'bold'))
        style.configure("Fake.TLabel", foreground='red', font=('Helvetica', 15, 'bold'))

    def load_scan_history(self):
        """Load scan history from JSON file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.scan_history = json.load(f)
            else:
                self.scan_history = []
        except Exception as e:
            print(f"Error loading scan history: {str(e)}")
            messagebox.showerror("Error", f"Failed to load scan history: {str(e)}")
            self.scan_history = []
        return self.scan_history

    def save_scan_history(self):
        """Save scan history to JSON file."""
        try:
            # Log current scan history state
            print("Current scan history:", self.scan_history)
            for i, scan in enumerate(self.scan_history):
                print(f"Scan {i}: {scan}")

            # Create reports directory if it doesn't exist
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            # Convert scan history to serializable format
            serializable_history = []
            for scan in self.scan_history:
                serializable_scan = {
                    'timestamp': scan['timestamp'],
                    'file_path': scan['file_path'],
                    'results': [float(r) for r in scan['results']],  # Convert numpy floats to Python floats
                    'face_thumbnails': scan['face_thumbnails']
                }
                serializable_history.append(serializable_scan)
            
            # Save to JSON file
            with open(self.history_file, 'w') as f:
                json.dump(serializable_history, f, indent=4)
            print("Scan history saved successfully.")
        except Exception as e:
            print(f"Error saving scan history: {str(e)}")
            messagebox.showerror("Error", f"Failed to save scan history: {str(e)}")

    def on_closing(self):
        """Save scan history before closing the application."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.stop_detection()
            if self.cap is not None:
                self.cap.release()
        if hasattr(self, 'face_extractor') and hasattr(self.face_extractor, 'temp_dir'):
            try:
                shutil.rmtree(self.face_extractor.temp_dir)  # Clean up temp directory
            except Exception as e:
                print(f"Error cleaning up temp directory: {str(e)}")
        self.save_scan_history()
        self.root.destroy()

    def clear_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def create_home_page(self):
        self.clear_frame()

        # Create main container
        main_container = ttk.Frame(self.root, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Label(main_container, 
                          text="Welcome to FakeSeeker", 
                          style='Header.TLabel')
        header.pack(pady=20)

        # Subheader
        subheader = ttk.Label(main_container,
                             text="Deepfake Detection Using EfficientNet",
                             style='Normal.TLabel')
        subheader.pack(pady=10)

        # Button container
        button_frame = ttk.Frame(main_container)
        button_frame.pack(pady=30)

        
        upload_btn = ttk.Button(button_frame, 
                                text="Upload Image/Video",
                                command=self.open_upload_page,
                                width=25,
                                image=self.upload_icon,
                                compound=tk.TOP)
        upload_btn.grid(row=0, column=0, padx=10)
        
        realtime_btn = ttk.Button(button_frame,
                                  text="Real-Time Detection",
                                  command=self.open_real_time_page,
                                  width=25,
                                  image=self.realtime_icon,
                                  compound=tk.TOP)
        realtime_btn.grid(row=0, column=1, padx=10)

        history_btn = ttk.Button(button_frame,
                                 text="View Results",
                                 command=self.show_scan_history,
                                 width=25,
                                 image=self.results_icon,
                                 compound=tk.TOP)
        history_btn.grid(row=0, column=2, padx=10)

        # Footer
        footer = ttk.Label(self.root,
                          text="Version Beta - Enhanced with EfficientNet-b2",
                          style='Normal.TLabel')
        footer.pack(side=tk.BOTTOM, pady=10)

    def open_upload_page(self):
        self.clear_frame()

        container = ttk.Frame(self.root, padding="20")
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(container, text="Upload Media for Analysis", style='Header.TLabel')
        header.pack(pady=20)

        # Preview frame
        self.preview_frame = ttk.Frame(container)
        self.preview_frame.pack(pady=20)
        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.pack()

        # Button frame
        button_frame = ttk.Frame(container)
        button_frame.pack(pady=20)

        upload_btn = ttk.Button(button_frame, text="Select File", command=self.upload_file)
        upload_btn.grid(row=0, column=0, padx=10)

        self.scan_btn = ttk.Button(button_frame, text="Start Scan", 
                                 command=self.start_scan, state=tk.DISABLED)
        self.scan_btn.grid(row=0, column=1, padx=10)

        back_btn = ttk.Button(button_frame, text="Back", command=self.create_home_page)
        back_btn.grid(row=0, column=2, padx=10)

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=[
                ("All Supported Files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov"),
                ("Image Files", "*.jpg *.jpeg *.png"),
                ("Video Files", "*.mp4 *.avi *.mov"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.selected_file = file_path
            self.show_preview(file_path)
            self.scan_btn.config(state=tk.NORMAL)

    def show_preview(self, file_path):
        try:
            if file_path.endswith(('.jpg', '.jpeg', '.png')):
                image = Image.open(file_path)
                image.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(image)
                self.preview_label.configure(image=photo)
                self.preview_label.image = photo
            else:
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2_im)
                    img.thumbnail((400, 400))
                    photo = ImageTk.PhotoImage(img)
                    self.preview_label.configure(image=photo)
                    self.preview_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preview: {str(e)}")

    def create_face_thumbnail(self, face, size=(150, 150)):
        """Create a thumbnail from a face image."""
        if isinstance(face, Image.Image):
            face = face.copy()
        else:
            face = Image.fromarray(face)
        
        face.thumbnail(size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(face)

    def start_scan(self):
        if not self.selected_file:
            messagebox.showwarning("No File Selected", "Please select a file first.")
            return

        try:
            file_path = self.selected_file
            
            # Clear preview frame and show status
            for widget in self.preview_frame.winfo_children():
                widget.destroy()

            # Create status label
            status_label = ttk.Label(self.preview_frame, text="Initializing...", 
                                   style='Normal.TLabel')
            status_label.pack(pady=20)
            self.root.update()

            def update_scan_status(message):
                status_label.config(text=message)
                self.root.update()

            # Process the file
            update_scan_status("Detecting faces...")
            
            # Extract faces using face extractor
            if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                faces = self.face_extractor.extract_faces_from_video(file_path)
            else:
                faces = self.face_extractor.extract_faces_from_image(file_path)
            
            if not faces:
                status_label.destroy()
                messagebox.showwarning("No Faces Detected", "No faces were detected in the file.")
                return

            # Save face thumbnails
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            thumbnails_dir = os.path.join(self.thumbnails_dir, timestamp)
            os.makedirs(thumbnails_dir, exist_ok=True)
            
            update_scan_status("Saving detected faces...")
            face_thumbnails = []
            for i, face in enumerate(faces):
                thumbnail_path = os.path.join(thumbnails_dir, f'face_{i}.png')
                face.save(thumbnail_path)
                # Store relative path from base directory
                rel_path = os.path.relpath(thumbnail_path, self.base_dir)
                face_thumbnails.append(rel_path)

            update_scan_status("Analyzing faces...")
            results = []
            for i, face_path in enumerate(face_thumbnails):
                update_scan_status(f"Analyzing face {i+1} of {len(face_thumbnails)}...")
                # Use absolute path for prediction
                abs_path = os.path.join(self.base_dir, face_path)
                result = self.detector.predict(abs_path)
                if result is not None:
                    results.append(float(result))
            
            if not results:
                status_label.destroy()
                messagebox.showerror("Error", "Failed to analyze faces.")
                return

            # Save results
            scan_data = {
                'timestamp': timestamp,
                'file_path': file_path,
                'results': results,
                'face_thumbnails': face_thumbnails
            }
            
            # Log scan data before appending
            print("Scan data to append:", scan_data)

            # Save to history
            self.scan_history.append(scan_data)
            self.save_scan_history()

            # Remove status label
            status_label.destroy()

            # Show the detailed report immediately
            self.show_detailed_report(scan_data)

        except Exception as e:
            if 'status_label' in locals():
                status_label.destroy()
            messagebox.showerror("Error", f"An error occurred while processing the file: {str(e)}")

    def show_detailed_report(self, scan_data):
        """Display detailed report in the main window."""
        # First unbind any existing mousewheel bindings
        self.root.unbind_all("<MouseWheel>")
        
        self.clear_frame()

        # Create main container with padding
        main_container = ttk.Frame(self.root, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Create a canvas with scrollbar
        canvas = tk.Canvas(main_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # Configure the canvas
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Header with larger font and padding
        header = ttk.Label(scrollable_frame, text="Detailed Scan Report", 
                          style='Header.TLabel', font=('Helvetica', 16, 'bold'))
        header.pack(pady=(0, 20))

        # Navigation buttons frame with better spacing
        nav_frame = ttk.Frame(scrollable_frame)
        nav_frame.pack(pady=(0, 30), fill=tk.X)

        # Center the buttons
        button_frame = ttk.Frame(nav_frame)
        button_frame.pack(anchor=tk.CENTER)

        # Back to scan button
        back_btn = ttk.Button(button_frame, text="Back to Scan", 
                            command=lambda: [self.root.unbind_all("<MouseWheel>"), 
                                           self.open_upload_page()])
        back_btn.pack(side=tk.LEFT, padx=10)

        # View History button
        history_btn = ttk.Button(button_frame, text="View History", 
                               command=lambda: [self.root.unbind_all("<MouseWheel>"), 
                                              self.show_scan_history()])
        history_btn.pack(side=tk.LEFT, padx=10)

        # Content sections with better spacing and borders
        # File details section
        details_frame = ttk.LabelFrame(scrollable_frame, text="File Details", padding="15")
        details_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        file_details = self.get_file_details(scan_data['file_path'])
        for key, value in file_details.items():
            detail_label = ttk.Label(details_frame, text=f"{key}: {value}")
            detail_label.pack(anchor=tk.W, pady=2)

        # Results summary section
        results_frame = ttk.LabelFrame(scrollable_frame, text="Detection Results", padding="15")
        results_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        # Calculate average probability
        avg_prob = sum(scan_data['results']) / len(scan_data['results'])
        status = "Potential Deepfake" if avg_prob > 0.5 else "Likely Real"
        status_style = 'Fake.TLabel' if avg_prob > 0.5 else 'Real.TLabel'
        
        # Create probability bar for average
        self.create_probability_bar(results_frame, "Average Probability", avg_prob)
        status_label = ttk.Label(results_frame, 
                               text=f"Status: {status}",
                               style=status_style)
        status_label.pack(pady=(10, 0))

        # Individual faces section
        faces_frame = ttk.LabelFrame(scrollable_frame, text="Detected Faces", padding="15")
        faces_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        # Grid layout for faces
        grid_frame = ttk.Frame(faces_frame)
        grid_frame.pack(fill=tk.BOTH, expand=True)

        # Create a frame for face thumbnails and probabilities
        row = 0
        col = 0
        max_cols = 2  # Number of faces per row

        for i, (rel_path, prob) in enumerate(zip(scan_data['face_thumbnails'], scan_data['results'])):
            face_container = ttk.Frame(grid_frame, padding=5)
            face_container.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            try:
                # Convert relative path to absolute path
                abs_path = os.path.join(self.base_dir, rel_path)
                img = Image.open(abs_path)
                img.thumbnail((150, 150))  # Larger thumbnails
                photo = ImageTk.PhotoImage(img)
                
                img_label = ttk.Label(face_container, image=photo)
                img_label.image = photo  # Keep a reference
                img_label.pack(pady=(0, 10))
                
                # Create probability bar
                self.create_probability_bar(face_container, f"Face {i+1}", prob)
                
            except Exception as e:
                print(f"Error loading thumbnail {abs_path}: {e}")
                error_label = ttk.Label(face_container, text=f"Error loading face {i+1}")
                error_label.pack(pady=5)

            # Update grid position
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # Configure grid weights
        for i in range(max_cols):
            grid_frame.columnconfigure(i, weight=1)

        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=(0, 10))
        scrollbar.pack(side="right", fill="y")

        # Configure mouse wheel scrolling
        self.root.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

    def get_file_details(self, file_path):
        """Get detailed information about the file."""
        details = {}
        try:
            # Basic file info
            file_stats = os.stat(file_path)
            details['size'] = self.format_file_size(file_stats.st_size)
            details['type'] = os.path.splitext(file_path)[1].upper()[1:]  # Remove the dot
            
            # Get image/video resolution
            if details['type'].lower() in ['jpg', 'jpeg', 'png', 'bmp']:
                with Image.open(file_path) as img:
                    details['resolution'] = f"{img.width}x{img.height}"
            elif details['type'].lower() in ['mp4', 'avi', 'mov']:
                cap = cv2.VideoCapture(file_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                details['resolution'] = f"{width}x{height} @ {fps}fps"
                cap.release()
            
            # Get file creation and modification times
            details['created'] = datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            details['modified'] = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            # Get file source (directory path)
            details['source'] = os.path.dirname(file_path)
            
        except Exception as e:
            print(f"Error getting file details: {e}")
        
        return details

    def format_file_size(self, size_in_bytes):
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_in_bytes < 1024:
                return f"{size_in_bytes:.1f} {unit}"
            size_in_bytes /= 1024
        return f"{size_in_bytes:.1f} TB"

    def create_probability_bar(self, parent, label, probability):
        """Create a probability bar with label and percentage."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        # Label
        label_text = f"{label}: {probability*100:.1f}%"
        label = ttk.Label(frame, text=label_text, width=30, anchor=tk.W)
        label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Progress bar frame
        bar_frame = ttk.Frame(frame)
        bar_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Progress bar
        bar = ttk.Progressbar(bar_frame, length=200, mode='determinate')
        bar['value'] = probability * 100
        bar.pack(side=tk.LEFT, padx=(0, 10))
        
        # Percentage label
        percentage = ttk.Label(frame, text=f"{probability*100:.1f}%")
        percentage.pack(side=tk.LEFT)
        
        return frame

    def show_scan_history(self):
        """Display scan history in the main window."""
        self.clear_frame()

        # Create main container
        main_container = ttk.Frame(self.root, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Label(main_container, text="Scan History", style='Header.TLabel')
        header.pack(pady=(0, 20))

        # Back button
        back_btn = ttk.Button(main_container, text="Back", command=self.create_home_page)
        back_btn.pack(pady=(0, 20))

        # Create scrollable frame
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Configure grid for boxes (2 columns)
        scrollable_frame.grid_columnconfigure(0, weight=1, pad=10)
        scrollable_frame.grid_columnconfigure(1, weight=1, pad=10)

        # Display history items in boxes
        if not self.scan_history:
            ttk.Label(scrollable_frame, text="No scan history available", style='Normal.TLabel').grid(
                row=0, column=0, columnspan=2, pady=20)
        else:
            for i, scan in enumerate(reversed(self.scan_history)):
                row = i // 2
                col = i % 2

                # Create box frame with border
                box_frame = ttk.Frame(scrollable_frame, style='Card.TFrame')
                box_frame.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)
                box_frame.grid_columnconfigure(0, weight=1)

                # Box content
                content_frame = ttk.Frame(box_frame, padding=10)
                content_frame.pack(fill=tk.BOTH, expand=True)

                # Status with larger font
                avg_prob = sum(scan['results']) / len(scan['results'])
                status_style = 'Fake.TLabel' if avg_prob > 0.5 else 'Real.TLabel'
                ttk.Label(content_frame, text="Potential Deepfake" if avg_prob > 0.5 else "Likely Real", 
                         style=status_style).pack(anchor="w")

                # File name
                ttk.Label(content_frame, 
                         text=os.path.basename(scan['file_path']),
                         style='Normal.TLabel').pack(anchor="w", pady=5)

                # Timestamp
                ttk.Label(content_frame, 
                         text=f"Scanned: {scan['timestamp']}",
                         style='Small.TLabel').pack(anchor="w")

                # Number of faces
                ttk.Label(content_frame,
                         text=f"Faces detected: {len(scan['results'])}",
                         style='Small.TLabel').pack(anchor="w")

                # Confidence
                ttk.Label(content_frame,
                         text=f"Average confidence: {sum(scan['results']) / len(scan['results']):.1%}",
                         style='Small.TLabel').pack(anchor="w")

                # Buttons frame
                buttons_frame = ttk.Frame(content_frame)
                buttons_frame.pack(fill=tk.X, pady=(10, 0))

                # View button
                view_btn = ttk.Button(buttons_frame, text="View Details",
                                    command=lambda s=scan: self.show_detailed_report(s))
                view_btn.pack(side=tk.LEFT, padx=5)

                # Delete button
                delete_btn = ttk.Button(buttons_frame, text="Delete",
                                      command=lambda s=scan: self.delete_scan(s))
                delete_btn.pack(side=tk.LEFT, padx=5)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Configure mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(-1*(event.delta//120), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # Cleanup bindings when leaving page
        def cleanup_bindings():
            canvas.unbind_all("<MouseWheel>")
            self.root.update()

        back_btn.configure(command=lambda: [cleanup_bindings(), self.create_home_page()])

    def delete_scan(self, scan_data):
        """Delete a scan from history and its associated files."""
        try:
            # Remove thumbnails
            if 'face_thumbnails' in scan_data:
                for thumbnail in scan_data['face_thumbnails']:
                    try:
                        if os.path.exists(thumbnail):
                            os.remove(thumbnail)
                    except Exception as e:
                        print(f"Error removing thumbnail {thumbnail}: {e}")

                # Remove thumbnail directory
                try:
                    thumbnail_dir = os.path.dirname(scan_data['face_thumbnails'][0])
                    if os.path.exists(thumbnail_dir):
                        # Check if directory is empty
                        if not os.listdir(thumbnail_dir):
                            os.rmdir(thumbnail_dir)
                except Exception as e:
                    print(f"Error removing thumbnail directory: {e}")

            # Remove from history
            self.scan_history = [scan for scan in self.scan_history if scan['timestamp'] != scan_data['timestamp']]

            # Save updated history
            self.save_scan_history()

            # Refresh the display
            self.show_scan_history()

            messagebox.showinfo("Success", "Scan deleted successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Error deleting scan: {e}")

    def open_real_time_page(self):
        self.clear_frame()
        
        # Initialize camera variables if not already done
        if not hasattr(self, 'detection_active'):
            self.detection_active = False
        if not hasattr(self, 'screen_monitoring'):
            self.screen_monitoring = False
        
        container = ttk.Frame(self.root, padding="20")
        container.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Label(container, text="Real-Time Detection", style='Header.TLabel')
        header.pack(pady=20)

        # Control buttons frame
        control_frame = ttk.Frame(container)
        control_frame.pack(pady=10)

        # Camera toggle button
        self.camera_toggle_btn = ttk.Button(
            control_frame,
            text="Turn Camera On" if not hasattr(self, 'camera_on') or not self.camera_on else "Turn Camera Off",
            command=self.toggle_camera,
            width=20
        )
        self.camera_toggle_btn.pack(side=tk.LEFT, padx=5)

        # Screen monitoring toggle button
        self.screen_toggle_btn = ttk.Button(
            control_frame,
            text="Start Screen Monitor",
            command=self.toggle_screen_monitoring,
            width=20
        )
        self.screen_toggle_btn.pack(side=tk.LEFT, padx=5)

        # Detection control buttons
        self.start_detection_btn = ttk.Button(
            control_frame,
            text="Start Detection",
            command=self.start_detection,
            width=20,
            state='disabled'  # Initially disabled until camera is on
        )
        self.start_detection_btn.pack(side=tk.LEFT, padx=5)

        self.stop_detection_btn = ttk.Button(
            control_frame,
            text="Stop Detection",
            command=self.stop_detection,
            width=20,
            state='disabled'  # Initially disabled
        )
        self.stop_detection_btn.pack(side=tk.LEFT, padx=5)

        # Video frame
        self.video_frame = ttk.Frame(container, borderwidth=2, relief="solid")
        self.video_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Video label
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(pady=10)

        # Back button
        back_btn = ttk.Button(container, text="Back", command=self.back_to_home_from_realtime)
        back_btn.pack(pady=20)

    def toggle_camera(self):
        if not hasattr(self, 'camera_on') or not self.camera_on:
            # Turn camera on
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Added CAP_DSHOW for Windows
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not access the camera.")
                return
            self.camera_on = True
            self.camera_toggle_btn.configure(text="Turn Camera Off")
            self.start_detection_btn.configure(state='normal')
            self.update_camera_feed()
        else:
            # Turn camera off
            self.stop_detection()  # Stop detection if running
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            self.camera_on = False
            self.camera_toggle_btn.configure(text="Turn Camera On")
            self.start_detection_btn.configure(state='disabled')
            self.stop_detection_btn.configure(state='disabled')
            # Clear the video label
            self.video_label.configure(image='')

    def start_detection(self):
        self.detection_active = True
        self.start_detection_btn.configure(state='disabled')
        self.stop_detection_btn.configure(state='normal')

    def stop_detection(self):
        self.detection_active = False
        if hasattr(self, 'camera_on') and self.camera_on:
            self.start_detection_btn.configure(state='normal')
        self.stop_detection_btn.configure(state='disabled')

    def update_camera_feed(self):
        """Update the camera feed and handle face detection."""
        if not hasattr(self, 'camera_on') or not self.camera_on:
            return

        try:
            ret, frame = self.cap.read()
            if not ret:
                return

            # Flip the frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.detection_active:
                # Convert to PIL Image for face detection
                pil_image = Image.fromarray(rgb_frame)
                
                # Get face locations
                face_locations = self.face_extractor.mtcnn.detect(pil_image)
                
                if face_locations[0] is not None:
                    boxes = face_locations[0]
                    for box in boxes:
                        # Convert box coordinates to integers
                        box = [int(b) for b in box]
                        
                        # Extract and analyze the face
                        face = pil_image.crop((box[0], box[1], box[2], box[3]))
                        
                        # Save face temporarily and get prediction
                        temp_path = os.path.join(self.face_extractor.temp_dir, 'temp_face.jpg')
                        face.save(temp_path)
                        probability = self.detector.predict(temp_path)
                        
                        if probability is not None:
                            # Draw rectangle and text
                            result = "FAKE" if probability > 0.5 else "REAL"
                            color = (0, 0, 255) if probability > 0.5 else (0, 255, 0)
                            confidence = max(probability, 1 - probability) * 100
                            text = f"{result} ({confidence:.1f}% confidence)"
                            
                            # Convert color from RGB to BGR for OpenCV
                            color = color[::-1]
                            
                            # Draw on the frame
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                            cv2.putText(frame, text, (box[0], box[1] - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Convert to RGB for display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and resize
            image = Image.fromarray(display_frame)
            image.thumbnail((800, 600), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=image)
            
            # Update the video label
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error in camera feed: {str(e)}")
        finally:
            # Schedule next update if camera is still on
            if hasattr(self, 'camera_on') and self.camera_on:
                self.root.after(10, self.update_camera_feed)

    def toggle_screen_monitoring(self):
        if not hasattr(self, 'screen_monitoring') or not self.screen_monitoring:
            # Start screen monitoring
            self.screen_monitoring = True
            self.camera_toggle_btn.configure(state='disabled')  # Disable camera while screen monitoring
            self.screen_toggle_btn.configure(text="Stop Screen Monitor")
            self.start_detection_btn.configure(state='normal')
            self.update_screen_feed()
        else:
            # Stop screen monitoring
            self.stop_detection()  # Stop detection if running
            self.screen_monitoring = False
            self.camera_toggle_btn.configure(state='normal')
            self.screen_toggle_btn.configure(text="Start Screen Monitor")
            self.start_detection_btn.configure(state='disabled')
            self.video_label.configure(image='')

    def update_screen_feed(self):
        """Update the screen feed and handle face detection."""
        if not hasattr(self, 'screen_monitoring') or not self.screen_monitoring:
            return

        try:
            with mss() as sct:
                # Capture the primary monitor
                monitor = sct.monitors[1]  # Primary monitor
                screenshot = sct.grab(monitor)
                
                # Convert to numpy array
                frame = np.array(screenshot)
                
                # Convert from BGRA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                if hasattr(self, 'detection_active') and self.detection_active:
                    # Extract and analyze faces if detection is active
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    faces = self.face_extractor.extract_faces_from_image_realtime(pil_image)
                    
                    for face in faces:
                        # Convert PIL face image to numpy array for OpenCV
                        face_np = np.array(face)
                        # Get prediction
                        temp_path = os.path.join(self.face_extractor.temp_dir, 'temp_face.jpg')
                        face.save(temp_path)
                        probability = self.detector.predict(temp_path)
                        
                        if probability is not None:
                            # Draw rectangle and text
                            result = "FAKE" if probability > 0.5 else "REAL"
                            color = (0, 0, 255) if probability > 0.5 else (0, 255, 0)
                            confidence = max(probability, 1 - probability) * 100
                            text = f"{result} ({confidence:.1f}% confidence)"
                            
                            # Get face location
                            face_location = self.face_extractor.get_face_location(rgb_frame)
                            if face_location:
                                top, right, bottom, left = face_location
                                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                                cv2.putText(frame, text, (left, top - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Resize frame to fit display
                display_size = (700, 600)
                frame = cv2.resize(frame, display_size)
                
                # Convert to PhotoImage
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(img)
                
                self.video_label.configure(image=photo)
                self.video_label.image = photo  # Keep a reference

        except Exception as e:
            print(f"Error in screen capture: {str(e)}")
            self.toggle_screen_monitoring()  # Stop monitoring on error
            return

        # Schedule the next update
        self.root.after(100, self.update_screen_feed)  # Slightly longer delay for screen capture

    def back_to_home_from_realtime(self):
        """Safely handle camera shutdown when returning to home page."""
        if hasattr(self, 'camera_on') and self.camera_on:
            self.stop_detection()  # First stop detection
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
        if hasattr(self, 'screen_monitoring') and self.screen_monitoring:
            self.toggle_screen_monitoring()
        self.create_home_page()

if __name__ == "__main__":
    root = tk.Tk()
    app = FakeSeekerApp(root)
    root.mainloop()