import cv2
import numpy as np
import insightface
import customtkinter as ctk
from PIL import Image, ImageOps
import os
import pyvirtualcam
from pyvirtualcam import PixelFormat
import torch
import time
from threading import Thread, Lock
from queue import Queue
import traceback
from pygrabber.dshow_graph import FilterGraph
import json
from CTkMessagebox import CTkMessagebox
import webbrowser
from datetime import datetime

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.fastest = True

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class ModernButton(ctk.CTkButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure(
            corner_radius=10,
            border_spacing=10,
            fg_color="#2b2b2b",
            hover_color="#3b3b3b",
            border_color="#00ff9d",
            border_width=2,
            text_color="#ffffff",
            height=40,
            font=("Helvetica", 13, "bold")
        )

class ModernFrame(ctk.CTkFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure(
            corner_radius=15,
            fg_color="#1e1e1e",
            border_color="#2d2d2d",
            border_width=2
        )

class ModernLabel(ctk.CTkLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure(
            font=("Helvetica", 13),
            text_color="#e0e0e0"
        )

class FaceSwapper:
    def __init__(self):
        self.root = None
        self.source_path = None
        self.source_label = None
        self.face_analyser = None
        self.swapper = None
        self.running = False
        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.last_update = 0
        self.fps = 0
        self.lock = Lock()
        self.last_frame = None
        self.error_count = 0
        self.max_errors = 5
        self.recovery_time = 1.0
        self.last_error_time = 0
        self.source_face = None
        self.process_threads = []
        self.num_threads = 3
        self.camera_index = 0
        self.cameras = self.get_available_cameras()
        self.settings = self.load_settings()
        self.available_models = self.get_available_models()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)
            self.device = torch.device("cuda:0")
            torch.cuda.set_per_process_memory_fraction(0.95)
            
    def get_available_models(self):
        models = []
        for file in os.listdir():
            if file.endswith(".onnx"):
                models.append(file)
        return models if models else ["inswapper_128_fp16.onnx"]

    def load_settings(self):
        default_settings = {
            "width": 1280,
            "height": 720,
            "fps": 60,
            "num_threads": 3,
            "buffer_size": 4,
            "det_size": 640,
            "skip_frames": 0,
            "model": "inswapper_128_fp16.onnx"
        }
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", "r") as f:
                    return {**default_settings, **json.load(f)}
        except:
            pass
        return default_settings

    def save_settings(self):
        try:
            with open("settings.json", "w") as f:
                json.dump(self.settings, f)
        except:
            pass

    def get_available_cameras(self):
        devices = FilterGraph().get_input_devices()
        return [{"index": i, "name": name} for i, name in enumerate(devices)]

    def init_ui(self):
        self.root = ctk.CTk()
        self.root.title("FaceSwap Studio Pro")
        self.root.geometry("400x800")
        self.root.resizable(False, False)
        self.root.configure(fg_color="#171717")
        
        try:
            providers = ['CUDAExecutionProvider']
            self.face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
            self.face_analyser.prepare(ctx_id=0, det_size=(self.settings["det_size"], self.settings["det_size"]))
            model_path = self.settings["model"]
            if not os.path.exists(model_path):
                self.show_error("Error", f"Model file not found: {model_path}")
                return None
            self.swapper = insightface.model_zoo.get_model(model_path, providers=providers)
        except Exception as e:
            self.show_error("Error", f"Failed to initialize models: {str(e)}")
            return None
        
        main_frame = ModernFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        header_frame = ModernFrame(main_frame)
        header_frame.pack(fill="x", padx=15, pady=(15,5))
        
        title = ModernLabel(
            header_frame, 
            text="FaceSwap Studio Pro",
            font=("Helvetica", 24, "bold"),
            text_color="#00ff9d"
        )
        title.pack(pady=10)
        
        subtitle = ModernLabel(
            header_frame,
            text="Professional Real-time Face Swapping",
            font=("Helvetica", 12),
            text_color="#888888"
        )
        subtitle.pack(pady=(0,10))
        
        source_frame = ModernFrame(main_frame)
        source_frame.pack(fill="x", padx=15, pady=10)
        
        self.source_label = ModernLabel(source_frame, text="", width=250, height=250)
        self.source_label.pack(pady=15)
        
        controls_frame = ModernFrame(main_frame)
        controls_frame.pack(fill="x", padx=15, pady=10)
        
        camera_label = ModernLabel(controls_frame, text="Camera Source")
        camera_label.pack(pady=(10,5))
        
        camera_options = [camera["name"] for camera in self.cameras]
        self.camera_var = ctk.StringVar(value=camera_options[0] if camera_options else "No cameras found")
        camera_menu = ctk.CTkOptionMenu(
            controls_frame,
            variable=self.camera_var,
            values=camera_options,
            command=self.update_camera,
            width=300,
            fg_color="#2b2b2b",
            button_color="#00ff9d",
            button_hover_color="#00cc7a",
            dropdown_fg_color="#2b2b2b",
            font=("Helvetica", 12)
        )
        camera_menu.pack(pady=5)
        
        buttons_frame = ModernFrame(main_frame)
        buttons_frame.pack(fill="x", padx=15, pady=10)
        
        select_source = ModernButton(
            buttons_frame,
            text="Select Source Face",
            command=self.select_source,
            fg_color="#2b2b2b",
            hover_color="#00cc7a"
        )
        select_source.pack(pady=5, fill="x")
        
        self.toggle_button = ModernButton(
            buttons_frame,
            text="Start Processing",
            command=self.toggle_processing,
            fg_color="#00ff9d",
            hover_color="#00cc7a",
            text_color="#000000"
        )
        self.toggle_button.pack(pady=5, fill="x")
        
        settings_button = ModernButton(
            buttons_frame,
            text="Advanced Settings",
            command=self.show_settings_window
        )
        settings_button.pack(pady=5, fill="x")
        
        status_frame = ModernFrame(main_frame)
        status_frame.pack(fill="x", padx=15, pady=10)
        
        metrics_frame = ctk.CTkFrame(status_frame, fg_color="transparent")
        metrics_frame.pack(fill="x", padx=10, pady=10)
        
        self.status_label = ModernLabel(
            metrics_frame,
            text="Status: Ready",
            font=("Helvetica", 12),
            text_color="#888888"
        )
        self.status_label.pack(side="left")
        
        self.fps_label = ModernLabel(
            metrics_frame,
            text="FPS: 0",
            font=("Helvetica", 12, "bold"),
            text_color="#00ff9d"
        )
        self.fps_label.pack(side="right")

        footer_frame = ModernFrame(main_frame)
        footer_frame.pack(fill="x", padx=15, pady=(0,10))
        
        footer_text = ModernLabel(
            footer_frame,
            text=f"© {datetime.now().year} FaceSwap Studio Pro",
            font=("Helvetica", 10),
            text_color="#666666"
        )
        footer_text.pack(pady=5)
        
        return self.root
        
    def show_settings_window(self):
        if self.running:
            self.show_error("Error", "Stop processing before changing settings")
            return
            
        settings_window = ctk.CTkToplevel(self.root)
        settings_window.title("Advanced Settings")
        settings_window.geometry("500x800")
        settings_window.resizable(False, False)
        settings_window.configure(fg_color="#171717")

        main_frame = ctk.CTkScrollableFrame(settings_window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        title = ModernLabel(main_frame, text="Advanced Settings", font=("Helvetica", 20, "bold"), text_color="#00ff9d")
        title.pack(pady=(0,20))

        model_frame = ModernFrame(main_frame)
        model_frame.pack(fill="x", pady=10)
        
        ModernLabel(model_frame, text="Model Selection", font=("Helvetica", 14, "bold")).pack(pady=5)
        ModernLabel(model_frame, text="Higher number = better quality, lower FPS", text_color="#888888").pack()
        
        model_var = ctk.StringVar(value=self.settings["model"])
        model_menu = ctk.CTkOptionMenu(
            model_frame,
            variable=model_var,
            values=self.available_models,
            width=300,
            fg_color="#2b2b2b",
            button_color="#00ff9d",
            button_hover_color="#00cc7a"
        )
        model_menu.pack(pady=10)

        resolution_frame = ModernFrame(main_frame)
        resolution_frame.pack(fill="x", pady=10)
        
        ModernLabel(resolution_frame, text="Camera Resolution", font=("Helvetica", 14, "bold")).pack(pady=5)
        ModernLabel(resolution_frame, text="Higher = better quality, lower FPS", text_color="#888888").pack()
        
        res_frame = ctk.CTkFrame(resolution_frame, fg_color="transparent")
        res_frame.pack(pady=10)
        
        width_var = ctk.StringVar(value=str(self.settings["width"]))
        height_var = ctk.StringVar(value=str(self.settings["height"]))
        
        width_entry = ctk.CTkEntry(res_frame, textvariable=width_var, width=100, fg_color="#2b2b2b")
        width_entry.pack(side="left", padx=5)
        
        ModernLabel(res_frame, text="x").pack(side="left", padx=5)
        
        height_entry = ctk.CTkEntry(res_frame, textvariable=height_var, width=100, fg_color="#2b2b2b")
        height_entry.pack(side="left", padx=5)

        performance_frame = ModernFrame(main_frame)
        performance_frame.pack(fill="x", pady=10)
        
        ModernLabel(performance_frame, text="Performance Settings", font=("Helvetica", 14, "bold")).pack(pady=5)

        ModernLabel(performance_frame, text="Target FPS").pack(pady=(10,0))
        fps_var = ctk.StringVar(value=str(self.settings["fps"]))
        fps_slider = ctk.CTkSlider(performance_frame, from_=15, to=120, number_of_steps=105)
        fps_slider.set(self.settings["fps"])
        fps_slider.pack(pady=(0,10), padx=20)

        ModernLabel(performance_frame, text="Processing Threads").pack(pady=(10,0))
        threads_var = ctk.StringVar(value=str(self.settings["num_threads"]))
        threads_slider = ctk.CTkSlider(performance_frame, from_=1, to=12, number_of_steps=11)
        threads_slider.set(self.settings["num_threads"])
        threads_slider.pack(pady=(0,10), padx=20)

        ModernLabel(performance_frame, text="Frame Buffer Size").pack(pady=(10,0))
        buffer_var = ctk.StringVar(value=str(self.settings["buffer_size"]))
        buffer_slider = ctk.CTkSlider(performance_frame, from_=1, to=8, number_of_steps=7)
        buffer_slider.set(self.settings["buffer_size"])
        buffer_slider.pack(pady=(0,10), padx=20)

        detection_frame = ModernFrame(main_frame)
        detection_frame.pack(fill="x", pady=10)
        
        ModernLabel(detection_frame, text="Detection Settings", font=("Helvetica", 14, "bold")).pack(pady=5)

        ModernLabel(detection_frame, text="Face Detection Size").pack(pady=(10,0))
        det_size_var = ctk.StringVar(value=str(self.settings["det_size"]))
        det_size_slider = ctk.CTkSlider(detection_frame, from_=320, to=960, number_of_steps=8)
        det_size_slider.set(self.settings["det_size"])
        det_size_slider.pack(pady=(0,10), padx=20)

        ModernLabel(detection_frame, text="Skip Frames").pack(pady=(10,0))
        skip_frames_var = ctk.StringVar(value=str(self.settings["skip_frames"]))
        skip_frames_slider = ctk.CTkSlider(detection_frame, from_=0, to=4, number_of_steps=4)
        skip_frames_slider.set(self.settings["skip_frames"])
        skip_frames_slider.pack(pady=(0,10), padx=20)

        def save():
            try:
                self.settings["model"] = model_var.get()
                self.settings["width"] = int(width_var.get())
                self.settings["height"] = int(height_var.get())
                self.settings["fps"] = int(fps_slider.get())
                self.settings["num_threads"] = int(threads_slider.get())
                self.settings["buffer_size"] = int(buffer_slider.get())
                self.settings["det_size"] = int(det_size_slider.get())
                self.settings["skip_frames"] = int(skip_frames_slider.get())
                self.save_settings()
                self.num_threads = self.settings["num_threads"]
                
                if self.face_analyser:
                    self.face_analyser.prepare(ctx_id=0, det_size=(self.settings["det_size"], self.settings["det_size"]))
                    try:
                        providers = ['CUDAExecutionProvider']
                        self.swapper = insightface.model_zoo.get_model(self.settings["model"], providers=providers)
                    except:
                        self.show_error("Error", "Failed to load model")
                        
                settings_window.destroy()
                self.show_success("Success", "Settings saved successfully")
            except:
                self.show_error("Error", "Failed to save settings")

        save_button = ModernButton(
            main_frame,
            text="Save Settings",
            command=save,
            fg_color="#00ff9d",
            hover_color="#00cc7a",
            text_color="#000000"
        )
        save_button.pack(pady=20)
        
    def show_error(self, title, message):
        CTkMessagebox(
            title=title,
            message=message,
            icon="cancel",
            option_1="OK"
        )

    def show_success(self, title, message):
        CTkMessagebox(
            title=title,
            message=message,
            icon="check",
            option_1="OK"
        )

    def select_source(self):
        try:
            path = ctk.filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
            if path:
                self.source_path = path
                source_image = cv2.imread(path)
                source_faces = self.face_analyser.get(source_image)
                if len(source_faces) > 0:
                    self.source_face = source_faces[0]
                    image = Image.open(path)
                    photo = ctk.CTkImage(image, size=(250, 250))
                    self.source_label.configure(image=photo)
                    self.update_status("Source face selected")
                else:
                    self.show_error("Error", "No face detected in source image")
        except Exception as e:
            self.show_error("Error", f"Failed to select source: {str(e)}")
    
    def update_camera(self, selection):
        try:
            for camera in self.cameras:
                if camera["name"] == selection:
                    self.camera_index = camera["index"]
                    self.update_status(f"Camera selected: {selection}")
                    break
        except Exception as e:
            self.show_error("Error", f"Failed to update camera: {str(e)}")
    
    def update_status(self, message):
        try:
            self.status_label.configure(text=f"Status: {message}")
            self.root.update()
        except:
            pass
            
    def process_frame(self, frame):
        try:
            if frame is None or self.source_face is None:
                return frame
                
            target_faces = self.face_analyser.get(frame)
            
            if target_faces:
                result = frame.copy()
                for face in target_faces:
                    try:
                        result = self.swapper.get(result, face, self.source_face, paste_back=True)
                    except:
                        continue
                return result
            return frame
        except:
            return frame
            
    def process_frames_thread(self, thread_id):
        while self.running:
            try:
                frame = None
                with self.lock:
                    if not self.frame_queue.empty():
                        frame = self.frame_queue.get_nowait()
                
                if frame is not None:
                    result = self.process_frame(frame)
                    if result is not None:
                        with self.lock:
                            if self.result_queue.full():
                                try:
                                    self.result_queue.get_nowait()
                                except:
                                    pass
                            self.result_queue.put(result)
                else:
                    time.sleep(0.001)
            except:
                continue

    def toggle_processing(self):
        if not self.running:
            self.toggle_button.configure(text="Stop")
            self.toggle_button.configure(fg_color="#ff3333", hover_color="#cc0000")
            self.safe_start_swap()
        else:
            self.toggle_button.configure(text="Start")
            self.toggle_button.configure(fg_color="#00ff9d", hover_color="#00cc7a")
            self.stop_swap()
    
    def safe_start_swap(self):
        try:
            self.start_swap()
        except Exception as e:
            self.show_error("Error", f"Failed to start: {str(e)}")
            self.stop_swap()
    
    def start_swap(self):
        if self.source_face is None:
            self.show_error("Error", "Please select source face first")
            return
            
        self.running = True
        self.error_count = 0
        self.last_error_time = 0
        self.update_status("Starting...")
        
        try:
            capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings["width"])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings["height"])
            capture.set(cv2.CAP_PROP_FPS, self.settings["fps"])
            capture.set(cv2.CAP_PROP_BUFFERSIZE, self.settings["buffer_size"])
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            
            if not capture.isOpened():
                self.show_error("Error", "Failed to open camera")
                return
                
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(capture.get(cv2.CAP_PROP_FPS))
            
            self.process_threads = []
            for i in range(self.num_threads):
                thread = Thread(target=self.process_frames_thread, args=(i,))
                thread.daemon = True
                thread.start()
                self.process_threads.append(thread)
            
            with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=PixelFormat.BGR) as camera:
                self.update_status("Running")
                frame_count = 0
                start_time = time.time()
                last_fps_update = time.time()
                skip_count = 0
                
                while self.running and capture.isOpened():
                    try:
                        ret, frame = capture.read()
                        if not ret:
                            continue
                            
                        if skip_count < self.settings["skip_frames"]:
                            skip_count += 1
                            continue
                        skip_count = 0
                            
                        with self.lock:
                            if not self.frame_queue.full():
                                self.frame_queue.put(frame)
                            
                            if not self.result_queue.empty():
                                result = self.result_queue.get()
                                camera.send(result)
                                    
                        frame_count += 1
                        current_time = time.time()
                        
                        if current_time - last_fps_update >= 0.5:
                            elapsed_time = current_time - start_time
                            self.fps = frame_count / elapsed_time
                            self.fps_label.configure(text=f"FPS: {self.fps:.1f}")
                            last_fps_update = current_time
                            
                        self.root.update()
                        
                    except:
                        continue
                        
                capture.release()
                
        except Exception as e:
            self.show_error("Error", f"Processing error: {str(e)}")
            
        finally:
            self.running = False
            for thread in self.process_threads:
                thread.join(timeout=1.0)
            torch.cuda.empty_cache()
            self.update_status("Stopped")
    
    def stop_swap(self):
        self.running = False
        torch.cuda.empty_cache()
        self.update_status("Stopped")
                
    def run(self):
        try:
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
            else:
                print("CUDA not available, using CPU")
                
            root = self.init_ui()
            root.mainloop()
        except Exception as e:
            self.show_error("Fatal Error", str(e))
            print(traceback.format_exc())

if __name__ == "__main__":
    app = FaceSwapper()
    app.run()