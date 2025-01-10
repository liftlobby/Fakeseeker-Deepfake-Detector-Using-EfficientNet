from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from typing import List, Optional, Tuple

class FaceExtractor:
    def __init__(self):
        """Initialize the MTCNN face detector with optimal parameters for deepfake detection."""
        self.mtcnn = MTCNN(
            margin=14,        # Add margin around faces
            keep_all=True,    # Keep all faces found
            factor=0.7,       # For faster detection
            device='cpu'      # Use CPU for wider compatibility
        )
        # Create temporary directory for extracted faces
        self.temp_dir = tempfile.mkdtemp()

    def extract_faces_from_image(self, image_path: str) -> List[Image.Image]:
        """
        Extract faces from a single image file.
        Returns list of PIL Image objects containing faces.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            faces = self.mtcnn(img)
            if faces is None:
                return []
                
            boxes = self.mtcnn.detect(img)[0]
            if boxes is None:
                return []

            extracted_faces = []
            for box in boxes:
                box = [int(b) for b in box]
                face = img.crop((box[0], box[1], box[2], box[3]))
                extracted_faces.append(face)
            
            return extracted_faces
        except Exception as e:
            print(f"Error extracting faces from image: {str(e)}")
            return []

    def extract_faces_from_video(self, video_path: str, n_frames: int = 20) -> List[Image.Image]:
        """
        Extract faces from video frames.
        Args:
            video_path: Path to video file
            n_frames: Number of frames to sample from video
        Returns list of PIL Image objects containing faces.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame indices to extract
            if n_frames > total_frames:
                n_frames = total_frames
            frame_indices = np.linspace(0, total_frames-1, n_frames, dtype=int)
            
            extracted_faces = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Detect faces
                faces = self.mtcnn(frame_pil)
                if faces is None:
                    continue
                    
                boxes = self.mtcnn.detect(frame_pil)[0]
                if boxes is None:
                    continue
                
                # Extract and save faces
                for box in boxes:
                    box = [int(b) for b in box]
                    face = frame_pil.crop((box[0], box[1], box[2], box[3]))
                    extracted_faces.append(face)
            
            cap.release()
            return extracted_faces
        except Exception as e:
            print(f"Error extracting faces from video: {str(e)}")
            return []

    def extract_faces_from_image_realtime(self, image: Image.Image) -> List[Image.Image]:
        """
        Extract faces from a single image in real-time.
        Optimized for speed in real-time applications.
        """
        try:
            faces = self.mtcnn(image)
            if faces is None:
                return []
                
            boxes = self.mtcnn.detect(image)[0]
            if boxes is None:
                return []

            extracted_faces = []
            for box in boxes:
                box = [int(b) for b in box]
                face = image.crop((box[0], box[1], box[2], box[3]))
                extracted_faces.append(face)
            
            return extracted_faces
        except Exception as e:
            print(f"Error extracting faces from image in real-time: {str(e)}")
            return []

    def get_face_location(self, image) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the location of faces in the image.
        Returns tuple of (top, right, bottom, left) coordinates.
        """
        try:
            boxes = self.mtcnn.detect(image)[0]
            if boxes is None:
                return None
            
            box = [int(b) for b in boxes[0]]  # Get first face
            return (box[1], box[2], box[3], box[0])  # Convert to (top, right, bottom, left)
        except Exception as e:
            print(f"Error getting face location: {str(e)}")
            return None

    def save_extracted_faces(self, faces: List[Image.Image], output_dir: str) -> List[str]:
        """
        Save extracted faces to directory.
        Returns list of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for i, face in enumerate(faces):
            try:
                output_path = os.path.join(output_dir, f'face_{i}.jpg')
                face.save(output_path)
                saved_paths.append(output_path)
            except Exception as e:
                print(f"Error saving face {i}: {str(e)}")
                continue
        
        return saved_paths
