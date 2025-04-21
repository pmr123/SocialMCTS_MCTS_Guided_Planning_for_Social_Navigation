import os
import torch
import requests
import PIL.Image
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional
import base64
import json
import logging
from pathlib import Path
from unsloth import FastLanguageModel
from transformers import AutoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VLMInterface:
    def __init__(self, model_id: str = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 log_dir: Optional[str] = None):
        """
        Initialize the Vision Language Model interface
        
        Args:
            model_id: Hugging Face model ID
            device: Device to run the model on
            log_dir: Directory to save logs
        """
        self.model_id = model_id
        self.device = device
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        
        # Check if model is already downloaded
        self._ensure_model_downloaded()
        
        logger.info(f"Loading VLM model {model_id} on {device}")
        
        try:
            # Load Llama 3.2 Vision model
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=4096,
                dtype=torch.bfloat16 if device == "cuda" else None,
                load_in_4bit=device == "cuda",
            )
            
            # Get the processor for handling images
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            logger.info("VLM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading VLM model: {str(e)}")
            raise
    
    def _ensure_model_downloaded(self):
        """Ensure the model is downloaded from Hugging Face"""
        from huggingface_hub import snapshot_download
        try:
            # Try to download the model if not already present
            snapshot_download(repo_id=self.model_id)
            logger.info(f"Model {self.model_id} already downloaded or downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise
    
    def _image_to_base64(self, image):
        """Convert RGB numpy array to base64 string"""
        # Convert numpy array to PIL Image
        if isinstance(image, PIL.Image.Image):
            pil_image = image
        else:
            pil_image = PIL.Image.fromarray(image)
            
        # Convert PIL Image to base64
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
            
    def generate_actions(self, 
                        image,
                        robot_pos: Tuple[float, float, float],
                        robot_orientation: float,
                        goal_pos: Tuple[float, float, float],
                        detection_results: Dict[str, Any],
                        k: int = 2,
                        log_output: bool = False) -> Dict[str, Any]:
        """
        Generate k actions using the VLM model
        
        Args:
            image: RGB image (numpy array or PIL Image)
            robot_pos: Current robot position (x, y, z)
            robot_orientation: Current robot orientation in degrees
            goal_pos: Goal position (x, y, z)
            detection_results: Detection results from human_goal_detector
            k: Number of actions to generate
            log_output: Whether to log the output
            
        Returns:
            Updated detection results with 'answer' field containing actions and scores
        """
        # Convert image to PIL if it's a numpy array
        if not isinstance(image, PIL.Image.Image):
            pil_image = PIL.Image.fromarray(image)
        else:
            pil_image = image
            
        # Create the prompt
        prompt = f"""
You are a social robot which aims to navigate a crowded environment similar to how a human will navigate the same crowded environment. You current have the following information:-
Current position = {robot_pos}
Current orientation = {robot_orientation}
Goal position = {goal_pos}
Data = {json.dumps(detection_results)}

Note that you use OpenGL's right-handed coordinate system. The ground plane lies along the X and Z axes, and the Y axis points up. The coordinate units are in meters by convention. When direction angles are specified, a positive angle corresponds to a counter-clockwise (leftward) rotation. Angles are in degrees for ease of hand-editing. By convention, angle zero points towards the positive X axis.

Using this information generate what action from the list should I take?

Answer format = append a "answer" field to the data json which should be populated with {k} values of the type {{number:score}}. Values for number:- 0 for turn left, 1 for turn right, and 2 for move forward. the score values should be a score from 0-1 (1 being highest) which signifies how close is that action to something which a human will do?
"""

        # Process the image and text
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode the output
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Log the output if requested
        if log_output and self.log_dir:
            self._log_vlm_output(pil_image, prompt, response, detection_results)
        
        try:
            # Parse the response to extract the JSON
            # Try to find a JSON-like structure in the response
            result = detection_results.copy()
            
            # Look for action scores in the format {number:score}
            import re
            action_scores = {}
            
            # Try to find JSON pattern in the response
            json_pattern = r'{.*}'
            json_match = re.search(json_pattern, response, re.DOTALL)
            
            if json_match:
                try:
                    json_str = json_match.group(0)
                    parsed_json = json.loads(json_str)
                    if isinstance(parsed_json, dict) and "answer" in parsed_json:
                        action_scores = parsed_json["answer"]
                except:
                    pass
            
            # If that didn't work, try to extract individual number:score pairs
            if not action_scores:
                pairs = re.findall(r'(\d+)\s*:\s*([01](\.\d+)?)', response)
                action_scores = {int(num): float(score) for num, score, _ in pairs[:k]}
            
            # Ensure we have enough actions
            if len(action_scores) < k:
                # Add default actions if needed
                default_actions = {0: 0.5, 1: 0.4, 2: 0.7}  # Default scores
                for i in range(3):
                    if len(action_scores) < k and i not in action_scores:
                        action_scores[i] = default_actions[i]
                    if len(action_scores) >= k:
                        break
            
            # Format and add to result
            result["answer"] = {str(key): value for key, value in list(action_scores.items())[:k]}
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing VLM response: {str(e)}")
            logger.error(f"Raw response: {response}")
            
            # Return default actions if parsing fails
            result = detection_results.copy()
            result["answer"] = {"0": 0.3, "1": 0.3, "2": 0.8}  # Default to moving forward
            return result
    
    def _log_vlm_output(self, image, prompt, response, detection_results):
        """Log VLM input/output for debugging"""
        timestamp = int(torch.cuda.current_stream().record_event().elapsed_time(torch.cuda.Event()))
        log_file = os.path.join(self.log_dir, f"vlm_log_{timestamp}.json")
        image_file = os.path.join(self.log_dir, f"vlm_image_{timestamp}.png")
        
        # Save the image
        if isinstance(image, PIL.Image.Image):
            image.save(image_file)
        
        # Save the prompt and response
        log_data = {
            "timestamp": timestamp,
            "prompt": prompt,
            "response": response,
            "detection_results": detection_results,
            "image_file": image_file
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"VLM log saved to {log_file}") 