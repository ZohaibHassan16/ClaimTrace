import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline, BitsAndBytesConfig
from sentence_transformers import CrossEncoder
from PIL import Image, ImageChops, ImageEnhance
import spacy
import io
import base64
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClaimTraceEngine")

class NewsVerifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing ClaimTrace on {self.device.upper()}...")

        
        self.fake_detector = pipeline(
            "image-classification", 
            model="umm-maybe/AI-image-detector", 
            device=0 if self.device == "cuda" else -1
        )

     
        self.nli_model = CrossEncoder(
            'cross-encoder/nli-distilroberta-base', 
            device=0 if self.device == "cuda" else -1
        )

      
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = None

        if self.device == "cuda":
            try:
                logger.info("Attempting 4-bit GPU quantization...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    quantization_config=bnb_config,
                    device_map="auto"
                )
                logger.info("Successfully loaded 4-bit Quantized Model on GPU.")
            except Exception as e:
                logger.error(f"GPU Load failed: {e}. Falling back to CPU.")
                self.load_cpu_model()
        else:
            self.load_cpu_model()

     
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        logger.info("All Systems Operational.")

    def load_cpu_model(self):
        """Fallback method for non-GPU environments (Laptops/CI)"""
        self.device = "cpu"
        logger.info("Loading standard model on CPU (optimized with bfloat16)...")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.bfloat16,  # Memory optimization for 16GB RAM
            low_cpu_mem_usage=True
        )
        logger.info("CPU Model Loaded.")

    def perform_ela(self, image: Image.Image):
        """Generates Error Level Analysis (ELA) Heatmap"""
        temp_buffer = io.BytesIO()
        image.convert("RGB").save(temp_buffer, format="JPEG", quality=90)
        temp_buffer.seek(0)

        ela_image = ImageChops.difference(image.convert("RGB"), Image.open(temp_buffer))
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        img_byte_arr = io.BytesIO()
        ela_image.save(img_byte_arr, format="PNG")
        return base64.b64encode(img_byte_arr.getvalue()).decode('ascii')

    def ask_vqa(self, image, question):
        prompt = f"Question: {question} Answer:"
        
       
        if self.device == "cpu":
             inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.bfloat16)
        else:
             inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
             
             dtype = self.model.dtype 
             inputs = {k: v.to(dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        
        output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return output

    def verify(self, image_bytes, text_caption):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        flags = []
        

        results = self.fake_detector(image)
        ai_prob = next((r['score'] * 100 for r in results if r['label'] == 'artificial'), 0.0)
        
        if ai_prob > 70:
            flags.append({
                "type": "AI Generated", 
                "severity": "Critical", 
                "details": f"{ai_prob:.1f}% probability of AI origin."
            })

    
        scene_desc = self.ask_vqa(image, "Describe the environment and main subject.")
        
        scores = self.nli_model.predict([(text_caption, scene_desc)])[0]
        label_mapping = ['contradiction', 'entailment', 'neutral']
        predicted_label = label_mapping[scores.argmax()]


        if predicted_label == 'contradiction' and scores[0] > 0.4:
            flags.append({
                "type": "Logical Contradiction",
                "severity": "High",
                "details": f"Caption claims '{text_caption}', but visual evidence suggests '{scene_desc}'."
            })

        return {
            "ai_probability": round(ai_prob, 2),
            "visual_summary": scene_desc,
            "inconsistencies": flags,
            "ela_heatmap": self.perform_ela(image)
        }