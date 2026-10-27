import os
from openai import OpenAI
import base64
import mimetypes
import cv2
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
llmapikey = os.getenv("APIKEY")  

client = OpenAI(api_key=llmapikey)

def image_to_base64(image_path):
	mime_type, _ = mimetypes.guess_type(image_path)
	if not mime_type or not mime_type.startswith('image'):
		raise ValueError("imagem invalida")
	with open(image_path, 'rb') as image_file:
		encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
	image_base64 = f"data:{mime_type};base64,{encoded_string}"
	return image_base64

def llm(img,prompt):
	base64_string = image_to_base64(img)
	#print(base64_string)
	
	prompt_text = (prompt)
	resp = client.responses.create(
		model="gpt-6.1-sol",
		input=[{"role": "user","content": [{"type": "input_text", "text": prompt_text},{"type": "input_image", "image_url": base64_string},],}],
		max_output_tokens=500,
	)
	return(resp.output_text)
	

output = llm("parque.jpg","descreve com pormenor a imagem")

print(output)