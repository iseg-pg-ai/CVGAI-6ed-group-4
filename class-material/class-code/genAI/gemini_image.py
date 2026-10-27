#pip install -U google-genai Pillow
from google import genai
import PIL.Image
from dotenv import load_dotenv
from pathlib import Path
import os.path


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
llmapikey = os.getenv("gemini_key")  
llmapikey = ""

client = genai.Client(api_key=llmapikey)

img = PIL.Image.open('parque.jpg')

response = client.models.generate_content(
	model='gemini-3.5-flash', 
	contents=["quantos carros estão nesta imagem?", img]
)

print(response.text)
