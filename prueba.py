import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la API key
api_key = os.getenv('API_KEY')

# Ahora puedes usar la API key en tu aplicación
print(api_key)  # Solo para verificar; quita esto en producción