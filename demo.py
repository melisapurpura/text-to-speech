import os
from google.cloud import texttospeech

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\xolo\Documents\1_text-to-speech\text-to-speech.json'

client = texttospeech.TextToSpeechClient()

text_block = '''

Este gráfico es un ejemplo del problema que un modelo supervisado podría intentar resolver.
Por ejemplo, digamos que eres el dueño de un restaurante.
Tienes datos históricos de la cantidad de la factura y cuánto dejaron de propina diferentes personas según el tipo de pedido y si fue recogido o entregado.
En el aprendizaje supervisado, el modelo aprende de ejemplos pasados para predecir valores futuros, en este caso, propinas.
Entonces, aquí el modelo usa la cantidad total de la factura para predecir la cantidad de propina futura según si un pedido fue recogido o entregado.
Este es un ejemplo del problema que un modelo no supervisado podría intentar resolver.
Entonces, aquí deseas observar la antigüedad y los ingresos y luego agrupar o clasificar a los empleados para ver si alguien está en el camino rápido.
Los problemas no supervisados son todo sobre el descubrimiento, sobre observar los datos sin procesar y ver si naturalmente se agrupan.
Profundicemos un poco y mostremos esto gráficamente, ya que entender estos conceptos es la base para tu comprensión de la IA generativa.
En el aprendizaje supervisado, los valores de datos de prueba o x se introducen en el modelo.
El modelo produce una predicción y compara esa predicción con los datos de entrenamiento utilizados para entrenar el modelo.
Si los valores de datos de prueba predichos y los valores de datos de entrenamiento reales están muy separados, eso se llama error.
Y el modelo intenta reducir este error hasta que los valores predichos y reales estén más cerca.
Esto es un problema clásico de optimización.
Ahora que hemos explorado la diferencia entre inteligencia artificial y aprendizaje automático, y aprendizaje supervisado y no supervisado, exploremos brevemente dónde encaja el aprendizaje profundo como un subconjunto de los métodos de aprendizaje automático.
Mientras que el aprendizaje automático es un campo amplio que abarca muchas técnicas diferentes, el aprendizaje profundo es un tipo de aprendizaje automático que utiliza redes neuronales artificiales, lo que les permite procesar patrones más complejos que el aprendizaje automático.
Las redes neuronales artificiales están inspiradas en el cerebro humano.


'''

synthesis_input =  texttospeech.SynthesisInput(text=text_block)

voice = texttospeech.VoiceSelectionParams(
    language_code="es-US",
    name= 'es-US-Neural2-B' 
)

audio_config = texttospeech.AudioConfig(
    #cambiamos de MP3 a LINEAR16 y suena mucho mejor, ya sin eco.
    audio_encoding=texttospeech.AudioEncoding.LINEAR16
    #speaking_rate=0.9
    #volume_gain_db = 12
    #pitch=1
)



response = client.synthesize_speech(
    input=synthesis_input, 
    voice=voice, 
    audio_config=audio_config
)

with open("Clase1_genAI_6.mp3", "wb") as out:
          out.write(response.audio_content)
          print('Audio done')
