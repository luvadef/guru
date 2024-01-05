import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Preguntas y respuestas
preguntas = [
    "¿Qué te trae por aquí?",
    "¿Es tu primera vez acudiendo al psicólogo?",
    "¿Tienes alguna idea concreta de lo que te gustaría hablar o lo que te molesta?",
    "¿Cómo te hace sentir?",
    "¿Cómo te describirías y cómo describirías tu estado de ánimo?",
    "¿Qué te hace sentir mejor cuando te sientes ansioso o abrumado?",
    "¿Qué esperas de este proceso? ¿Qué te gustaría lograr?",
]

respuestas = [
    "Contar con apoyo emocional.",
    "Sí, es mi primera vez.",
    "No estoy seguro, solo sé que algo no está bien.",
    "Me hace sentir ansioso y abrumado.",
    "Me veo como alguien estresado, y mi estado de ánimo es bajo.",
    "A veces dar un paseo o practicar la meditación.",
    "Espero entenderme mejor y encontrar formas de manejar el estrés.",
]

# Etiquetas de patologías (ajusta según tus necesidades)
etiquetas = ['depresion', 'ansiedad', 'estres', 'otra']

# Preguntas adicionales del sicólogo para cada etiqueta
preguntas_adicionales = {
    'depresion': [
        "¿Has experimentado cambios en tu apetito o en tu patrón de sueño recientemente?",
        "¿Cómo ha afectado tu interés en actividades que antes disfrutabas?",
        "¿Hay algún evento reciente que crees que pueda estar relacionado con tu estado de ánimo?",
    ],
    'ansiedad': [
        "¿Sientes una sensación constante de nerviosismo o tensión?",
        "¿Has experimentado ataques de pánico?",
        "¿Cómo afecta la ansiedad a tus relaciones personales?",
    ],
    'estres': [
        "¿Cuáles son las principales fuentes de estrés en tu vida actualmente?",
        "¿Cómo afecta el estrés a tus relaciones personales?",
        "¿Hay alguna actividad que te ayude a aliviar el estrés?",
    ],
    'otra': [
        "¿Puedes proporcionar más detalles sobre lo que estás experimentando?",
        "¿Hay algún evento reciente que creas que pueda estar relacionado?",
        "¿Cómo crees que puedo ayudarte mejor?",
    ],
}

# Tokenización
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preguntas)
total_palabras = len(tokenizer.word_index) + 1

# Convertir las preguntas y respuestas a secuencias numéricas
secuencias_preguntas = pad_sequences(tokenizer.texts_to_sequences(preguntas))
secuencias_respuestas = pad_sequences(tokenizer.texts_to_sequences(respuestas))

# Modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=total_palabras, output_dim=3, input_length=secuencias_preguntas.shape[1]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=3, activation='relu'),
    tf.keras.layers.Dense(units=len(etiquetas), activation='softmax')  # Salida softmax para clasificación multiclase
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='categorical_crossentropy',  # Categorical crossentropy para clasificación multiclase
    metrics=['accuracy']
)

# Convertir etiquetas a one-hot encoding
etiquetas_one_hot = tf.keras.utils.to_categorical(np.array([0]*len(respuestas)), num_classes=len(etiquetas))

print("Comenzando entrenamiento...")
historial = modelo.fit(secuencias_preguntas, etiquetas_one_hot, epochs=1000, verbose=False)
print("Modelo entrenado!")

def hacer_pregunta(pregunta):
    respuesta_usuario = input(f"Sicólogo: {pregunta}\nTú: ")
    nueva_secuencia = pad_sequences(tokenizer.texts_to_sequences([respuesta_usuario]), maxlen=secuencias_preguntas.shape[1])
    resultado = modelo.predict(nueva_secuencia)
    indice_clase_predicha = np.argmax(resultado)
    etiqueta_predicha = etiquetas[indice_clase_predicha]
    certeza = resultado[0][indice_clase_predicha]

    return etiqueta_predicha, certeza

# Iniciar interacción
pregunta_inicial = "Hola, ¿cómo te sientes hoy?"
patologia_predicha, certeza_prediccion = hacer_pregunta(pregunta_inicial)

umbral_certezas = 80.0  # Ajusta según sea necesario

while certeza_prediccion < umbral_certezas and preguntas_adicionales.get(patologia_predicha):
    pregunta_adicional = preguntas_adicionales[patologia_predicha].pop(0)
    print(f"Sicólogo: Parece que hay indicios de {patologia_predicha}. {pregunta_adicional}")
    patologia_predicha, certeza_prediccion = hacer_pregunta("")

print(f"Sicólogo: Entiendo. Basado en nuestra conversación, parece que hay indicios de {patologia_predicha}.")
