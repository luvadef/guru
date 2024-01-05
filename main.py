import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

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
etiquetas = [
    'depresion',
    'ansiedad',
    'estres',
    'trastorno_obsesivo_compulsivo',
    'trastorno_de_panico',
    'trastorno_de_personalidad',
    'trastorno_de_sueno',
    'trastorno_de_alimentacion',
    'trastorno_de_estado_de_animo',
    'trastorno_de_conducta',
    'trastorno_del_espectro_autista',
    'trastorno_disociativo',
    'trastorno_por_deficit_de_atencion',
    'trastorno_de_adaptacion',
    'trastorno_de_estres_post_traumatico',
    'trastorno_sexual',
    'trastorno_psicotico',
    'trastorno_somatomorfo',
    'trastorno_de_la_personalidad_paranoide',
    'trastorno_de_la_personalidad_esquizoide',
    'trastorno_de_la_personalidad_esquizotipico',
    'trastorno_de_la_personalidad_antisocial',
    'trastorno_de_la_personalidad_limite',
    'trastorno_de_la_personalidad_histriónico',
    'trastorno_de_la_personalidad_narcisista',
    'trastorno_de_la_personalidad_evitativo',
    'trastorno_de_la_personalidad_dependiente',
    'trastorno_de_la_personalidad_obsesivo_compulsivo',
    'trastorno_de_la_personalidad_no_especificado',
    'trastorno_de_la_conducta_alimentaria',
    'trastorno_de_sueño',
    'trastorno_del_juego',
    'adiccion',
    'abuso_de_sustancias',
    'problemas_de_relacion',
    'problemas_academicos',
    'problemas_laborales',
    'problemas_de_autoestima',
    'problemas_de_control_de_ira',
    'problemas_de_comunicacion',
    'duelo',
    'fobia_social',
    'problemas_de_orientacion_sexual',
    'trastorno_de_la_identidad_de_genero',
    'trastorno_de_la_conducta_sexual',
    'trastorno_del_desarrollo_intelectual',
    'trastorno_de_la_comunicacion',
    'trastorno_de_tics',
    'otros'
]

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
    'trastorno_obsesivo_compulsivo': [
        "¿Tienes pensamientos recurrentes no deseados que te causan ansiedad?",
        "¿Realizas comportamientos repetitivos para aliviar la ansiedad?",
        "¿Cuánto tiempo dedicas a las compulsiones diariamente?",
        "¿Te resulta difícil controlar estos pensamientos y comportamientos?",
    ],
    'trastorno_de_panico': [
        "¿Experimentas ataques de pánico repentinos e intensos?",
        "¿Hay situaciones específicas que desencadenan tus ataques de pánico?",
        "¿Cómo afectan los ataques de pánico a tu vida diaria?",
        "¿Has evitado lugares o situaciones debido al miedo a tener un ataque de pánico?",
        "¿Experimentas síntomas físicos como palpitaciones o dificultad para respirar durante los ataques?",
    ],
    'trastorno_de_personalidad': [
        "¿Has notado patrones persistentes en tu forma de percibir y relacionarte con los demás?",
        "¿Cómo afectan estos patrones a tus relaciones interpersonales?",
        "¿Te resulta difícil adaptarte a diferentes situaciones sociales?",
        "¿Experimentas cambios significativos en la imagen que tienes de ti mismo(a)?",
        "¿Has tenido dificultades para entender y compartir los sentimientos de los demás?",
    ],
    'trastorno_de_sueno': [
        "¿Experimentas dificultades para conciliar el sueño o permanecer dormido?",
        "¿Cuántas horas de sueño obtienes generalmente por noche?",
        "¿Te despiertas temprano y no puedes volver a dormir?",
        "¿Cómo afecta la falta de sueño a tu estado de ánimo y rendimiento diurno?",
    ],
    'trastorno_de_alimentacion': [
        "¿Tienes preocupaciones persistentes sobre tu peso o figura corporal?",
        "¿Has experimentado episodios de atracones o restricción extrema de alimentos?",
        "¿Cómo influye la alimentación en tu autoimagen y bienestar emocional?",
        "¿Has notado cambios significativos en tu peso en un corto período de tiempo?",
    ],
    'trastorno_de_estado_de_animo': [
        "¿Experimentas cambios bruscos en tu estado de ánimo, como períodos de tristeza intensa o euforia?",
        "¿Cuánto tiempo suelen durar estos cambios de humor?",
        "¿Cómo afectan estos cambios a tus actividades diarias y relaciones?",
        "¿Has notado algún patrón en los desencadenantes de tus cambios de humor?",
    ],
    'trastorno_de_conducta': [
        "¿Has participado en comportamientos problemáticos o ilegales?",
        "¿Cómo afectan estos comportamientos a tu vida diaria y relaciones?",
        "¿Has experimentado consecuencias negativas como resultado de tu comportamiento?",
        "¿Te sientes arrepentido(a) o culpable después de participar en estos comportamientos?",
    ],
    'trastorno_del_espectro_autista': [
        "¿Tienes dificultades para comprender las señales sociales?",
        "¿Prefieres rutinas y actividades específicas?",
        "¿Te resulta difícil mantener una conversación fluida?",
        "¿Tienes intereses o comportamientos repetitivos?",
        "¿Experimentas hipersensibilidad sensorial?",
    ],
    'trastorno_disociativo': [
        "¿Has experimentado períodos en los que te sientes desconectado(a) de tu entorno o de ti mismo(a)?",
        "¿Has experimentado pérdida de memoria durante estos períodos?",
        "¿Cómo afecta la disociación a tu vida diaria y relaciones?",
        "¿Puedes identificar desencadenantes específicos para estos episodios?",
    ],
    'trastorno_por_deficit_de_atencion': [
        "¿Experimentas dificultades para concentrarte en tareas específicas?",
        "¿Te resulta difícil permanecer sentado(a) o quieto(a) durante períodos prolongados?",
        "¿Tienes impulsividad en tus acciones y decisiones?",
        "¿Cómo afectan estos síntomas a tus responsabilidades diarias y relaciones?",
    ],
    'trastorno_de_adaptacion': [
        "¿Has experimentado recientemente eventos estresantes o cambios significativos en tu vida?",
        "¿Cómo has enfrentado estos cambios y cómo te han afectado emocionalmente?",
        "¿Te resulta difícil adaptarte a nuevas situaciones?",
        "¿Experimentas síntomas de ansiedad o depresión relacionados con estos cambios?",
    ],
    'trastorno_de_estres_post_traumatico': [
        "¿Has experimentado algún evento traumático en el pasado?",
        "¿Cómo te ha afectado emocional y físicamente este evento?",
        "¿Experimentas recuerdos intrusivos o pesadillas relacionadas con el trauma?",
        "¿Evitas situaciones o lugares que te recuerdan al evento traumático?",
    ],
    'trastorno_sexual': [
        "¿Experimentas dificultades o preocupaciones significativas relacionadas con la sexualidad?",
        "¿Cómo afectan estas preocupaciones a tu vida diaria y relaciones?",
        "¿Has notado cambios en tu deseo sexual o en la satisfacción sexual?",
        "¿Te sientes cómodo(a) hablando sobre tu sexualidad?",
    ],
    'trastorno_psicotico': [
        "¿Has experimentado cambios en la percepción de la realidad, como alucinaciones o delirios?",
        "¿Cómo afectan estas experiencias a tus pensamientos y comportamientos?",
        "¿Te resulta difícil distinguir entre lo que es real y lo que no lo es?",
        "¿Has notado cambios en tu capacidad para comunicarte o funcionar en la vida diaria?",
    ],
    'trastorno_somatomorfo': [
        "¿Experimentas síntomas físicos inexplicables o preocupaciones excesivas sobre tu salud?",
        "¿Cómo afectan estos síntomas a tu vida diaria y emocional?",
        "¿Has buscado atención médica repetidamente para síntomas que no tienen explicación médica?",
        "¿Cómo te sientes acerca de tu cuerpo y tu salud en general?",
    ],
    'trastorno_de_la_personalidad_paranoide': [
        "¿Tienes dificultades para confiar en los demás?",
        "¿Sientes que otros te están observando o conspirando en tu contra?",
        "¿Cómo afectan estos pensamientos y creencias a tus relaciones interpersonales?",
        "¿Te resulta difícil perdonar o dejar de lado las ofensas percibidas?",
    ],
    'trastorno_de_la_personalidad_esquizoide': [
        "¿Prefieres la soledad y tienes dificultades para establecer relaciones cercanas?",
        "¿Sientes que tus emociones son limitadas o difíciles de expresar?",
        "¿Cómo afecta tu estilo de vida reservado a tu bienestar emocional?",
        "¿Tienes intereses o actividades específicas en las que te sumerges profundamente?",
    ],
    'trastorno_de_la_personalidad_esquizotipico': [
        "¿Experimentas pensamientos o creencias inusuales?",
        "¿Sientes que tienes habilidades especiales o poderes no compartidos por los demás?",
        "¿Cómo afectan estas experiencias a tu vida diaria y relaciones?",
        "¿Te resulta difícil relacionarte con los demás de manera convencional?",
    ],
    'trastorno_de_la_personalidad_antisocial': [
        "¿Has tenido problemas recurrentes con la ley o violado los derechos de los demás?",
        "¿Te resulta difícil cumplir con las normas sociales y legales?",
        "¿Experimentas falta de empatía hacia los sentimientos de los demás?",
        "¿Cómo han afectado tus acciones a tu vida y relaciones?",
    ],
    'trastorno_de_la_personalidad_limite': [
        "¿Experimentas miedo intenso al abandono y cambios extremos en las relaciones?",
        "¿Tienes una imagen de ti mismo(a) que cambia rápidamente?",
        "¿Has participado en comportamientos impulsivos, como gastos excesivos o conductas sexuales arriesgadas?",
        "¿Cómo afecta la inestabilidad emocional a tus relaciones interpersonales?",
    ],
    'trastorno_de_la_personalidad_histriónico': [
        "¿Te sientes incómodo(a) cuando no eres el centro de atención?",
        "¿Buscas constantemente la aprobación de los demás?",
        "¿Cómo afecta tu necesidad de ser el centro de atención a tus relaciones?",
        "¿Experimentas cambios dramáticos en la expresión emocional?",
    ],
    'trastorno_de_la_personalidad_narcisista': [
        "¿Tienes una autoestima inflada y una necesidad excesiva de admiración?",
        "¿Te resulta difícil reconocer o entender las emociones de los demás?",
        "¿Cómo afecta tu búsqueda de éxito y reconocimiento a tus relaciones?",
        "¿Experimentas enojo o desprecio cuando no obtienes la atención que deseas?",
    ],
    'trastorno_de_la_personalidad_evitativo': [
        "¿Evitas actividades sociales o situaciones nuevas debido al miedo al rechazo o la crítica?",
        "¿Te sientes inadecuado(a) y temes ser avergonzado(a) o ridiculizado(a) por los demás?",
        "¿Cómo afecta el temor al rechazo a tu vida cotidiana y relaciones?",
        "¿Te resulta difícil iniciar relaciones debido a la ansiedad social?",
    ],
    'trastorno_de_la_personalidad_dependiente': [
        "¿Sientes la necesidad constante de ser cuidado(a) y protegido(a) por los demás?",
        "¿Te resulta difícil tomar decisiones cotidianas sin la aprobación de los demás?",
        "¿Cómo afecta la dependencia emocional a tus relaciones?",
        "¿Experimentas miedo intenso a ser abandonado(a) o dejado(a) solo(a)?",
    ],
    'trastorno_de_la_personalidad_obsesivo_compulsivo': [
        "¿Tienes la necesidad de tener el control sobre tu entorno y tus relaciones?",
        "¿Sientes la necesidad de seguir reglas y rutinas estrictas?",
        "¿Cómo afecta la obsesión por el control a tus relaciones interpersonales?",
        "¿Te resulta difícil delegar tareas a otros?",
    ],
    'trastorno_de_la_personalidad_no_especificado': [
        "¿Has notado patrones de pensamiento o comportamiento que no encajan claramente en ninguna categoría específica?",
        "¿Cómo afectan estas experiencias a tu vida diaria y relaciones?",
        "¿Te resulta difícil compartir tus pensamientos o creencias con los demás?",
    ],
    'trastorno_de_la_personalidad_antisocial': [
        "¿Has mostrado patrones de comportamiento irrespetuoso hacia los derechos de los demás?",
        "¿Has tenido problemas legales relacionados con tu comportamiento?",
        "¿Cómo afectan tus acciones a las relaciones y la vida diaria de los demás?",
        "¿Te sientes arrepentido(a) o culpable después de infringir normas sociales?",
    ],
    'trastorno_de_la_personalidad_limite': [
        "¿Experimentas miedo al abandono y cambios extremos en las relaciones?",
        "¿Has tenido episodios recurrentes de impulsividad, como gastos excesivos o conducta sexual riesgosa?",
        "¿Cómo afectan tus emociones intensas y cambios de humor a tus relaciones interpersonales?",
        "¿Te sientes inestable en tu identidad y autoimagen?",
    ],
    'trastorno_de_la_personalidad_histriónico': [
        "¿Buscas constantemente la atención y te sientes incómodo(a) cuando no la recibes?",
        "¿Tienes una necesidad excesiva de aprobación o agradar a los demás?",
        "¿Cómo afecta tu comportamiento a tus relaciones personales y laborales?",
        "¿Experimentas cambios rápidos en tus emociones y expresiones?",
    ],
    'trastorno_de_la_personalidad_narcisista': [
        "¿Tienes una percepción inflada de tu propia importancia y habilidades?",
        "¿Te preocupas mucho por fantasías de éxito ilimitado y poder?",
        "¿Cómo afecta tu falta de empatía y necesidad de admiración a tus relaciones?",
        "¿Te sientes frustrado(a) cuando los demás no reconocen tu superioridad?",
    ],
    'trastorno_de_la_personalidad_evitativo': [
        "¿Sientes una inhibición social extrema y temor al rechazo?",
        "¿Te resulta difícil iniciar relaciones personales debido al miedo al ridículo o desprecio?",
        "¿Cómo afecta tu evitación social a tu vida laboral y personal?",
        "¿Te gustaría participar más en actividades sociales pero te resulta difícil hacerlo?",
    ],
    'trastorno_de_la_personalidad_dependiente': [
        "¿Sientes una necesidad excesiva de ser cuidado(a) y temor a la separación?",
        "¿Te resulta difícil tomar decisiones sin la ayuda y consejo de los demás?",
        "¿Cómo afecta tu necesidad de aprobación y apoyo a tus relaciones?",
        "¿Te has sentido atrapado(a) en relaciones insatisfactorias por miedo a quedarte solo(a)?",
    ],
    'trastorno_de_la_personalidad_obsesivo_compulsivo': [
        "¿Tienes patrones de pensamiento y comportamiento perfeccionistas y rígidos?",
        "¿Te resulta difícil delegar tareas a otras personas?",
        "¿Cómo afecta tu necesidad de control a tus relaciones y actividades diarias?",
        "¿Sientes una preocupación excesiva por el orden y los detalles?",
    ],
    'trastorno_de_la_personalidad_no_especificado': [
        "¿Experimentas patrones de comportamiento que no se ajustan a otras categorías específicas?",
        "¿Te has sentido incomprendido(a) o diferente de los demás?",
        "¿Cómo afecta este patrón de comportamiento a tu vida diaria y relaciones?",
        "¿Has buscado ayuda profesional para comprender y manejar estos patrones?",
    ],
    'trastorno_de_la_conducta_alimentaria': [
        "¿Tienes preocupaciones persistentes sobre tu peso o figura corporal?",
        "¿Has experimentado episodios de atracones o restricción extrema de alimentos?",
        "¿Cómo influye la alimentación en tu autoimagen y bienestar emocional?",
        "¿Has notado cambios significativos en tu peso en un corto período de tiempo?",
    ],
    'trastorno_de_sueño': [
        "¿Experimentas dificultades para conciliar el sueño o permanecer dormido?",
        "¿Cuántas horas de sueño obtienes generalmente por noche?",
        "¿Te despiertas temprano y no puedes volver a dormir?",
        "¿Cómo afecta la falta de sueño a tu estado de ánimo y rendimiento diurno?",
    ],
    'trastorno_del_juego': [
        "¿Has experimentado dificultades para controlar o detener tus actividades de juego?",
        "¿Has apostado más dinero del que puedes permitirte perder?",
        "¿Cómo afecta el juego a tus relaciones personales y responsabilidades diarias?",
        "¿Has intentado sin éxito reducir o controlar tus hábitos de juego?",
    ],
    'adiccion': [
        "¿Has sentido una necesidad persistente de consumir una sustancia específica?",
        "¿Has experimentado problemas laborales, legales o de relación debido a tu consumo de sustancias?",
        "¿Cómo afecta la sustancia a tu vida diaria y bienestar emocional?",
        "¿Has intentado reducir o controlar tu consumo sin éxito?",
    ],
    'abuso_de_sustancias': [
        "¿Has utilizado sustancias de manera recurrente a pesar de los problemas que causan?",
        "¿Has experimentado síntomas de abstinencia al intentar reducir o dejar de consumir sustancias?",
        "¿Cómo afecta el abuso de sustancias a tu salud física y mental?",
        "¿Has buscado ayuda para superar tu abuso de sustancias?",
    ],
    'problemas_de_relacion': [
        "¿Experimentas conflictos recurrentes o dificultades en tus relaciones personales?",
        "¿Cómo afectan estos problemas a tu bienestar emocional?",
        "¿Has notado patrones específicos en tus relaciones que te preocupan?",
        "¿Te resulta difícil establecer o mantener relaciones saludables?",
    ],
    'problemas_academicos': [
        "¿Has experimentado dificultades persistentes en tu rendimiento académico?",
        "¿Cómo afecta el rendimiento académico a tu autoestima y bienestar emocional?",
        "¿Te sientes abrumado(a) por las expectativas académicas?",
        "¿Has buscado apoyo para superar tus dificultades académicas?",
    ],
    'problemas_laborales': [
        "¿Experimentas conflictos o dificultades en tu entorno laboral?",
        "¿Cómo afecta tu trabajo a tu bienestar emocional y físico?",
        "¿Te sientes satisfecho(a) con tu carrera y logros laborales?",
        "¿Has considerado cambiar de trabajo debido a problemas laborales?",
    ],
    'problemas_de_autoestima': [
        "¿Tienes una percepción negativa de ti mismo(a) de manera persistente?",
        "¿Cómo afecta la baja autoestima a tu vida diaria y relaciones?",
        "¿Has experimentado críticas constantes hacia ti mismo(a)?",
        "¿Qué aspectos específicos de tu apariencia o personalidad te generan más conflicto?",
    ],
    'problemas_de_control_de_ira': [
        "¿Experimentas dificultades para controlar tu enojo o irritación?",
        "¿Has expresado tu enojo de manera destructiva hacia ti mismo(a) o los demás?",
        "¿Cómo afecta la ira a tus relaciones personales y responsabilidades diarias?",
        "¿Has buscado estrategias para manejar constructivamente la ira?",
    ],
    'problemas_de_comunicacion': [
        "¿Experimentas malentendidos frecuentes o dificultades para expresar tus pensamientos?",
        "¿Cómo afectan las dificultades de comunicación a tus relaciones personales y profesionales?",
        "¿Te resulta difícil expresar tus necesidades y deseos de manera clara?",
        "¿Has considerado buscar ayuda para mejorar tus habilidades de comunicación?",
    ],
    'duelo': [
        "¿Has perdido a alguien cercano recientemente?",
        "¿Cómo has estado manejando el duelo y la pérdida?",
        "¿Experimentas sentimientos de tristeza, enojo o vacío relacionados con la pérdida?",
        "¿Has buscado apoyo para procesar el duelo?",
    ],
    'fobia_social': [
        "¿Experimentas miedo o ansiedad intensa en situaciones sociales?",
        "¿Te resulta difícil interactuar con otras personas o hablar en público?",
        "¿Cómo afecta la fobia social a tus relaciones y actividades sociales?",
        "¿Has evitado situaciones sociales debido al miedo al juicio o la vergüenza?",
    ],
    'problemas_de_orientacion_sexual': [
        "¿Has experimentado dificultades o conflictos relacionados con tu orientación sexual?",
        "¿Cómo afecta tu orientación sexual a tus relaciones y bienestar emocional?",
        "¿Te has sentido discriminado(a) o incomprendido(a) debido a tu orientación sexual?",
        "¿Has buscado apoyo para explorar y aceptar tu orientación sexual?",
    ],
    'trastorno_de_la_identidad_de_genero': [
        "¿Experimentas conflictos o incongruencia entre tu identidad de género y el sexo asignado al nacer?",
        "¿Cómo afecta la identidad de género a tu vida diaria y relaciones?",
        "¿Has considerado la posibilidad de realizar cambios en la expresión de género o la transición?",
        "¿Has buscado apoyo para comprender y aceptar tu identidad de género?",
    ],
    'trastorno_de_la_conducta_sexual': [
        "¿Has participado en comportamientos sexuales problemáticos o ilegales?",
        "¿Cómo afectan estos comportamientos a tus relaciones y bienestar emocional?",
        "¿Has experimentado consecuencias negativas como resultado de tu conducta sexual?",
        "¿Te sientes arrepentido(a) o culpable después de participar en estos comportamientos?",
    ],
    'trastorno_del_desarrollo_intelectual': [
        "¿Experimentas dificultades significativas en el desarrollo de habilidades intelectuales y adaptativas?",
        "¿Cómo afecta el trastorno del desarrollo intelectual a tu vida diaria y relaciones?",
        "¿Has buscado apoyo para superar los desafíos asociados con el desarrollo intelectual?",
        "¿Te sientes satisfecho(a) con tu capacidad para funcionar de manera independiente?",
    ],
    'trastorno_de_la_comunicacion': [
        "¿Experimentas dificultades persistentes en la comunicación verbal y no verbal?",
        "¿Cómo afecta el trastorno de la comunicación a tus relaciones personales y profesionales?",
        "¿Te resulta difícil expresar tus pensamientos de manera clara o comprender las expresiones de los demás?",
        "¿Has buscado estrategias para mejorar tus habilidades de comunicación?",
    ],
    'trastorno_de_tics': [
        "¿Experimentas movimientos o vocalizaciones repetitivas e involuntarias?",
        "¿Cómo afectan los tics a tu vida diaria y relaciones?",
        "¿Te resulta difícil controlar o suprimir los tics durante situaciones sociales?",
    ],
    'otros': [
        "¿Puedes proporcionar más detalles sobre lo que estás experimentando?",
        "¿Hay algún evento reciente que creas que pueda estar relacionado?",
        "¿Cómo crees que puedo ayudarte mejor?",
    ],
}

preguntas = [pregunta.lower() for pregunta in preguntas]
respuestas = [respuesta.lower() for respuesta in respuestas]
for lista_preguntas in preguntas_adicionales.values():
    lista_preguntas = [pregunta.lower() for pregunta in lista_preguntas]

# Tokenización
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preguntas)
total_palabras = len(tokenizer.word_index) + 1

# Convertir las preguntas y respuestas a secuencias numéricas
secuencias_preguntas = pad_sequences(tokenizer.texts_to_sequences(preguntas))
secuencias_respuestas = pad_sequences(tokenizer.texts_to_sequences(respuestas))

# Modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=total_palabras, output_dim=5, input_length=secuencias_preguntas.shape[1]),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=len(etiquetas), activation='softmax')
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Convertir etiquetas reales a one-hot encoding
etiquetas_one_hot = tf.keras.utils.to_categorical(np.array([etiquetas.index(label) for label in etiquetas]), num_classes=len(etiquetas))

print("Comenzando entrenamiento...")
#historial = modelo.fit(secuencias_preguntas, etiquetas_one_hot, epochs=2000, verbose=False)
historial = modelo.fit(secuencias_preguntas, etiquetas_one_hot, epochs=1000, validation_split=0.2, verbose=False)
print("Modelo entrenado!")

# Mostrar gráfico de pérdida y precisión
plt.figure(figsize=(12, 4))

# Gráfico de pérdida
plt.subplot(1, 2, 1)
plt.plot(historial.history['loss'], label='Entrenamiento')
plt.plot(historial.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Gráfico de precisión
plt.subplot(1, 2, 2)
plt.plot(historial.history['accuracy'], label='Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Mostrar gráficos
plt.tight_layout()
plt.show()

umbral_certezas = 0.7 # Ajusta según sea necesario

def hacer_pregunta(pregunta):
    respuesta_usuario = input(f"Sicólogo: {pregunta}\nTú: ")
    nueva_secuencia = pad_sequences(tokenizer.texts_to_sequences([respuesta_usuario]), maxlen=secuencias_preguntas.shape[1])
    resultado = modelo.predict(nueva_secuencia)
    indice_clase_predicha = np.argmax(resultado)
    certeza = resultado[0][indice_clase_predicha]

    if certeza is None or certeza < umbral_certezas:
        print("Sicólogo: No puedo hacer una predicción con suficiente certeza en este momento. Por favor, considera consultar a un profesional de la salud.")
        return None, None

    etiqueta_predicha = etiquetas[indice_clase_predicha]
    return etiqueta_predicha, certeza

# Iniciar interacción
pregunta_inicial = "Hola, ¿cómo te sientes hoy?"
patologia_predicha, certeza_prediccion = hacer_pregunta(pregunta_inicial)

certeza_prediccion = 70  # Inicialización con un valor base

# Después de cada respuesta
certeza_respuesta = 0.5  # Certidumbre basada en la respuesta actual
certeza_prediccion += certeza_respuesta

# Manejo de preguntas adicionales agotadas
while certeza_prediccion >= umbral_certezas and (patologia_predicha in preguntas_adicionales) and preguntas_adicionales[patologia_predicha]:
    pregunta_adicional = preguntas_adicionales[patologia_predicha].pop(0)
    print(f"Indicio: {patologia_predicha}, con una certeza de: {certeza_prediccion}")
    print(f"Sicólogo: {pregunta_adicional}")
    patologia_predicha, certeza_prediccion = hacer_pregunta("")

if certeza_prediccion >= umbral_certezas:
    print(f"Sicólogo: Entiendo. Basado en nuestra conversación, parece que hay indicios de {patologia_predicha}.")