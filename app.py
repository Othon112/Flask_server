from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})

faq_data = [


    {"question": "Hola!", "answer": "¡Hola! ¿Te puedo ayudar en algo?"},
    {"question": "Si", "answer": "¡Claro!"},
    {"question": "No", "answer": "¡Gracias! Si tienes alguna otra pregunta hazmelo saber"},

    {"question": "No te pregunté sobre eso", "answer": "¡Lo siento! Mi algortimo debió haberlo confundido con otra pregunta, ¿Podrías formular la pregunta de otra manera?"},


    {"question": "¿Cómo estás?", "answer": "Estoy bien, gracias."},
    {"question": "¿Estás bien?", "answer": "Estoy bien, gracias."},
    {"question": "¿Cómo te sientes actualmente?", "answer": "Estoy bien, gracias."},
    {"question": "¿Te encuentras en buen estado?", "answer": "Estoy bien, gracias."},
    {"question": "¿Cómo está tu día?", "answer": "Estoy bien, gracias."},


    {"question": "¿En qué me puede ayudar?", "answer": "Puedo responder preguntas frecuentes."},
    {"question": "¿Cuál es tu función?", "answer": "Puedo responder preguntas frecuentes."},
    {"question": "¿Tienes alguna función en específico?", "answer": "Puedo responder preguntas frecuentes."},
    {"question": "¿Puedes proporcionarme información útil?", "answer": "Puedo responder preguntas frecuentes."},
    {"question": "¿En qué áreas eres competente?", "answer": "Puedo responder preguntas frecuentes."},
    {"question": "¿Para qué fuiste creado?", "answer": "Puedo responder preguntas frecuentes."},


    {"question": "¿Cómo se llama su producto?", "answer": "Envase Compostable Biointellectus para Cosmética Natural"},
    {"question": "¿Cuál es el nombre de su producto?", "answer": "Envase Compostable Biointellectus para Cosmética Natural"},
    {"question": "¿Cómo identifican su producto?", "answer": "Envase Compostable Biointellectus para Cosmética Natural"},
    {"question": "¿Bajo qué nombre se comercializa su producto?", "answer": "Envase Compostable Biointellectus para Cosmética Natural"},
    {"question": "¿Puede darme el nombre de su producto?", "answer": "Envase Compostable Biointellectus para Cosmética Natural"},


    {"question": "¿Me puedes describir el producto?", "answer": "El envase compostable de Biointellectus es un contenedor eco-innovador diseñado para la industria cosmética y de belleza. Creado con el objetivo de combinar funcionalidad, estética y responsabilidad medioambiental, este envase está fabricado con bioplástico derivado del almidón, que ofrece una alternativa sostenible y biodegradable a los envases tradicionales."},
    {"question": "¿Cómo se caracteriza su producto?", "answer": "El envase compostable de Biointellectus es un contenedor eco-innovador diseñado para la industria cosmética y de belleza. Creado con el objetivo de combinar funcionalidad, estética y responsabilidad medioambiental, este envase está fabricado con bioplástico derivado del almidón, que ofrece una alternativa sostenible y biodegradable a los envases tradicionales."},
    {"question": "¿Puede proporcionar una descripción detallada de su producto?", "answer": "El envase compostable de Biointellectus es un contenedor eco-innovador diseñado para la industria cosmética y de belleza. Creado con el objetivo de combinar funcionalidad, estética y responsabilidad medioambiental, este envase está fabricado con bioplástico derivado del almidón, que ofrece una alternativa sostenible y biodegradable a los envases tradicionales."},
    {"question": "¿Qué características distinguen a su producto?", "answer": "El envase compostable de Biointellectus es un contenedor eco-innovador diseñado para la industria cosmética y de belleza. Creado con el objetivo de combinar funcionalidad, estética y responsabilidad medioambiental, este envase está fabricado con bioplástico derivado del almidón, que ofrece una alternativa sostenible y biodegradable a los envases tradicionales."},
    {"question": "¿Podría describir en detalle el producto que ofrece?", "answer": "El envase compostable de Biointellectus es un contenedor eco-innovador diseñado para la industria cosmética y de belleza. Creado con el objetivo de combinar funcionalidad, estética y responsabilidad medioambiental, este envase está fabricado con bioplástico derivado del almidón, que ofrece una alternativa sostenible y biodegradable a los envases tradicionales."},

    
    {"question": "¿De qué material está hecho?", "answer": "Bioplástico a base de almidón, proveniente de fuentes renovables y sostenibles. Compuesto totalmente libre de plásticos y 100% compostable, fabricado para descomponerse en un corto periodo de tiempo."},
    {"question": "¿Cuáles son los materiales utilizados en la fabricación?", "answer": "Bioplástico a base de almidón, proveniente de fuentes renovables y sostenibles. Compuesto totalmente libre de plásticos y 100% compostable, fabricado para descomponerse en un corto periodo de tiempo."},
    {"question": "¿De qué compuesto está hecho su producto?", "answer": "Bioplástico a base de almidón, proveniente de fuentes renovables y sostenibles. Compuesto totalmente libre de plásticos y 100% compostable, fabricado para descomponerse en un corto periodo de tiempo."},
    {"question": "¿Podría proporcionar información sobre los componentes?", "answer": "Bioplástico a base de almidón, proveniente de fuentes renovables y sostenibles. Compuesto totalmente libre de plásticos y 100% compostable, fabricado para descomponerse en un corto periodo de tiempo."},
    {"question": "¿Qué materiales forman parte de su producto?", "answer": "Bioplástico a base de almidón, proveniente de fuentes renovables y sostenibles. Compuesto totalmente libre de plásticos y 100% compostable, fabricado para descomponerse en un corto periodo de tiempo."},


    {"question": "¿Cuál es su máxima capacidad?", "answer": "Diseñado para contener hasta 10 gramos de cosméticos en polvo, sólidos y semi sólidos a base de aceites o grasas vegetales."},
     {"question": "¿Cuánto puede contener como máximo?", "answer": "Diseñado para contener hasta 10 gramos de cosméticos en polvo, sólidos y semi sólidos a base de aceites o grasas vegetales."},
    {"question": "¿Cuál es la capacidad máxima del envase?", "answer": "Diseñado para contener hasta 10 gramos de cosméticos en polvo, sólidos y semi sólidos a base de aceites o grasas vegetales."},
    {"question": "¿Hasta qué punto puede llenarse?", "answer": "Diseñado para contener hasta 10 gramos de cosméticos en polvo, sólidos y semi sólidos a base de aceites o grasas vegetales."},
    {"question": "¿Cuál es la máxima cantidad que puede almacenar?", "answer": "Diseñado para contener hasta 10 gramos de cosméticos en polvo, sólidos y semi sólidos a base de aceites o grasas vegetales."},


    {"question": "¿Qué durabilidad tiene?", "answer": "Resistente al manejo diario, ofreciendo protección y seguridad al producto contenido."},
    {"question": "¿Cuánto tiempo puede durar el envase?", "answer": "Resistente al manejo diario, ofreciendo protección y seguridad al producto contenido."},
    {"question": "¿Qué tan durable es el producto?", "answer": "Resistente al manejo diario, ofreciendo protección y seguridad al producto contenido."},
    {"question": "¿Cuál es la vida útil del envase?", "answer": "Resistente al manejo diario, ofreciendo protección y seguridad al producto contenido."},
    {"question": "¿Tiene alguna indicación sobre su durabilidad?", "answer": "Resistente al manejo diario, ofreciendo protección y seguridad al producto contenido."},


    {"question": "¿Su producto es compostable?", "answer": "Actualmente nuestro envase a demostrado una composabilidad completa en menos de 5 semanas, creando un precedente de sustentabilidad en esta industria. Hoy en dia estamos en el proceso de ser certificado para descomponerse en condiciones de compostaje industrial y compostaje casero, dejando un mínimo impacto ambiental."},
    {"question": "¿Se puede descomponer su producto?", "answer": "Actualmente nuestro envase a demostrado una composabilidad completa en menos de 5 semanas, creando un precedente de sustentabilidad en esta industria. Hoy en dia estamos en el proceso de ser certificado para descomponerse en condiciones de compostaje industrial y compostaje casero, dejando un mínimo impacto ambiental."},
    {"question": "¿Es su envase amigable con el medio ambiente?", "answer": "Actualmente nuestro envase a demostrado una composabilidad completa en menos de 5 semanas, creando un precedente de sustentabilidad en esta industria. Hoy en dia estamos en el proceso de ser certificado para descomponerse en condiciones de compostaje industrial y compostaje casero, dejando un mínimo impacto ambiental."},
    {"question": "¿Su producto es ecológico?", "answer": "Actualmente nuestro envase a demostrado una composabilidad completa en menos de 5 semanas, creando un precedente de sustentabilidad en esta industria. Hoy en dia estamos en el proceso de ser certificado para descomponerse en condiciones de compostaje industrial y compostaje casero, dejando un mínimo impacto ambiental."},
    {"question": "¿Tiene alguna certificación ecológica?", "answer": "Actualmente nuestro envase a demostrado una composabilidad completa en menos de 5 semanas, creando un precedente de sustentabilidad en esta industria. Hoy en dia estamos en el proceso de ser certificado para descomponerse en condiciones de compostaje industrial y compostaje casero, dejando un mínimo impacto ambiental."},


    {"question": "¿Es su producto es ecológico?", "answer": "Reduce la huella de carbono y el impacto en los vertederos en comparación con los envases plásticos convencionales. Ofrece a las marcas de cosmética natural la posibilidad de cumplir con sus promesas de sostenibilidad genuina"},
    {"question": "¿Cuál es el impacto ecológico de su producto?", "answer": "Reduce la huella de carbono y el impacto en los vertederos en comparación con los envases plásticos convencionales. Ofrece a las marcas de cosmética natural la posibilidad de cumplir con sus promesas de sostenibilidad genuina"},
    {"question": "¿Cómo contribuye su producto a la sostenibilidad ambiental?", "answer": "Reduce la huella de carbono y el impacto en los vertederos en comparación con los envases plásticos convencionales. Ofrece a las marcas de cosmética natural la posibilidad de cumplir con sus promesas de sostenibilidad genuina"},
    {"question": "¿Tiene beneficios ecológicos su producto?", "answer": "Reduce la huella de carbono y el impacto en los vertederos en comparación con los envases plásticos convencionales. Ofrece a las marcas de cosmética natural la posibilidad de cumplir con sus promesas de sostenibilidad genuina"},
    {"question": "¿Qué papel desempeña su producto en la ecología?", "answer": "Reduce la huella de carbono y el impacto en los vertederos en comparación con los envases plásticos convencionales. Ofrece a las marcas de cosmética natural la posibilidad de cumplir con sus promesas de sostenibilidad genuina"},


    {"question": "¿El producto es transparente?", "answer": "Aporta credibilidad a las marcas al evitar el greenwashing y promover prácticas transparentes y honestas."},
    {"question": "¿Cómo se caracteriza la transparencia de su producto?", "answer": "Aporta credibilidad a las marcas al evitar el greenwashing y promover prácticas transparentes y honestas."},
    {"question": "¿Puede hablarme sobre la transparencia de su producto?", "answer": "Aporta credibilidad a las marcas al evitar el greenwashing y promover prácticas transparentes y honestas."},
    {"question": "¿Qué papel juega la transparencia en su producto?", "answer": "Aporta credibilidad a las marcas al evitar el greenwashing y promover prácticas transparentes y honestas."},
    {"question": "¿Por qué es importante la transparencia en su producto?", "answer": "Aporta credibilidad a las marcas al evitar el greenwashing y promover prácticas transparentes y honestas."},


    {"question": "¿Cómo es la experiencia con el usuario?", "answer": "Mejora la percepción de la marca por parte del consumidor al alinearse con los valores de sostenibilidad."},
    {"question": "¿Cómo se percibe la marca por parte del consumidor?", "answer": "Mejora la percepción de la marca por parte del consumidor al alinearse con los valores de sostenibilidad."},
    {"question": "¿Qué impacto tiene en la percepción del consumidor?", "answer": "Mejora la percepción de la marca por parte del consumidor al alinearse con los valores de sostenibilidad."},
    {"question": "¿Cuál es el beneficio para la percepción del consumidor?", "answer": "Mejora la percepción de la marca por parte del consumidor al alinearse con los valores de sostenibilidad."},
    {"question": "¿En qué medida afecta la experiencia del usuario?", "answer": "Mejora la percepción de la marca por parte del consumidor al alinearse con los valores de sostenibilidad."},


    {"question": "¿Es sostenible el producto?", "answer": "Al ser completamente compostables, estos envases cierran el ciclo de vida del producto de manera sostenible."},
    {"question": "¿Cuál es el enfoque de sostenibilidad de su producto?", "answer": "Al ser completamente compostables, estos envases cierran el ciclo de vida del producto de manera sostenible."},
    {"question": "¿Cómo contribuye su producto a la sostenibilidad?", "answer": "Al ser completamente compostables, estos envases cierran el ciclo de vida del producto de manera sostenible."},
    {"question": "¿Qué hace que su producto sea sostenible?", "answer": "Al ser completamente compostables, estos envases cierran el ciclo de vida del producto de manera sostenible."},
    {"question": "¿Puede hablar sobre la sostenibilidad de su producto?", "answer": "Al ser completamente compostables, estos envases cierran el ciclo de vida del producto de manera sostenible."},


    {"question": "¿Qué beneficios trae para la marca?", "answer": "Imagen de Marca: Posiciona a las empresas como líderes en la adopción de prácticas sostenibles.\n Marketing: Ofrece una narrativa poderosa para campañas de marketing que resaltan la responsabilidad ambiental.\nMercado: Abre nuevas oportunidades de mercado entre consumidores eco-conscientes y mejora la competitividad."},
    {"question": "¿Cómo beneficia a la marca la adopción de prácticas sostenibles?", "answer": "Imagen de Marca: Posiciona a las empresas como líderes en la adopción de prácticas sostenibles.\n Marketing: Ofrece una narrativa poderosa para campañas de marketing que resaltan la responsabilidad ambiental.\nMercado: Abre nuevas oportunidades de mercado entre consumidores eco-conscientes y mejora la competitividad."},
    {"question": "¿Por qué adoptar prácticas sostenibles beneficia a la marca?", "answer": "Imagen de Marca: Posiciona a las empresas como líderes en la adopción de prácticas sostenibles.\n Marketing: Ofrece una narrativa poderosa para campañas de marketing que resaltan la responsabilidad ambiental.\nMercado: Abre nuevas oportunidades de mercado entre consumidores eco-conscientes y mejora la competitividad."},
    {"question": "¿Cuáles son los resultados positivos para la marca al adoptar prácticas sostenibles?", "answer": "Imagen de Marca: Posiciona a las empresas como líderes en la adopción de prácticas sostenibles.\n Marketing: Ofrece una narrativa poderosa para campañas de marketing que resaltan la responsabilidad ambiental.\nMercado: Abre nuevas oportunidades de mercado entre consumidores eco-conscientes y mejora la competitividad."},
    {"question": "¿Cómo afecta la adopción de prácticas sostenibles a la marca?", "answer": "Imagen de Marca: Posiciona a las empresas como líderes en la adopción de prácticas sostenibles.\n Marketing: Ofrece una narrativa poderosa para campañas de marketing que resaltan la responsabilidad ambiental.\nMercado: Abre nuevas oportunidades de mercado entre consumidores eco-conscientes y mejora la competitividad."},


    {"question": "¿Qué uso o aplicación se le puede dar?", "answer": "Especialmente adecuado para productos hasta 10 gramos, como maquillajes en polvo, bálsamos y barritas de cuidado persona."},
    {"question": "¿Para qué tipo de productos es adecuado su envase?", "answer": "Especialmente adecuado para productos hasta 10 gramos, como maquillajes en polvo, bálsamos y barritas de cuidado persona."},
    {"question": "¿Cuáles son las aplicaciones recomendadas para su envase?", "answer": "Especialmente adecuado para productos hasta 10 gramos, como maquillajes en polvo, bálsamos y barritas de cuidado persona."},
    {"question": "¿Puede darme ejemplos de productos compatibles con su envase?", "answer": "Especialmente adecuado para productos hasta 10 gramos, como maquillajes en polvo, bálsamos y barritas de cuidado persona."},
    {"question": "¿Cuál es el rango de productos recomendado para su envase?", "answer": "Especialmente adecuado para productos hasta 10 gramos, como maquillajes en polvo, bálsamos y barritas de cuidado persona."},


    {"question": "¿Qué objetivo tiene?", "answer": "Cosmética y belleza, incluyendo maquillaje, cuidado de la piel y productos para el cabello."},
    {"question": "¿Cuáles son los objetivos de su producto?", "answer": "Cosmética y belleza, incluyendo maquillaje, cuidado de la piel y productos para el cabello."},
    {"question": "¿Puede describir los objetivos de su producto?", "answer": "Cosmética y belleza, incluyendo maquillaje, cuidado de la piel y productos para el cabello."},
    {"question": "¿A qué se dirige su producto?", "answer": "Cosmética y belleza, incluyendo maquillaje, cuidado de la piel y productos para el cabello."},
    {"question": "¿En qué categorías se enfoca su producto?", "answer": "Cosmética y belleza, incluyendo maquillaje, cuidado de la piel y productos para el cabello."},

    {"question": "¿Me pueden enviar su catálogo?", "answer": "Por el momento, no contamos con un catálogo, pues solo manejamos un tipo de envase. Nuestro envase es especialmenete para cométicos naturales, ideales par productos sólidos y semisólidos de hasta 10gr. Es precio de la unidad es de 20. con un pedido mínimo de 200 unidades."},
    {"question": "¿Cuentan con algún catálogo?", "answer": "Por el momento, no contamos con un catálogo, pues solo manejamos un tipo de envase. Nuestro envase es especialmenete para cométicos naturales, ideales par productos sólidos y semisólidos de hasta 10gr. Es precio de la unidad es de 20. con un pedido mínimo de 200 unidades."},
    
    {"question": "¿Podrían darme más infomación del envase?", "answer": "¡Por supuesto! Nuestros envases están diseñados especialmente para cosméticos naturales, ideales para productos sólidos de hasta 10gr. \n\n Son completamente compostables, lo que los hace perfectos para marcas conscientes del medio ambiente. El precio por unidad es de $20, con un pedido mínimo de 200 unidades. \n\n ¿Hay algo específico que te gustaría saber?"},

    {"question": "¿Cuánto tiempo tarda el envío?", "answer": "Normalmente, el tiempo de producción y envío puede variar. Actualmente tenemos una alta demanda por el lanzamiento del producto pero nos esforzamos por ser los más eficientes posible.\n\nUna vez recibimos un adelanto del 50%, el pedido tardaría entre 3 a 4 semanas en estar listo para enviar. Una vez que el paquete esté listo, se te notificará para pagar el saldo restante y posteriormente hacer el envío.\n\n¿Tienes alguna fecha específica para la entrega?\nHaremos todo lo posible para cumplir tus plazos."},

    {"question": "¿En qué producto se pueden usar los envases?", "answer": "Nuestros envases son ideales para cosméticos naturales en polvo, sólidos y semisólidos a base de aceite o grasa vegetal, hasta 10gr. Son perfectos para marcas que buscan una solución sostenible y elegante. Cada envase cuesta $20 y el pedido mínimo es de 200 unidades."},
    
    {"question": "¿Cómo puedo comprar un producto?", "answer": "El proceso de compra se hará mediante pago en el sitio web. Se integrará una pasarela de pago 🛒. Deberá crear un usuario y validar su cuenta. Una vez que agregue productos al carrito de compra, podrá proceder al pago y confirmar su pedido. Después de confirmar el pago, recibirá un correo de confirmación del pedido, y se iniciará la producción. A través de su usuario, podrá monitorear el proceso del pedido. Una vez que esté listo, se le enviará la guía de envío 📦."},
    
    {"question": "El mínimo de piezas es mucho para mi.", "answer": "Entiendo que pueda paracer un compromiso grande. Sin embargo, este volumen nos permite mantener la calidad y sostenibilidad a nuestros envases."},

    {"question": "Me preocupa que el precio sea muy alto", "answer": "Es una inversión importante, lo entiendo. Nuestro producrto refelja la calidad premium y la sostenibilidad del producto. Estos envases no solo protegen tus productos, sino que también apoyan tus esfuerzos de marca de responsabilidad ambiental. \n\n  Además ofrecemos soporte y asesoría para que puedas comunicar estos valores a tus clientes, lo que podría ayudarte a justificar el precio ante ellos.\n\n¿Te gustaría saber cómo otros clientes han utilizado esto a su favor?"},



]

questions = [entry["question"] for entry in faq_data]
answers = [entry["answer"] for entry in faq_data]

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(questions, answers)

# Función para calcular la similitud coseno entre dos textos
def calculate_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0, 0]

# Función para encontrar la pregunta predefinida más similar a la pregunta del usuario
def find_most_similar_question(user_input, predefined_questions, threshold=0.5):
    similarities = [calculate_similarity(user_input, q) for q in predefined_questions]
    max_similarity = max(similarities)
    
    if max_similarity >= threshold:
        max_similarity_index = np.argmax(similarities)
        return predefined_questions[max_similarity_index]
    else:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
@cross_origin()
def chat():
    user_input = request.json['user_input']
    
    matched_question = find_most_similar_question(user_input, questions)
    
    if matched_question is not None:
        response = [entry["answer"] for entry in faq_data if entry["question"] == matched_question][0]
    else:
        # Si no hay coincidencia por encima del umbral, devuelve un mensaje específico
        response = "Lo siento, no tengo información sobre esa pregunta."

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)