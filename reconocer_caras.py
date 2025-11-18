import face_recognition
from PIL import Image, ImageDraw
import os

# lstas donde vamos a cargar caras y nombres de las personas, codificados
known_face_encodings = []
known_face_names = []

#buvle para recorrer todos os archivos de la carpeta "people" y que se guarde
#normalmente será una foto de una unica persona
for filename in os.listdir("people"):
    #filtramos archivos  por extensión
    if filename.endswith(".jpg") or filename.endswith(".png"):
        #carga imagenes
        image = face_recognition.load_image_file(f"people/{filename}")
        #[0] para tomar unicamente el primer vector (prmera cara que reconoce) que devuelve la funcion
        encoding = face_recognition.face_encodings(image)[0]
        #guarda la cara y nombre en listas
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])

#test
print("✅ Loaded known faces:", known_face_names)

#mismo bucle pero para reconocer personas previamente reconocidas
for test_file in os.listdir("images_to_test"):
    if test_file.endswith(".jpg") or test_file.endswith(".png"):
        test_path = f"images_to_test/{test_file}"
        test_image = face_recognition.load_image_file(test_path)

        #la función lo que hace es vectorizar las coordenadas de las caras (top, right, bottom, left)
        face_locations = face_recognition.face_locations(test_image)
        #lista los vectores de los rostros detectados, codificados
        face_encodings = face_recognition.face_encodings(test_image, face_locations)

        #test
        pil_image = Image.fromarray(test_image)
        #test
        draw = ImageDraw.Draw(pil_image)

        #guarda en una lista los nombres de las personas conocidas que detecta en la imagen
        names_in_image = []

        #recorre cada rostro detectado con sus vector de coordenadas
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            #hay rostro coincidiente
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            #no hay rostro coincidiente, por tanto, unknown
            name = "Unknown"

            #asigna nombres a sus rostros
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            #guarda el nombre en la lista de resultados e la imagen a analizar
            names_in_image.append(name)

            #test, gacias a convertirlo en objeto PIL, y permitir el "draw", dibuja un rectangulo
            #en cada cara reconocida, conocida o desconocida
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)
            draw.text((left, top - 10), name, fill=(0, 255, 0))

        #test, abre la imagen y la guarda
        pil_image.show()
        pil_image.save(f"resultado_{test_file}")

        #test, muesta por consola el resultado
        print(f"✅ Processed {test_file}. Faces found: {names_in_image}")
