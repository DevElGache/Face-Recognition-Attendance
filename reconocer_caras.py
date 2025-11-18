import face_recognition
from PIL import Image, ImageDraw
import os

# --- Step 1: Load known people ---
known_face_encodings = []
known_face_names = []

for filename in os.listdir("people"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = face_recognition.load_image_file(f"people/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])

print("✅ Loaded known faces:", known_face_names)

# --- Step 2: Process all images in the test folder ---
for test_file in os.listdir("images_to_test"):
    if test_file.endswith(".jpg") or test_file.endswith(".png"):
        test_path = f"images_to_test/{test_file}"
        test_image = face_recognition.load_image_file(test_path)

        # Detect faces and encodings
        face_locations = face_recognition.face_locations(test_image)
        face_encodings = face_recognition.face_encodings(test_image, face_locations)

        pil_image = Image.fromarray(test_image)
        draw = ImageDraw.Draw(pil_image)

        # Keep track of all recognized names in this image
        names_in_image = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            names_in_image.append(name)

            # Draw rectangle and label
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)
            draw.text((left, top - 10), name, fill=(0, 255, 0))

        # --- Step 3: Show results ---
        pil_image.show()
        pil_image.save(f"resultado_{test_file}")

        print(f"✅ Processed {test_file}. Faces found: {names_in_image}")
