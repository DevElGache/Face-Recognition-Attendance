import face_recognition
from PIL import Image, ImageDraw, ImageFont
import os

# Lists where we will store known face encodings and corresponding person names
known_face_encodings = []
known_face_names = []

# Loop through all files in the "people" folder and load them
# Typically each file is a photo of a single person used as a reference
for filename in os.listdir("people"):
    # Only process JPEG and PNG files
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image file from the people directory
        image = face_recognition.load_image_file(f"people/{filename}")
        # Compute face encodings for faces in the image, take the first encoding
        encoding = face_recognition.face_encodings(image)[0]
        # Append the encoding and the base filename (name without extension)
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Print loaded faces for quick verification
print("✅ Loaded known faces:", known_face_names)

# Try to load a TrueType font for drawing names; fall back to default font if unavailable
try:
    # Common DejaVu font path on Ubuntu systems
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font_size = 24  # Set the font size (adjust as needed)
    font = ImageFont.truetype(font_path, font_size)
except Exception:
    # If loading the TTF font fails, use Pillow's default bitmap font
    font = ImageFont.load_default()

# Iterate over images in the "images_to_test" folder tzo find and label faces
for test_file in os.listdir("images_to_test"):
    # Only process JPEG and PNG files
    if test_file.endswith(".jpg") or test_file.endswith(".png"):
        test_path = f"images_to_test/{test_file}"
        # Load the test image into a numpy array (face_recognition format)
        test_image = face_recognition.load_image_file(test_path)

        # Detect face locations (top, right, bottom, left tuples)
        face_locations = face_recognition.face_locations(test_image)
        # Compute face encodings for the detected face locations
        face_encodings = face_recognition.face_encodings(test_image, face_locations)

        # Convert the image to a PIL Image for drawing
        pil_image = Image.fromarray(test_image)
        draw = ImageDraw.Draw(pil_image)

        # Keep track of names detected in the current image
        names_in_image = []

        # Iterate over each found face location and its encoding
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare this face encoding against the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            # Default label if no known match is found
            name = "Unknown"
            # If any match is True, take the first matched known face
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Record the detected name for reporting
            names_in_image.append(name)

            # Draw a rectangle around the face (green outline, 3px width)
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)

            # Calculate text size to position label so it stays inside the image
            try:
                # Preferred: use font.getsize if available to get text width and height
                text_width, text_height = font.getsize(name)
            except Exception:
                # Fallback: use draw.textbbox (Pillow >= 8.0) to compute text bounding box
                try:
                    bbox = draw.textbbox((0, 0), name, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except Exception:
                    # Last resort: approximate text size from font properties or name length
                    text_width = len(name) * (font.size // 2 if hasattr(font, "size") else 10)
                    text_height = font.size if hasattr(font, "size") else 12

            # Start text at the left edge of the face rectangle
            text_x = left
            # Try to place text above the face; if it does not fit, place below the face
            text_y = top - text_height - 4
            if text_y < 0:
                text_y = top + 4  # If not enough room above, put label below the face

            # Draw the name using the selected font and green color
            draw.text((text_x, text_y), name, fill=(0, 255, 0), font=font)

        # Show the resulting image with drawn rectangles and labels
        pil_image.show()
        # Save result image with a "result_" prefix in the current directory
        pil_image.save(f"result_{test_file}")

        # Print a short summary of processing for this test file
        print(f"✅ Processed {test_file}. Faces found: {names_in_image}")
