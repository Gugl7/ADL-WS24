from paddleocr import PaddleOCR
import csv
import os

class ImageProcessor:
    """Handles all image processing and OCR operations."""

    def write_image_to_csv(self, img_path, csv_file_path):
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w') as file:
                pass
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(img_path, cls=True)

        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for row in result[0]:
                coordinates = row[0]
                text = row[1][0]
                csv_row = [coordinates[0][0], coordinates[0][1], coordinates[1][0], coordinates[1][1], coordinates[2][0], coordinates[2][1], coordinates[3][0], coordinates[3][1], text]
                writer.writerow(csv_row)
                
if __name__ == "__main__":
    img_path = "sample.jpg"
    csv_file_path = "sample.csv"
    ImageProcessor.write_image_to_csv(ImageProcessor(), img_path, csv_file_path)