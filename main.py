import tkinter as tk
from tkinter import filedialog
import base64
import numpy as np
import cv2
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import threading
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn


chrome_driver_path = "C:/Program Files/chromedriver-win64/chromedriver.exe"
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service)


class TrainedModel:
    def __init__(self):
        start_time = time.time()
        self.model = models.squeezenet1_0(pretrained=False)
        self.model.classifier[1] = nn.Conv2d(512, 30, kernel_size=(1, 1), stride=(1, 1))
        model_path = "C:/Users/ccl/Desktop/trained_model.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        print(f"Model loaded in {time.time() - start_time:.4f} seconds")

    def predict(self, img):
        resized_image = cv2.resize(img, (160, 90))  # Resize for faster processing
        pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to grayscale for faster computation
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5], [0.5])
        ])
        tensor_image = preprocess(pil_image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(tensor_image).view(-1, 30)

        num1_preds = outputs[:, :10]
        operation_preds = outputs[:, 10:13]
        num2_preds = outputs[:, 13:]

        _, num1_predicted = torch.max(num1_preds, 1)
        _, operation_predicted = torch.max(operation_preds, 1)
        _, num2_predicted = torch.max(num2_preds, 1)

        operation_map = {0: "+", 1: "-", 2: "×"}
        predicted_operation = operation_map[operation_predicted.item()]

        return num1_predicted.item(), predicted_operation, num2_predicted.item()


def show_captcha(captcha_data):
    try:
        captcha_base64 = captcha_data.split(",")[1] if "," in captcha_data else captcha_data
        captcha_image_data = np.frombuffer(base64.b64decode(captcha_base64), dtype=np.uint8)
        captcha_image = cv2.imdecode(captcha_image_data, cv2.IMREAD_COLOR)
        if captcha_image is None:
            print("Failed to decode captcha image from memory.")
            return None
        return captcha_image
    except Exception as e:
        print(f"Error decoding captcha: {e}")
        return None


def switch_to_ecsc_tab():
    for handle in driver.window_handles:
        driver.switch_to.window(handle)
        if "ecsc.gov.sy" in driver.current_url:
            print(f"Switched to tab with URL: {driver.current_url}")
            return True
    print("No tab found with ecsc.gov.sy")
    return False


def click_yes_buttons():
    try:
        driver.execute_script("""
            var confirmButton = document.querySelector('button.swal2-confirm.swal2-styled');
            if (confirmButton) { confirmButton.click(); }
        """)
        driver.execute_script("""
            var yesButton = document.querySelector('div[mat-dialog-actions] button[type="button"]');
            if (yesButton) { yesButton.click(); }
        """)
    except Exception as e:
        print(f"Error clicking buttons: {e}")


class OCRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OCR with Captcha")
        self.background_images = []
        self.running = False

        self.instruction_label = tk.Label(self, text="يرجى إضافة خلفيات مرة واحدة قبل التشغيل", font=("Arial", 14))
        self.instruction_label.pack(pady=20)

        self.add_background_button = tk.Button(self, text="إضافة خلفية", command=self.add_background)
        self.add_background_button.pack(pady=10)

        self.start_button = tk.Button(self, text="بدء التشغيل", command=self.start_automatic_process)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self, text="إيقاف", command=self.stop_automatic_process, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

    def add_background(self):
        background_paths = filedialog.askopenfilenames(
            title="اختر صور الخلفية", filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )
        for path in background_paths:
            background_image = cv2.imread(path)
            if background_image is not None:
                self.background_images.append(background_image)

    def remove_background(self, captcha_image):
        if not self.background_images:
            return self.denoise_image(captcha_image)  # Apply denoise if no background

        best_background = None
        min_diff = float("inf")

        # Resize the images for faster processing
        captcha_image_resized = cv2.resize(captcha_image, (160, 90))

        for background in self.background_images:
            resized_bg = cv2.resize(background, (captcha_image_resized.shape[1], captcha_image_resized.shape[0]))
            diff = cv2.absdiff(captcha_image_resized, resized_bg)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            score = np.sum(gray_diff)
            if score < min_diff:
                min_diff = score
                best_background = resized_bg

        if best_background is not None:
            return self.remove_background_keep_original_colors(captcha_image, best_background)
        return self.denoise_image(captcha_image)

    def remove_background_keep_original_colors(self, captcha_image, background_image):
        diff = cv2.absdiff(captcha_image, background_image)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Use faster thresholding method
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        result = cv2.bitwise_and(captcha_image, captcha_image, mask=mask)
        
        # Optional: convert all white pixels to black to remove residuals
        result[np.all(result == [255, 255, 255], axis=-1)] = [0, 0, 0]

        return self.denoise_image(result)

    def denoise_image(self, image):
        # Fast denoising method from OpenCV
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    def start_automatic_process(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.running = True
        thread = threading.Thread(target=self.automatic_process_thread)
        thread.start()

    def stop_automatic_process(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def automatic_process_thread(self):
        model = TrainedModel()

        while self.running:
            try:
                if not switch_to_ecsc_tab():
                    continue

                image_element = driver.find_element(By.CSS_SELECTOR, "div > img.swal2-image")
                captcha_data = image_element.get_attribute('src')

                captcha_image = show_captcha(captcha_data)
                if captcha_image is None:
                    continue

                # Remove background and denoise the image
                captcha_image = self.remove_background(captcha_image)

                # Model prediction
                num1, operation, num2 = model.predict(captcha_image)
                result = self.calculate_result(num1, operation, num2)
                print(f"العملية: {num1} {operation} {num2} = {result}")

                input_element = driver.find_element(By.CSS_SELECTOR, "input.swal2-input")
                input_element.clear()
                input_element.send_keys(str(result))

                confirm_button = driver.find_element(By.CSS_SELECTOR, "button.swal2-confirm.swal2-styled")
                confirm_button.click()

                click_yes_buttons()

            except Exception as e:
                print(f"Error during process: {e}")

    def calculate_result(self, num1, operation, num2):
        if operation == "+":
            return num1 + num2
        elif operation == "-":
            return num1 - num2
        elif operation == "×":
            return num1 * num2
        else:
            raise ValueError("عملية غير معروفة")


app = OCRApp()
app.mainloop()

driver.quit()
