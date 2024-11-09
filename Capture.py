import cv2
import numpy as np
import os
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from ultralytics import YOLO
import tensorflow as tf

class CameraApp(App):
    def build(self):
        # Layout principal
        self.layout = BoxLayout(orientation='vertical')

        # Widget para exibir o feed da câmera ou a imagem capturada
        self.img_widget = Image()
        self.layout.add_widget(self.img_widget)

        # Botão para tirar a foto
        self.capture_button = Button(text="Tirar Foto", size_hint=(1, 0.2))
        self.capture_button.bind(on_press=self.capture_image)
        self.layout.add_widget(self.capture_button)

        # Variável para controlar se a câmera está ativa ou congelada
        self.camera_active = True
        self.image_frozen = False  # Adiciona variável para controle de congelamento

        # Iniciar a câmera (OpenCV)
        self.capture = cv2.VideoCapture(0)

        # Atualizar o feed da câmera a cada intervalo de tempo
        Clock.schedule_interval(self.update_camera_feed, 1.0 / 30.0)

        # Definir pasta para salvar imagens
        self.save_directory = os.path.join(os.getcwd(), 'imagens')
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        # Carregar modelos
        self.yolo_model = YOLO('caminho para o seu modelo yolo')
        # Carregar modelos TFLite
        self.interpreter1 = tf.lite.Interpreter(model_path='caminho para seu TFLite')
        self.interpreter1.allocate_tensors()
        self.input_details1 = self.interpreter1.get_input_details()
        self.output_details1 = self.interpreter1.get_output_details()

        self.interpreter2 = tf.lite.Interpreter(model_path='caminho para seu TFLite')
        self.interpreter2.allocate_tensors()
        self.input_details2 = self.interpreter2.get_input_details()
        self.output_details2 = self.interpreter2.get_output_details()

        self.interpreter3 = tf.lite.Interpreter(model_path='caminho para seu TFLite')
        self.interpreter3.allocate_tensors()
        self.input_details3 = self.interpreter3.get_input_details()
        self.output_details3 = self.interpreter3.get_output_details()

        return self.layout

    def update_camera_feed(self, dt):
        if self.camera_active:
            # Captura o frame da câmera
            ret, frame = self.capture.read()
            if ret:
                # Converter o frame para um formato de textura (Kivy)
                buf = cv2.flip(frame, 0).tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.img_widget.texture = texture

    
    def capture_image(self, *args):
        # Se a imagem está congelada, reative a câmera
        if self.image_frozen:
            self.camera_active = True  # Ativa a câmera novamente
            self.image_frozen = False  # Marca a imagem como não congelada
            # Reinicia o feed da câmera
            Clock.schedule_interval(self.update_camera_feed, 1.0 / 60.0)
        else:
            # Captura a imagem atual da câmera e congela
            ret, frame = self.capture.read()
            if ret:
                self.camera_active = False  # Congela a câmera
                self.image_frozen = True  # Marca a imagem como congelada

                # Analisar a imagem capturada
                self.analyze_image(frame)

    def analyze_image(self, frame):
        # Limiar de confiança
        confidence_threshold = 0.65  

        # Detectar objetos usando YOLO com o limiar de confiança
        results = self.yolo_model.track(frame, persist=True, conf=confidence_threshold)
        height, width, _ = frame.shape

        # Criar uma imagem para desenhar as previsões
        frame_ = frame.copy()

        # Processar as detecções
        if results:
            boxes = results[0].boxes
            predictions = []

            for box in boxes:
                confidence = box.conf[0].item()  # Pegando a confiança da detecção
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da caixa delimitadora
                crop_obj = frame[y1:y2, x1:x2]
                crop_obj = cv2.resize(crop_obj, (150, 150))
                crop_obj = crop_obj / 255.0
                crop_obj = np.expand_dims(crop_obj, axis=0).astype(np.float32)  # TFLite espera que a entrada seja float32

                # Previsão para o modelo TFLite
                self.interpreter1.set_tensor(self.input_details1[0]['index'], crop_obj)
                self.interpreter1.invoke()
                prediction1 = self.interpreter1.get_tensor(self.output_details1[0]['index'])[0][0]  # Rust

                self.interpreter2.set_tensor(self.input_details2[0]['index'], crop_obj)
                self.interpreter2.invoke()
                prediction2 = self.interpreter2.get_tensor(self.output_details2[0]['index'])[0][0]  # Healthy

                self.interpreter3.set_tensor(self.input_details3[0]['index'], crop_obj)
                self.interpreter3.invoke()
                prediction3 = self.interpreter3.get_tensor(self.output_details3[0]['index'])[0][0]  # Apple Scab

                # Armazenar as previsões
                predictions.append((confidence, prediction1, prediction2, prediction3, (x1, y1, x2, y2)))

                # Desenhar a caixa delimitadora na imagem congelada
                cv2.rectangle(frame_, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Caixa em azul

            # Analisar as previsões para encontrar as duas maiores
            for _, pred1, pred2, pred3, (x1, y1, x2, y2) in predictions:
                total = pred1 + pred2 + pred3
                if total > 0:  # Evitar divisão por zero
                    rust_percentage = (pred1 / total) * 100
                    healthy_percentage = (pred2 / total) * 100
                    apple_scab_percentage = (pred3 / total) * 100

                    # Criar uma lista de previsões e suas porcentagens
                    results = [
                        ("FERRUGEM", rust_percentage),
                        ("SAUDAVEL", healthy_percentage),
                        ("SARNA DE MACA", apple_scab_percentage)

                    ]

                    # Classificar as previsões com base na porcentagem
                    results.sort(key=lambda x: x[1], reverse=True)

                    # Obter os dois maiores resultados
                    top_predictions = results[:2]

                    # Desenhar as labels na imagem
                    for i, (label, percentage) in enumerate(top_predictions):
                        cv2.putText(frame_, f'{label}: {percentage:.2f}%', (x1, y1 - 10 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Atualiza a imagem congelada com as previsões
            buf = cv2.flip(frame_, 0).tobytes()
            texture = Texture.create(size=(frame_.shape[1], frame_.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img_widget.texture = texture

        else:
            # Se nenhum objeto for detectado, exibir mensagem na imagem congelada
            cv2.putText(frame_, 'Nenhum objeto detectado.', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            buf = cv2.flip(frame_, 0).tobytes()
            texture = Texture.create(size=(frame_.shape[1], frame_.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img_widget.texture = texture

    def on_stop(self):
        # Libera a câmera ao fechar o aplicativo
        if self.capture.isOpened():
            self.capture.release()
if __name__ == '__main__':
    CameraApp().run()
