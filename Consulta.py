from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Mapear os números das classes para os nomes das doenças
class_names = {
    0: 'Carcinoma basocelular',
    1: 'Dermatofibroma',
    2: 'Nevo melanocítico',
    3: 'Melanoma',
    4: 'Granulomas piogênicos/hemangioma capilar lobular',
    5: 'Ceratose seborreica',
    6: 'Carcinoma espinocelular',
    7: 'Lesões vasculares',
    8: 'Ceratoses actínicas/Doença de Bowen (carcinoma intraepitelial)'
}

# Limiar de confiança
threshold = 0.95

# Carregar o modelo
loaded_model = load_model('skin_cancer_model.keras', compile=False)

# Caminho para a imagem a ser testada
image_path = r"C:\users\Danni\Downloads\nova pasta\tcc\IMAGEM_TESTE.jpg"

# Carregar a imagem
img = Image.open(image_path)

# Redimensionar a imagem para o tamanho esperado pelo modelo
img = img.resize((100, 75))

# Converter a imagem para um array numpy
img_array = image.img_to_array(img)

# Adicionar uma dimensão extra para a amostra
img_array = np.expand_dims(img_array, axis=0)

# Pré-processar a imagem (normalização)
img_array = img_array.astype('float32') / 255.0

# Fazer a previsão usando o modelo carregado
prediction = loaded_model.predict(img_array)

# Obter a classe prevista e a probabilidade máxima
predicted_class = np.argmax(prediction)
max_probability = np.max(prediction)

# Verificar se a probabilidade máxima está acima do limiar de confiança
if max_probability >= threshold:
    # Obter o nome da doença correspondente à classe prevista
    predicted_disease = class_names[predicted_class]
else:
    # Atribuir "desconhecido" se a probabilidade máxima estiver abaixo do limiar
    predicted_disease = "desconhecido"

# Exibir a doença prevista
print("Doença prevista:", predicted_disease)
