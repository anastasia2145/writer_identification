# Writer identification 

### Code will be added 18.05

### Классический подход

Для воспроизведения результатов классического подхода, основанного на построении кодовой книги воспользуйтесь **Demonstrate_results.ipynb**

Вектор-признаки, полученные с использованием кодовой книги, и лейблы для них расположены в:

* data/train_embeddings_13.npy 
* data/val_embeddings_13.npy 
* data/test_embeddings_13.npy 
* data/train_y.npy 
* data/val_y.npy 
* data/test_y.npy 

Вектор-признаки, полученные с помощью признаков OBIFs:

* OBIFs/train_obifs.npy 
* OBIFs/val_obifs.npy 
* OBIFs/test_obifs.npy 

### Нейросетевой подход
Для воспроизведения нейросетевого подхода необходимо скачать **words_test** и положить в папку **data**: https://www.dropbox.com/sh/u88myildupn1jlk/AABpHmtpFMLpsWYPljdN9lLra?dl=0

words_test - это извлеченные слова, для тестовой выборки. Т.к. в некоторых случаях для извлечения слов необходимо подбирать параметры.

1) Классификация 

Использовать **classification/main.py**

Финальные веса моделей находятся в папках **classification/final_weigths_alpha_1** и **classification/final_weigths_alpha_0.75**

2) Triplet loss

Использовать **triplets/main.py**

Финальные веса моделей находятся в папках **triplets/final_weigths_alpha_1** и **triplets/final_weigths_alpha_0.75**

Embedding-и для тренеровочного множества **data/triplet_embeddings_1.pkl** и **data/triplet_embeddings_75.pkl**

3) Contrastive loss

Использовать **contrastive/main.py**

Финальные веса в папке **contrastive/final_weigths_alpha_1** 

Embedding-и для тренеровочного множества **data/contrastive_embeddings_1.pkl** 
