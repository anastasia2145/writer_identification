from contrastive_model import ContrastiveModel

train_dir = 'C:/Users/Anastasia/Pictures/words_train'
validation_dir = 'C:/Users/Anastasia/Pictures/words_validation'
test_dir = '../data/words_test'

# Train
# model = ContrastiveModel(alpha=1, input_shape=(160, 160, 3), cache_dir="contrastive_cache")
# model.train(train_dir, "../train.csv", validation_dir, "../validation.csv", epochs=500)

# Predict
# model = ContrastiveModel(alpha=1, input_shape=(160, 160, 3), cache_dir="contrastive_cache")
# model.load_weights("final_weigths_alpha_1/final.h5")
# model.load_embeddings('../data/embeddings_contrastive_1.pkl')
# model.make_embeddings(train_dir, "train.csv", batch_size=1)
# model.predict(test_dir, "../data/test.csv", batch_size=1)
