from triplet_model import TripletModel

train_dir = 'C:/Users/Anastasia/Pictures/words_train'
validation_dir = 'C:/Users/Anastasia/Pictures/words_validation'
test_dir = '../data/words_test'

# Train
# model = TripletModel(input_shape=(160, 160, 3), cache_dir="triplet_cache_new")
# model.load_weights("triplet_cache_new/train_all/checkpoint-06.h5")
# model.train(train_dir, "train.csv", validation_dir, "validation.csv", epochs=200)
 
# Predict
model = TripletModel(alpha=0.75, input_shape=(160, 160, 3), cache_dir="triplet_cache")
model.load_weights("final_weigths_alpha_0.75/final.h5")
model.load_embeddings('../data/triplet_embeddings_75.pkl')
model.make_embeddings(train_dir, "train.csv", batch_size=1)
model.predict(test_dir, "../data/test.csv", batch_size=1)