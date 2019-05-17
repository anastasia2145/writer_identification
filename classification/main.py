from classification import Classification_model

cache_dir = 'classification_cache'
# train_dir = 'C:/Users/Anastasia/Pictures/words_train'
# validation_dir = 'C:/Users/Anastasia/Pictures/words_validation'
test_dir = '../data/words_test'

# Train
# model = Classification_model(alpha=1, input_shape=(160,160,3), num_classes=95, cache_dir=cache_dir, train_head=False)
# model.load_weights('classification_cache_new/train_all_2/checkpoint-20.h5')
# model.train(train_dir, "train.csv", epochs=100)

# Predict
model = Classification_model(alpha=1, input_shape=(160,160,3), num_classes=95, cache_dir=cache_dir, train_head=False)
model.load_weights('final_weigths_alpha_1/final.h5') #42
model.predict(test_dir, "../data/test.csv")


