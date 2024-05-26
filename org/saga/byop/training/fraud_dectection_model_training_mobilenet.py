from keras.applications import mobilenet_v2
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


class PrintCallback(keras.callbacks.Callback):
    def __init__(self, num_epochs=10):
        self.num_epochs = num_epochs

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch No. {epoch + 1}/{self.num_epochs}")

    def on_batch_begin(self, batch, logs=None):
        print(f"\tBatch No. {batch + 1}")


class fraud_dectection_model_training_mobilenet:
    def __init__(self, folder_path, target_size=(180, 180), batch_size=32,EPOCH=10):
        self.model = None
        self.target_size = target_size
        self.batch_size = batch_size
        self.EPOCH=EPOCH
        validation_split = 0.2
        self.test_generator = None
        datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=validation_split)

        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split
        )

        self.train_generator = train_datagen.flow_from_directory(
            folder_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        self.validation_generator = datagen.flow_from_directory(
            folder_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        print("Train DS", len(self.train_generator))
        print("Validation DS", len(self.validation_generator))
        self.class_names = list(self.train_generator.class_indices.keys())
        print(self.class_names)
        self.num_classes = len(self.train_generator.class_indices)

    def build_model(self):
        base_model = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)

    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(self.model.summary())

    def fit_model(self, checkpoint_filepath, epochs=10):
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )
        print_callback = PrintCallback(num_epochs=epochs)
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=[model_checkpoint_callback, print_callback]
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()

    def run_training(self, folder_path, checkpoint_filepath):
        self.__init__(folder_path)
        self.build_model()
        self.compile_model()
        self.fit_model(checkpoint_filepath, self.EPOCH)

    def evaluate(self, test_data_dir, batch_size=32):
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        return self.model.evaluate(self.test_generator)

    def predict(self):
        image_batch, label_batch = self.test_generator.next()
        predictions = self.model.predict_on_batch(image_batch)

        print('Predictions:\n', predictions)
        print('Labels:\n', label_batch)

        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].astype("uint8"))
            plt.title(self.class_names[predictions[i].argmax()])
            plt.axis("off")


    def save_model(self, model_path):
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")


# Example usage
if __name__ == "__main__":
    folder_path = "C:\\Users\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_training"
    checkpoint_filepath = "/tmp/ckpt/checkpoint.model.keras"
    epochs = 10
    model_save_path = "/tmp/saved_model/my_model"

    classifier = fraud_dectection_model_training_mobilenet(folder_path)
    classifier.run_training(folder_path, checkpoint_filepath, epochs)
    classifier.save_model(model_save_path)
    classifier.evaluate("C:\\Users\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_testing")

    # Load the model and evaluate again to confirm it was saved and loaded correctly
    classifier.load_model(model_save_path)
    classifier.evaluate("C:\\Users\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_testing")