# Part B Learning Log: Improving the Playing Card Classifier

## 1. What I was trying to improve

In Part A, I used a normal CNN workflow to classify playing cards.  
In Part B, I tried to make the model better by using more advanced methods.

My main goal was:

- make the model learn small card details better
- reduce overfitting
- make the model more stable during training
- improve the final accuracy on the test set

The final model reached about **81% test accuracy**.

---

# Technique 1: Transfer Learning with EfficientNetV2B2

## The Concept

Transfer learning means using a model that has already learned from many images before.

Instead of training everything from zero, I used **EfficientNetV2B2** with ImageNet weights.  
This model already knows basic image features like:

- lines
- corners
- shapes
- colours
- patterns

This helps because playing cards also have important visual features, such as suits, numbers, and face card patterns.

## Why I used it

A normal CNN starts with no image knowledge.  
That means it needs more time and more data to learn good features.

EfficientNetV2B2 is useful because it already has strong image understanding.  
I only needed to train the new classifier part for my 53 playing card classes.

## Implementation

```python
from tensorflow.keras.applications import EfficientNetV2B2

base_model = EfficientNetV2B2(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,),
    include_preprocessing=True
)

base_model.trainable = False
```

Then I added my own classifier head:

```python
inputs = keras.Input(shape=IMG_SIZE + (3,), name="input_image")

x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

x = layers.BatchNormalization(name="bn_1")(x)
x = layers.Dropout(0.40, name="dropout_1")(x)

x = layers.Dense(512, activation="swish", name="dense_1")(x)
x = layers.BatchNormalization(name="bn_2")(x)
x = layers.Dropout(0.35, name="dropout_2")(x)

x = layers.Dense(256, activation="swish", name="dense_2")(x)
x = layers.BatchNormalization(name="bn_3")(x)
x = layers.Dropout(0.25, name="dropout_3")(x)

outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

model = keras.Model(inputs, outputs)
```

## The Learning

The base model works like a feature extractor.  
It looks at the card image and finds useful patterns.

For example, it can help detect:

- the card number
- the suit shape
- the face card drawing
- the overall card layout

The new dense layers then use those patterns to decide the final card class.

I learned that transfer learning can make training easier because the model does not need to learn all image features from the beginning.

## Impact on Performance

Transfer learning gave the model a much stronger starting point than training from zero.  
It helped the model learn the playing card dataset faster and improved the final result.

---

# Technique 2: Label Smoothing

## The Concept

Label smoothing is a method that makes the training labels less extreme.

Normally, the correct class is treated as **1.0** and every wrong class is treated as **0.0**.

For example, if the correct card is `ace of hearts`, the normal target is like this:

```text
ace of hearts = 1.0
all other cards = 0.0
```

With label smoothing, the correct class is still the highest, but it is not treated as 100% perfect.

For example, with label smoothing:

```text
ace of hearts = about 0.9
other cards = small values
```

## Why I used it

Playing cards can look very similar.  
For example, `five of hearts` and `five of diamonds` both have the same number, but different suit shapes.

If the model becomes too confident, it may overfit and make strong wrong predictions.  
Label smoothing helps reduce over-confidence.

## Implementation

In my notebook, I used:

```python
LABEL_SMOOTHING = 0.1
```

Then I used label smoothing inside the categorical crossentropy part of the loss function:

```python
ce = tf.keras.losses.categorical_crossentropy(
    y_true_one_hot,
    y_pred,
    label_smoothing=LABEL_SMOOTHING
)
```

Because my original labels were integer labels, I first changed them into one-hot labels:

```python
y_true_one_hot = tf.one_hot(y_true_int, depth=num_classes)
```

## The Learning

Label smoothing does not change the model structure.  
It changes how the model learns from the labels.

It tells the model:

> Be confident, but do not be too over-confident.

This can help the model generalise better to validation and test images.

## Impact on Performance

Label smoothing helped make the training more stable.  
Together with weighted focal loss, it helped the model reach about **80.75% test accuracy**.

---

# Technique 3: Safe Data Augmentation for Playing Cards

## The Concept

Data augmentation means changing training images a little bit so the model sees more variety.

For playing cards, I used safe changes only:

- small rotation
- small zoom
- small movement
- small contrast change
- small brightness change if TensorFlow supports it

I did **not** use horizontal flip because flipping a card can make the text and symbols look unrealistic.

## Why I used it

In real photos, cards may not always be perfectly straight.  
They may be slightly rotated, darker, brighter, or shifted.

Data augmentation helps the model learn that these small changes should not change the card class.

## Implementation

```python
augmentation_layers = [
    layers.RandomRotation(0.04),
    layers.RandomZoom(height_factor=(-0.08, 0.05), width_factor=(-0.08, 0.05)),
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    layers.RandomContrast(0.15),
]

if hasattr(layers, "RandomBrightness"):
    augmentation_layers.append(layers.RandomBrightness(0.10))

data_augmentation = keras.Sequential(
    augmentation_layers,
    name="safe_card_augmentation"
)
```

## The Learning

Data augmentation makes the model more flexible.

Without augmentation, the model may only learn the exact images in the training folder.  
With augmentation, the model sees slightly different versions of the same card.

This can reduce overfitting because the model cannot just memorise the training images.

## Impact on Performance

This helped the model handle real image changes better.  
The model became less dependent on one exact card position or lighting condition.

---

# Technique 4: Batch Normalization

## The Concept

Batch Normalization is a layer that makes the values inside the model more stable.

During training, the numbers inside a neural network can change a lot.  
If they change too much, training can become slow or unstable.

Batch Normalization helps keep those numbers in a better range.

## Why I used it

I used Batch Normalization after important layers so the model could train more smoothly.

## Implementation

```python
x = layers.BatchNormalization(name="bn_1")(x)
x = layers.Dropout(0.40, name="dropout_1")(x)

x = layers.Dense(512, activation="swish", name="dense_1")(x)
x = layers.BatchNormalization(name="bn_2")(x)
x = layers.Dropout(0.35, name="dropout_2")(x)

x = layers.Dense(256, activation="swish", name="dense_2")(x)
x = layers.BatchNormalization(name="bn_3")(x)
x = layers.Dropout(0.25, name="dropout_3")(x)
```

## The Learning

Batch Normalization helps the model train in a more controlled way.

It does not directly classify the card.  
Instead, it helps the model learn better by keeping the internal values more balanced.

This is useful when the model is deeper and has many layers.

## Impact on Performance

Batch Normalization helped make the training more stable.  
It also worked together with Dropout to reduce overfitting.

---

# Technique 5: Dropout

## The Concept

Dropout randomly turns off some neurons during training.

This sounds strange, but it helps the model because it cannot depend too much on one small part of the network.

## Why I used it

Playing card classification can overfit because many cards look similar.  
For example, a five of hearts and a five of diamonds may look close if the model only focuses on the number.

Dropout forces the model to use more than one feature.

## Implementation

```python
x = layers.Dropout(0.40)(x)
x = layers.Dropout(0.35)(x)
x = layers.Dropout(0.25)(x)
```

## The Learning

Dropout is like making the model practise without always using the same neurons.

This makes the model stronger because it learns more general patterns.

## Impact on Performance

Dropout helped reduce overfitting.  
It made the model less likely to memorise only the training images.

---

# Technique 6: Weighted Focal Loss and Class Weights

## The Concept

Weighted Focal Loss is a loss function that gives more attention to difficult examples.

Normal loss treats all mistakes more evenly.  
Focal loss focuses more on images that the model finds hard.

Class weights also help when some classes have more training images than others.

## Why I used it

Some card classes may have more training images than others.  
If the model sees one class more often, it may become better at that class and weaker at smaller classes.

Weighted Focal Loss helps balance this problem.

## Implementation

```python
class_weights_tensor = tf.constant(class_weight_values, dtype=tf.float32)

@keras.saving.register_keras_serializable()
def weighted_focal_loss(y_true, y_pred):
    gamma = 2.0

    y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_true_one_hot = tf.one_hot(y_true_int, depth=num_classes)

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    ce = tf.keras.losses.categorical_crossentropy(
        y_true_one_hot,
        y_pred,
        label_smoothing=LABEL_SMOOTHING
    )

    pt = tf.exp(-ce)
    focal = tf.pow(1.0 - pt, gamma) * ce

    weights = tf.gather(class_weights_tensor, y_true_int)

    return focal * weights

loss_function = weighted_focal_loss
```

## The Learning

Focal loss makes easy examples less important and hard examples more important.

For example, if the model already predicts an easy card correctly, it does not need to spend too much learning power on that image.  
Instead, it can focus more on cards it keeps getting wrong.

## Impact on Performance

This helped the model focus on harder card classes.  
It worked together with label smoothing to make the loss function stronger and less over-confident.

---

# Technique 7: Fine-Tuning

## The Concept

Fine-tuning means training part of the pre-trained model after the new classifier head has already learned something.

At first, I froze the EfficientNetV2B2 base model.  
After that, I unfroze the top part of the base model so it could adjust more to playing cards.

## Why I used it

The ImageNet model was trained on general images, not playing cards only.  
Fine-tuning helps the model become more suitable for card images.

## Implementation

```python
base_model.trainable = True

total_layers = len(base_model.layers)
fine_tune_at = int(total_layers * 0.70)

for i, layer in enumerate(base_model.layers):
    if i < fine_tune_at:
        layer.trainable = False
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
```

Then I used a smaller learning rate:

```python
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-5),
    loss=loss_function,
    metrics=["accuracy"]
)
```

## The Learning

Fine-tuning must be done carefully.

If the learning rate is too high, the model may destroy the useful knowledge from the pre-trained model.  
That is why I used a small learning rate, `1e-5`.

## Impact on Performance

Fine-tuning improved validation accuracy.

The best validation accuracy reached about **0.8377**, or **84%**.  
The final test accuracy was about **0.8075**, or **81%**.

---

# Technique 8: Training Callbacks

## The Concept

Callbacks help control training automatically.

I used:

- `ModelCheckpoint` to save the best model
- `ReduceLROnPlateau` to reduce the learning rate when improvement slows down
- `EarlyStopping` to stop training when validation accuracy stops improving

## Implementation

```python
callbacks = [
    keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        mode="max",
        verbose=1
    )
]
```

## The Learning

Callbacks make training safer.

`ModelCheckpoint` keeps the best model, not just the last model.  
`ReduceLROnPlateau` helps the model learn more slowly when it gets stuck.  
`EarlyStopping` stops training before the model wastes time or overfits too much.

## Impact on Performance

This helped save the best version of the model.  
It also prevented the training from running forever.

---

# Extra Change: Showing 20 Random Training Samples

In the original version, the code only showed one batch of images.  
Because the batch size was 16, it only showed 16 examples.

I changed it to unbatch the dataset first, then take exactly 20 images.

```python
SAMPLE_COUNT = 20

for image, label in train_data.unbatch().shuffle(1000, seed=SEED).take(SAMPLE_COUNT):
    sample_images.append(image.numpy().astype("uint8"))
    sample_labels.append(int(label.numpy()))
```

This made the visualisation match the requirement of showing **20 examples**.

---

# Final Result

The final model result was:

```text
Test Loss: 0.9029
Test Accuracy: 0.8075
```

This means the model correctly classified about **81%** of the test images.

The classification report also showed:

```text
accuracy = 0.81
macro avg f1-score = 0.79
weighted avg f1-score = 0.79
```

The model performed well on many card classes, but it was not perfect.  
Some cards were still confused with other cards because many playing cards have very similar shapes and layouts.

---

# What I Learned Overall

From Part B, I learned that improving a computer vision model is not just about adding more layers.

I learned that:

- Transfer learning helps because the model already understands image features.
- Label smoothing helps reduce over-confidence.
- Data augmentation helps the model handle different image positions and lighting.
- Batch Normalization makes training more stable.
- Dropout helps reduce overfitting.
- Weighted Focal Loss helps the model pay more attention to hard examples.
- Fine-tuning helps the pre-trained model adjust to the playing card dataset.
- Callbacks make training safer and more efficient.

The most useful method was **transfer learning with fine-tuning**, because it gave the model a stronger starting point and helped it learn playing card details better.

Label smoothing was also useful because it helped the model avoid being too confident when many card classes looked similar.

---

# Short Presentation Version

For Part B, my first technique was **transfer learning with EfficientNetV2B2**.  
This means I used a model that was already trained on many images before.  
I added my own classifier layers for the 53 playing card classes.

My second technique was **label smoothing**.  
This made the model less over-confident by using softer labels instead of only using 1.0 for the correct class and 0.0 for all wrong classes.

I also used safe data augmentation, Batch Normalization, Dropout, Weighted Focal Loss, fine-tuning, and training callbacks.

These methods helped the model learn better and reduce overfitting.  
The best validation accuracy was about **84%**, and the final test accuracy was about **81%**.

The model improved compared with the previous result, but it still had problems with some difficult cards.  
This shows that the model is better than the basic version, but there is still room to improve.
