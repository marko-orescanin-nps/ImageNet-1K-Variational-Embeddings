import tensorflow as tf


def create_model(hparams):

    model_type = hparams.model_type.lower()

    if model_type == "resnet50":
        base_model, preprocess_input = create_resnet50_model(hparams)
    else:
        print("unsupported model type %s" % (model_type))
        return None

    model = config_model(hparams, preprocess_input, base_model)
    return model


def config_model(hparams, preprocess_input, base_model):

    # Input layer
    inputs = tf.keras.Input(shape=(hparams.input_shape_x, hparams.input_shape_y, 3))

    # Augmentation layer support
    flip = tf.keras.layers.RandomFlip("horizontal")
    rotation = tf.keras.layers.RandomRotation(0.1)
    brightness = tf.keras.layers.RandomBrightness(factor=0.2)
    contrast = tf.keras.layers.RandomContrast(factor=0.2)

    # Output layer
    # Transforms a multidimensional layer into a 1D layer.
    if hparams.pool_layer == 'avg2d':
        pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
    elif hparams.pool_layer == 'flat':
        pooling_layer = tf.keras.layers.Flatten()
    else:
        raise("Error, pooling layer used not found")
    #This will be 8, for the 8 classes 
    prediction_layer = tf.keras.layers.Dense(units=hparams.num_classes, activation="softmax")

    x = inputs
    if hparams.data_augmentation:
        x = flip(x)
        x = rotation(x)
        x = brightness(x)
        x = contrast(x)
    x = preprocess_input(x)
    #Freeze base layer for initial fixed parameter training
    x = base_model(x, training=False)  #Set base model to ResNet50V2
    x = pooling_layer(x)
    if hparams.embedded_drop:
        x = tf.keras.layers.Dropout(0.2)(x)
        #Add a single embedded layer for optimal performance
        x = tf.keras.layers.Dense(units=hparams.neurons, activation=hparams.activation_fn)(x)
        #x = tf.keras.layers.Dropout(0.2)(x)
        #Add a single embedded layer for optimal performance
        x = tf.keras.layers.Dense(units=hparams.neurons, activation=hparams.activation_fn)(x)
    if hparams.output_drop:
        x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    #Output is is softmax prediction over the 8 classes 
    model = tf.keras.Model(inputs, outputs)

    return model, base_model 


def unfreeze_layers(model, base_model, layers):
    #Set all to trainable, then fine tune below 
    model.trainable = True

    layer_name = base_model._name
    print("Base layer name: ", layer_name)

    base_layers = model.get_layer(str(layer_name))

    print("Number of layers in the base model (these are frozen at first): ", len(base_layers.layers))
    count = 0
    for layer in base_layers.layers[: (len(base_layers.layers) - 1 - layers)]:
        layer.trainable = False
        count += 1

    for layer in base_layers.layers[(len(base_layers.layers) - 1 - layers):]:
        print("Name of layer that is now trainable: ", layer.name)
    
    print("Number of layers frozen layers remaining (in function): " + str(count))


def create_resnet50_model(hparams):

    IMG_SIZE = (hparams.input_shape_x, hparams.input_shape_y)
    IMG_SHAPE = IMG_SIZE + (3,)
    preprocess_input = tf.keras.applications.resnet50.preprocess_input

    base_model = tf.keras.applications.ResNet50(
        input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    )
    print("Resnet 50 Regular Base Model: ")
    base_model.summary()
    print("Number of layers in base_model: ")
    print(len(base_model.layers))
    base_model.trainable = False

    return base_model, preprocess_input



