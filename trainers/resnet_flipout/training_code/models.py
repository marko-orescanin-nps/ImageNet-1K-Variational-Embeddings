import tensorflow as tf
from resnet50VI_ub import *

IMG_SIZE = (224,224)
IMG_SHAPE = IMG_SIZE + (3,)
NUM_CLASSES = 1000
APPROX_IMAGENET_TRAIN_IMAGES = 1281167

def create_model(hparams):

    model_type = hparams.model_type.lower()

    if model_type == "resnet50_flipout":
        base_model, preprocess_input = create_resnet50_flipout_model(hparams)
    else:
        print("unsupported model type %s" % (model_type))
        return None

    model = config_model(hparams, preprocess_input, base_model)
    return model, base_model


def config_model(hparams, preprocess_input, base_model):

    #Iterature through checkpoint.model.layers and place into new model. Then call model(training=False)?

    #Setup architecture for checkpoint restoration
    checkpoint = tf.train.Checkpoint(model=base_model)  

    #Restore the pre-trained ResNet50V2 checkpoint to the model object
    #checkpoint.restore("/data/kraken/coastal_project/imagenetcp/checkpoint-29")
    checkpoint.restore(hparams.checkpoint_path)
    
    #Freeze layers in the base model 
    for layer in checkpoint.model.layers:
        layer.trainable = False

    new_model = tf.keras.models.Model(inputs = checkpoint.model.layers[0].input, outputs = checkpoint.model.layers[-2].output) 

    #Dense
    embed_layer_1 = tf.keras.layers.Dense(units=hparams.neurons, activation=hparams.activation_fn)(new_model.output)
    #Dense
    embed_layer_2 = tf.keras.layers.Dense(units=hparams.neurons, activation=hparams.activation_fn)(embed_layer_1)

    classification_head = tf.keras.layers.Dense(8, activation='softmax', name="class_head")(embed_layer_2)

    model = tf.keras.models.Model(inputs=new_model.input, outputs=classification_head)

    return model 

def unfreeze_layers(model, base_model, layers):
    #Set all to trainable, then fine tune below 

    model.trainable = True
    print("Number of layers in the base model (these are frozen at first): ", len(base_model.layers))

    
    count = 0
    for layer in base_model.layers[:layers]:
        layer.trainable = False
        count +=1 
    for layer in base_model.layers[:layers]:
        print("Name of layer that will remain frozen: ", layer.name)

    print("Number of layers frozen layers remaining: " + str(count))


def create_resnet50_flipout_model(hparams):

    IMG_SIZE = (hparams.input_shape_x, hparams.input_shape_y)
    IMG_SHAPE = IMG_SIZE + (3,)
    INITIAL_NUM_CLASSES = 1000

    #Put the custom pre-processing code here
    preprocess_input = tf.keras.applications.resnet50.preprocess_input
    

    base_model = resnet50_variational(
          input_shape=(224,224,3),
          num_classes=NUM_CLASSES, 
          prior_stddev=None,
          dataset_size=APPROX_IMAGENET_TRAIN_IMAGES,
          stddev_mean_init=0.00097618,
          stddev_stddev_init=0.41994,
          tied_mean_prior=True,)

    print("Resnet 50 VI Flipout Base Model: ")
    base_model.summary(show_trainable=True)
    print("Number of layers in base_model: ")
    print(len(base_model.layers))
    return base_model, preprocess_input



