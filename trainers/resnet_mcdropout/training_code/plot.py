import pandas as pd
from matplotlib import pyplot as plt
import params 
import os

hparams = params.get_hparams()

src = os.path.join(hparams.model_dir, "log.csv") 
df = pd.read_csv(src, delimiter=';')


lines = df.plot.line(x='epoch', y=['accuracy', 'val_accuracy'])
plt.title('MC Dropout Accuracy Curves')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'], loc='lower right')
plt.show()
plt.savefig('plot.png')
