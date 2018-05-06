from keras.applications.nasnet import NASNetLarge
from coremltools.converters.keras import convert

model = NASNetLarge(include_top=True, weights=None)

model.summary()

coreml_model = convert(model)
coreml_model.save('nasmobile.mlmodel')