import tensorflow.compat.v1 as tf
import io
import PIL
import numpy as np

def representative_dataset_gen():
  record_iterator = tf.python_io.tf_record_iterator(path='../data/animals_validation_00000-of-00001.tfrecord')
  
  for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    image_stream = io.BytesIO(example.features.feature['image/encoded'].bytes_list.value[0])
    image = PIL.Image.open(image_stream)
    image = image.resize((96, 96))
    image = image.convert('L')
    array = np.array(image)
    array = np.expand_dims(array, axis=2)
    array = np.expand_dims(array, axis=0)
    array = ((array / 127.5) - 1.0).astype(np.float32)
    yield([array])
    
        
converter =tf.lite.TFLiteConverter.from_frozen_graph('../Model/vww_96_grayscale_frozen.pb',
['input'], ['MobilenetV1/Predictions/Reshape_1'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_float_model = converter.convert()

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
#converter.experimental_enable_mlir_converter = True
tflite_quant_model = converter.convert()
open("../Model/animals/vww_96_grayscale_quantized.tflite", "wb").write(tflite_quant_model)


interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)

for cnt in range(5):
    image_path = "../data/test/Bird/b"+str(cnt+1)+".jpg"
    image_data = PIL.Image.open(image_path).resize((96, 96)).convert('L')   

    array = np.array(image_data)
    array = np.expand_dims(array, axis=2)
    array = np.expand_dims(array, axis=0)
    #input_data = (array/127.5 - 1).astype(np.float32)
    input_data = (array - 128.0).astype(np.int8)
    #print(input_data)
    # set input data
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
   