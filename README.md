# ObjectDetection
The system was developed as a subsystem of the digital assistant driver to alert the driver of the presence of dangerous objects near the railway track.

A YOLOv8 neural network model architecture was used to detect foreign objects.The model was pre-trained on the user dataset to detect the rail-track, as this object class was not part of the base class set of the trained model.

The file real_time_onemodel.py uses the only model I trained and is capable of recognizing 5 object classes: person, rail-track, car, train and traffic light.

The files real_time_twomodel.py and photo_twomodel.py use a bundle of two models: The one trained by me on a custom dataset for detection only rail_track and the YOLOv8x-seg model, which is in the public domain.

When a person crosses the bounding box of the rail track segment, an audible signal starts playing in a separate stream to signal an obstacle

![image](https://github.com/drinkmorewaterz/ObjectDetection/assets/124874733/2de55e2d-191d-4f97-8273-9b028132e660)

![image](https://github.com/drinkmorewaterz/ObjectDetection/assets/124874733/000b942d-aeec-4b4f-b1e6-cf49a8ac2f91)

![image](https://github.com/drinkmorewaterz/ObjectDetection/assets/124874733/412ad696-ac4d-40e9-9bd9-c74aabe9cd5a)

![image](https://github.com/drinkmorewaterz/ObjectDetection/assets/124874733/d758f357-fe28-49b3-bcfa-72a351e1d354)
