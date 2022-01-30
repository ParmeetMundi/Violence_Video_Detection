# Violence_Video_Detection </br>
# Dataset </br>
Dataset is taken from Kaggle [https://www.kaggle.com/mohamedmustafa/real-life-violence-situations-dataset](url) </br>
It contain 2000 video clips with 1000 of Violence Catagory and 1000 of Non-Violence Catagory</br>
</br>
# Preprocessing
20 Frame are extracted from each video at equal intervals and store in disk</br>
As data is given to model we have to store which frame are extracted from which video, to do this list is created with paths of frames and stored in disk</br>
</br>
# Model 
First, batch generator is created to fetch images from disk and to give to model in batches of size 32</br>
</br>
Model->(Tried other combination but this one has given good results)
</br>
self_model=models.Sequential()</br>
self_model.add( layers.InputLayer(input_shape=(20,img_size,img_size,3)) )</br>
self_model.add(TimeDistributed(layers.Conv2D( 32,(3,3),activation='relu',padding='same',input_shape=(img_size,img_size,3) )))</br>
self_model.add(TimeDistributed(layers.MaxPooling2D(4,4)))</br>
self_model.add(TimeDistributed(layers.Dropout(.25)))</br>
</br>
self_model.add(TimeDistributed(layers.Conv2D(64,(3,3),activation='relu',padding='same')))</br>
self_model.add(TimeDistributed(layers.MaxPooling2D(4,4)))</br>
self_model.add(TimeDistributed(layers.Dropout(.25)))</br>
</br>
self_model.add(TimeDistributed(layers.Conv2D(128,(3,3),activation='relu',padding='same')))</br>
self_model.add(TimeDistributed(layers.MaxPooling2D(2,2)))</br>
self_model.add(TimeDistributed(layers.Dropout(.25)))</br>
</br>
self_model.add(TimeDistributed(layers.Flatten()))</br>
self_model.add(layers.LSTM(200))</br>
self_model.add(layers.Dropout(.25))</br>
</br>
self_model.add(layers.Dense(100,activation="relu"))</br>
self_model.add(layers.Dense(1,activation="sigmoid"))</br>
self_model.build()</br>
</br>

# Results
After 50 Epoches</br>
Training loss: 0.0230</br> 
Training accuracy: 0.9912 </br>
val_loss: 0.4305 </br>
val_accuracy: 0.9025</br>
</br>
precision: [0.90862944, 0.89655172]</br>
recall: [0.895 ,0.91 ]</br>
fscore: [0.90176322, 0.90322581]</br>
support: [200, 200]</br>
