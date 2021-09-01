
### STEP 1 - IMPORT DATA INFO ###

from Utilities import *
from sklearn.model_selection import train_test_split

path = 'myData'
data = importDataInfo(path)

### STEP 2 - VISUALIZATION of DATA ###

data = balanceData(data, display=False)

### STEP 3 - PREPARE DATA FOR PROCESSING ###

imagesPath, steerings = loadData(path, data)
#print(imagesPath[0], steering[0])

### STEP 4 - SPLIT DATA UP INTO TRAINING and VALIDATION ###
xTrain, xValid, yTrain, yValid = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
print(len(xTrain), len(xValid))

### STEP 5 - AUGMENTATION of DATA

### STEP 6 - PREPROCESSING of IMAGE

### STEP 7 - BATCH GENERATOR

### STEP 8 - CREATE and COMPILE THE MODEL
model = createModel()
model.summary()

### STEP 9 - TRAINING THE MODEL
history = model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch=300,epochs=10,
         validation_data=batchGen(xValid,yValid,100,0), validation_steps=200)

### STEP 10 - SAVE THE MODEL ###
model.save('model.h5')
print('Model Saved')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0,1])
plt.title('loss')
plt.xlabel('Epoch')
plt.show()
