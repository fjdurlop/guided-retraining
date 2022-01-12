import numpy as np
import keras
from datetime import datetime
import matplotlib.pyplot as plt
from numpy.core.arrayprint import DatetimeFormat
import tensorflow as tf
from tensorflow.python.eager.context import num_gpus

##############################################################################
# CLASSES
##############################################################################

class My_model():
    """Sequential model predefined, CNN arquitecture
    """
    def __init__(self,dataset,loaded_model = False,saved_model_dir=None):
        """Instantiate a model of My_model, you need to compile model before training model (my_model.compile_model())

        Params:
            loaded_model: If True -> load model from saved_model_dir
                            False -> create a new Sequential model (See definition for more info) 
            saved_model_dir: File path to the saved model
        """
        self.dataset = dataset
        self.num_classes = 43 if self.dataset == 'gtsrb' else 6 # Change if another dataset is added
        self.model = self.get_model()
        
        
        if(loaded_model):
            try:
                self.model = keras.models.load_model(saved_model_dir)
                print("Model loaded correctly")
            except:
                print("There is a problem with the file path")
        self.last_history = None 

    def this_model(self):
        return self.model

    def get_model(self):
        if(self.dataset =="gtsrb" ):
            return keras.models.Sequential([
                                        keras.layers.Conv2D(32, kernel_size= (3, 3), 
                                                            padding= 'same', activation= 'relu'), 
                                        keras.layers.Conv2D(32, kernel_size= (3, 3), padding= 'same', 
                                                            activation= 'relu'), 
                                        keras.layers.MaxPooling2D(pool_size= (2, 2)),
                                        
                                        keras.layers.Dropout(0.2), 

                                        keras.layers.Conv2D(64, kernel_size= (3, 3), 
                                                            padding= 'same', activation= 'relu'), 
                                        keras.layers.Conv2D(64, kernel_size= (3, 3), padding= 'same', 
                                                            activation= 'relu'), 
                                        keras.layers.MaxPooling2D(pool_size= (2, 2)), 
                                        
                                        keras.layers.Dropout(0.2),

                                        keras.layers.Conv2D(128, kernel_size= (3, 3), 
                                                            padding= 'same', activation= 'relu'), 
                                        keras.layers.Conv2D(128, kernel_size= (3, 3), padding= 'same', 
                                                            activation= 'relu'), 
                                        keras.layers.MaxPooling2D(pool_size= (2, 2)), 
                                        
                                        keras.layers.Dropout(0.2),

                                        keras.layers.Flatten(), 
                                        keras.layers.Dense(512, activation= 'relu'), 
                                        keras.layers.Dropout(0.5), 
                                        keras.layers.Dense(256, activation= 'relu'), 
                                        
                                        keras.layers.Dropout(0.5), 
                                        keras.layers.Dense(self.num_classes, activation = 'softmax')])

        elif(self.dataset == "intel"):
            return keras.models.Sequential([  
                                keras.layers.Conv2D(32, kernel_size= (3, 3), padding= 'same', 
                                                     activation= 'relu'), 
                                 keras.layers.MaxPooling2D(pool_size= (2, 2)), 
                               
                                keras.layers.Conv2D(64, kernel_size= (3, 3), 
                                                     padding= 'same', activation= 'relu'), 

                                 keras.layers.MaxPooling2D(pool_size= (2, 2)), 
  
                                 keras.layers.Conv2D(128, kernel_size= (3, 3), 
                                                     padding= 'same', activation= 'relu'),
                                keras.layers.MaxPooling2D(pool_size= (2, 2)),
                                keras.layers.Dropout(0.3),
                                 keras.layers.Conv2D(128, kernel_size= (3, 3), padding= 'same', 
                                                     activation= 'relu'), 
                                 keras.layers.MaxPooling2D(pool_size= (2, 2)),
                                 keras.layers.Dropout(0.3),

                                 keras.layers.Flatten(),
                                keras.layers.Dropout(0.5), 
                                
                                 keras.layers.Dense(512, activation= 'relu'), 
                                keras.layers.Dropout(0.5), 
                                
                                 keras.layers.Dense(100, activation= 'relu'),
 
                                 keras.layers.Dense(self.num_classes, activation = 'softmax')])
            """
            return keras.models.Sequential([  
                                keras.layers.Conv2D(32, kernel_size= (3, 3), padding= 'same', 
                                                     activation= 'relu'), 
                                 keras.layers.MaxPooling2D(pool_size= (2, 2)), 
                               
                                keras.layers.Conv2D(64, kernel_size= (3, 3), 
                                                     padding= 'same', activation= 'relu'), 

                                 keras.layers.MaxPooling2D(pool_size= (2, 2)), 
  
                                 keras.layers.Conv2D(128, kernel_size= (3, 3), 
                                                     padding= 'same', activation= 'relu'),
                                keras.layers.MaxPooling2D(pool_size= (2, 2)),
                                keras.layers.Dropout(0.3),
                                 keras.layers.Conv2D(128, kernel_size= (3, 3), padding= 'same', 
                                                     activation= 'relu'), 
                                 keras.layers.MaxPooling2D(pool_size= (2, 2)),
                                 keras.layers.Dropout(0.3),

                                 keras.layers.Flatten(),
                                keras.layers.Dropout(0.5), 
                                
                                 keras.layers.Dense(512, activation= 'relu'), 
                                keras.layers.Dropout(0.5), 
                                
                                 keras.layers.Dense(100, activation= 'relu'),
 
                                 keras.layers.Dense(self.num_classes, activation = 'softmax')])
            """


    def compile_model(self):
        try:
            self.model.compile(loss= 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
            print("Model compiled")
        except:
            print("Error with your model or compilation")

    def fit_model(self,x_train,y_train,x_val,y_val,epochs=10,batch_size = 64):
        """Train model
        Params:
            x_train: Training x
            y_train: Training y
            x_val: Validation x
            y_val: Validation y
        """
        start_time = datetime.now()
        history= self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_val, y_val),
        )

        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))

        self.last_history = history
        return history
    
    def evaluate(self,x_test,y_test):
        return self.model.evaluate(x_test,y_test)

    def save(self,dir_save:str):
        """ Save model into dir_save folder
        """
        try:
            self.model.save(dir_save)
            print("Model has been saved")
        except:
            print("Error while trying to save the model")

    def get_weights(self):
        return self.model.weights

##############################################################################
# FUNCTIONS
##############################################################################


def get_data(dataset = "gtsrb",set_type = "Train"):
    """
    Get the numpy arrays of the dataset

    Params:
        set_type: Train,Val,Test 
            or Train_and_adversay

    Returns:
        x_set,y_set (Numpy arrays)
    """
    if (dataset =="gtsrb"):
        if(set_type == "Train"):
            #x_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/data/x_train.npy"
            #y_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/data/y_train.npy"
            x_dir = "D:/data/x_train.npy"
            y_dir = "D:/data/y_train.npy"

        elif (set_type == "Test"):
            #x_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/data/x_test.npy"
            #y_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/data/y_test.npy"
            x_dir = "D:/data/x_test.npy"
            y_dir = "D:/data/y_test.npy"

        elif (set_type == "Val"):
            #x_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/data/x_val.npy"
            #y_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/data/y_val.npy"
            x_dir = "D:/data/x_val.npy"
            y_dir = "D:/data/y_val.npy"   

        elif (set_type == "Train_and_adversary"):
            # Train set + adversarial examples obtained from train set
            x_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/DL_notebooks/adv_examples/train_and_images_fgsm_5000.npy"
            y_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/DL_notebooks/adv_examples/train_and_labels_fgsm_5000.npy"    
        else:
            print("There is not data in your directions, see the function definition") 
    elif (dataset =="intel"):
        if(set_type == "Train"):
            #x_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/data/intel_dataset/x_train_intel_2.npy"
            #y_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/data/intel_dataset/y_train_intel_2.npy"
            x_dir = "D:/data/intel_dataset/x_train_intel_2.npy"
            y_dir = "D:/data/intel_dataset/y_train_intel_2.npy"

        elif (set_type == "Test"):
            #x_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/data/intel_dataset/x_test_intel_2.npy"
            #y_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/data/intel_dataset/y_test_intel_2.npy"
            x_dir = "D:/data/intel_dataset/x_test_intel_2.npy"
            y_dir = "D:/data/intel_dataset/y_test_intel_2.npy"

        elif (set_type == "Val"):
            #x_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/data/intel_dataset/x_val_intel_2.npy"
            #y_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/data/intel_dataset/y_val_intel_2.npy"  
            x_dir = "D:/data/intel_dataset/x_val_intel_2.npy"
            y_dir = "D:/data/intel_dataset/y_val_intel_2.npy"  


        elif (set_type == "Train_and_adversary"):
            # Train set + adversarial examples obtained from train set
            x_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/DL_notebooks/adv_examples/train_and_images_intel_2.npy"
            y_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/DL_notebooks/adv_examples/train_and_labels_intel_2.npy"    
        else:
            print("There is not data in your directions, see the function definition") 

    x_set = np.load(x_dir)
    y_set = np.load(y_dir)

    return x_set,y_set

def get_adversarial_data(dataset = "gtsrb",set_type = "Test_fgsm"):
    """
    Get the numpy arrays of the dataset for test 

    Params:
        set_type: 
            Test_fgsm -> Test set + adversarial examples from FGSM
            Test_jsma -> Test set + adversarial examples from JSMA    
    Returns:
        x_set,y_set (Numpy arrays)
    """
    if dataset =="gtsrb":
        if(set_type == "Test_fgsm"):
            x_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/DL_notebooks/adv_examples/test_and_images_fgsm.npy"
            y_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/DL_notebooks/adv_examples/test_and_labels_fgsm.npy"
        elif (set_type == "Test_jsma"):
            x_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/DL_notebooks/adv_examples/test_and_images_jsma.npy"
            y_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/DL_notebooks/adv_examples/test_and_labels_jsma.npy"
        else:
            print("There is not data in your directions, inside the function")   
    elif dataset =="intel":
        if(set_type == "Test_fgsm"):
            x_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/DL_notebooks/adv_examples/test_and_images_fgsm_intel_2.npy"
            y_dir = "C:/Users/fjdur/Desktop/upc/project_notebooks/github_project/DL_notebooks/adv_examples/test_and_labels_fgsm_intel_2.npy"
    x_set = np.load(x_dir)
    y_set = np.load(y_dir)

    print(x_dir)
    print(y_dir)

    return x_set,y_set


def get_x_of_indexes(index_list,values_list):
    """Obtain a new list with the values of values_list ordered by index_list
        
        Example: random_images_0 = get_x_of_indexes(list_random_of_random_numbers,my_list)

        Returns:
            New list(np.array)
    """
    return np.array([values_list[i] for i in index_list])


def get_model(dataset = "gtsrb"):
    """Get the model structure of project

        Returns:
            Keras sequential model (keras.models.Sequential)
    """
    if dataset == "gtsrb":
        return keras.models.Sequential([
                                        keras.layers.Conv2D(32, kernel_size= (3, 3), 
                                                            padding= 'same', activation= 'relu'), 
                                        keras.layers.Conv2D(32, kernel_size= (3, 3), padding= 'same', 
                                                            activation= 'relu'), 
                                        keras.layers.MaxPooling2D(pool_size= (2, 2)),
                                        
                                        keras.layers.Dropout(0.2), 

                                        keras.layers.Conv2D(64, kernel_size= (3, 3), 
                                                            padding= 'same', activation= 'relu'), 
                                        keras.layers.Conv2D(64, kernel_size= (3, 3), padding= 'same', 
                                                            activation= 'relu'), 
                                        keras.layers.MaxPooling2D(pool_size= (2, 2)), 
                                        
                                        keras.layers.Dropout(0.2),

                                        keras.layers.Conv2D(128, kernel_size= (3, 3), 
                                                            padding= 'same', activation= 'relu'), 
                                        keras.layers.Conv2D(128, kernel_size= (3, 3), padding= 'same', 
                                                            activation= 'relu'), 
                                        keras.layers.MaxPooling2D(pool_size= (2, 2)), 
                                        
                                        keras.layers.Dropout(0.2),

                                        keras.layers.Flatten(), 
                                        keras.layers.Dense(512, activation= 'relu'), 
                                        keras.layers.Dropout(0.5), 
                                        keras.layers.Dense(256, activation= 'relu'), 
                                        
                                        keras.layers.Dropout(0.5), 
                                        keras.layers.Dense(43, activation = 'softmax')])
    elif dataset == "intel":
        return keras.models.Sequential([  
                                keras.layers.Conv2D(32, kernel_size= (3, 3), padding= 'same', 
                                                     activation= 'relu'), 
                                 keras.layers.MaxPooling2D(pool_size= (2, 2)), 
                               
                                keras.layers.Conv2D(64, kernel_size= (3, 3), 
                                                     padding= 'same', activation= 'relu'), 

                                 keras.layers.MaxPooling2D(pool_size= (2, 2)), 
  
                                 keras.layers.Conv2D(128, kernel_size= (3, 3), 
                                                     padding= 'same', activation= 'relu'),
                                keras.layers.MaxPooling2D(pool_size= (2, 2)),
                                keras.layers.Dropout(0.3),
                                 keras.layers.Conv2D(128, kernel_size= (3, 3), padding= 'same', 
                                                     activation= 'relu'), 
                                 keras.layers.MaxPooling2D(pool_size= (2, 2)),
                                 keras.layers.Dropout(0.3),

                                 keras.layers.Flatten(),
                                keras.layers.Dropout(0.5), 
                                
                                 keras.layers.Dense(512, activation= 'relu'), 
                                keras.layers.Dropout(0.5), 
                                
                                 keras.layers.Dense(100, activation= 'relu'),
 
                                 keras.layers.Dense(6, activation = 'softmax')])
        """
        return keras.models.Sequential([  
                                keras.layers.Conv2D(32, kernel_size= (3, 3), padding= 'same', 
                                                     activation= 'relu'), 
                                 keras.layers.MaxPooling2D(pool_size= (2, 2)), 
                               
                                keras.layers.Conv2D(64, kernel_size= (3, 3), 
                                                     padding= 'same', activation= 'relu'), 

                                 keras.layers.MaxPooling2D(pool_size= (2, 2)), 
  
                                 keras.layers.Conv2D(128, kernel_size= (3, 3), 
                                                     padding= 'same', activation= 'relu'),
                                keras.layers.MaxPooling2D(pool_size= (2, 2)),
                                keras.layers.Dropout(0.3),
                                 keras.layers.Conv2D(128, kernel_size= (3, 3), padding= 'same', 
                                                     activation= 'relu'), 
                                 keras.layers.MaxPooling2D(pool_size= (2, 2)),
                                 keras.layers.Dropout(0.3),

                                 keras.layers.Flatten(),
                                keras.layers.Dropout(0.5), 
                                
                                 keras.layers.Dense(512, activation= 'relu'), 
                                keras.layers.Dropout(0.5), 
                                
                                 keras.layers.Dense(100, activation= 'relu'),
 
                                 keras.layers.Dense(6, activation = 'softmax')])
                                 """

def compile_model(model):
    try:
        model.compile(loss= 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
        print("Model compiled")
    except:
        print("Error with your model or compilation")

def fit_model(model,x_train,y_train,x_val,y_val,epochs=10):
    """
    """
    start_time = datetime.now()
    history= model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=128,
        shuffle=True,
        validation_data=(x_val, y_val)
    )

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return history


def see_x_image(x,y):
    ''' Visualize an image with matplotlib

    Params:
        x: (Numpy ndarray (48,48,1))
        y: (Numpy ndarray (43,))
    
    '''
    plt.figure()
    
    plt.imshow((x.reshape((48,48))*255).astype("uint8"))
    plt.title(str(np.argmax(y)))
    plt.axis("off")