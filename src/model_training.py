import joblib
import comet_ml
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,TensorBoard,EarlyStopping

from utils.common_functions import read_yaml
from src.logger import get_logger
from src.custom_exception import CustomException
from src.base_model import BaseModel
from config.paths_config import *

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self,data_path,config_path=CONFIG_PATH):

        self.data_path= data_path
        self.config = read_yaml(config_path)

        self.experiment = comet_ml.Experiment(
            api_key="BSZvyyW8JhRJWISs1v6l2OfK4",
            project_name="anime_recommender_system_ann",
            workspace="hrithikrag"
        )

        logger.info("Initializing Model Training & COMET ML....")
    

    def load_data(self):

        try:
            X_train_array = joblib.load(X_TRAIN_ARRAY)
            X_test_array = joblib.load(X_TEST_ARRAY)
            y_train = joblib.load(Y_TRAIN)
            y_test = joblib.load(Y_TEST)

            logger.info("Data loaded successfully for Model Trainig....")
            return X_train_array,X_test_array,y_train,y_test
        
        except Exception as e:
            logger.error("Error while loading data for Model Trainig....")
            raise CustomException("Failed to load data for model training",e)
        

    def train_model(self):

        try:
            X_train_array,X_test_array,y_train,y_test = self.load_data()

            n_users = len(joblib.load(USER2USER_ENCODED))
            n_anime = len(joblib.load(ANIME2ANIME_ENCODED))

            base_model = BaseModel(config_path=CONFIG_PATH)

            model = base_model.RecommenderNet(n_users=n_users,n_anime=n_anime)

            start_lr = self.config["training_parameters"]["start_lr"]
            min_lr = self.config["training_parameters"]["min_lr"]
            max_lr = self.config["training_parameters"]["max_lr"]
            batch_size = self.config["training_parameters"]["batch_size"]

            ramup_epochs = self.config["training_parameters"]["ramup_epochs"]
            sustain_epochs = self.config["training_parameters"]["sustain_epochs"]
            exp_decay = self.config["training_parameters"]["exp_decay"]

            def lrfn(epoch):

                if epoch<ramup_epochs:
                    return (max_lr-start_lr)/ramup_epochs*epoch + start_lr
                elif epoch<ramup_epochs+sustain_epochs:
                    return max_lr
                else:
                    return (max_lr-min_lr) * exp_decay ** (epoch-ramup_epochs-sustain_epochs)+min_lr
            

            lr_callback = LearningRateScheduler(lambda epoch:lrfn(epoch) , verbose=0)

            model_checkpoint = ModelCheckpoint(filepath=CHECKPOINT_FILE_PATH,
                                               save_weights_only=True,
                                               monitor=self.config["training_parameters"]["monitor"],
                                               mode=self.config["training_parameters"]["mode"],
                                               save_best_only=True)

            early_stopping = EarlyStopping(patience=self.config["training_parameters"]["patience"],
                                           monitor=self.config["training_parameters"]["monitor"],
                                           mode=self.config["training_parameters"]["mode"],
                                           restore_best_weights=True)

            my_callbacks = [model_checkpoint,lr_callback,early_stopping]


            os.makedirs(os.path.dirname(CHECKPOINT_FILE_PATH),exist_ok=True)
            os.makedirs(MODEL_DIR,exist_ok=True)
            os.makedirs(WEIGHTS_DIR,exist_ok=True)

            try:
                logger.info("Model training started.....")

                history = model.fit(
                        x=X_train_array,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=self.config["training_parameters"]["epochs"],
                        verbose=1,
                        validation_data = (X_test_array,y_test),
                        callbacks=my_callbacks
                    )
                
                model.load_weights(CHECKPOINT_FILE_PATH)

                logger.info("Model training Completed.....") 

                for epoch in range(len(history.history['loss'])):
                    train_loss = history.history["loss"][epoch]
                    val_loss = history.history["val_loss"][epoch]

                    self.experiment.log_metric('train_loss',train_loss,step=epoch)
                    self.experiment.log_metric('val_loss',val_loss,step=epoch)
            
            except Exception as e:
                logger.error("Error while Model Trainig....")
                raise CustomException("Model training failed.....",e)
            
            self.save_model_weights(model)

        except Exception as e:
            logger.error("Error while Model Trainig....")
            raise CustomException("Model Trainig Process failed",e)
        

    def extract_weights(self,layer_name,model):

        try:
            weight_layer = model.get_layer(layer_name)
            weights = weight_layer.get_weights()[0]
            weights = weights/np.linalg.norm(weights,axis=1).reshape((-1,1))

            logger.info(f"Extracting weights for {layer_name}....")

            return weights
        
        except Exception as e:
            logger.error("Error during Weight Extraction Process....")
            raise CustomException("Weight Extraction failed",e)
    

    def save_model_weights(self,model):

        try:
            model.save(MODEL_PATH)
            logger.info(f"Model saved to {MODEL_PATH}")

            user_weights = self.extract_weights('user_embedding',model)
            anime_weights = self.extract_weights('anime_embedding',model)

            joblib.dump(user_weights,USER_WEIGHTS_PATH)
            joblib.dump(anime_weights,ANIME_WEIGHTS_PATH)

            self.experiment.log_asset(MODEL_PATH)
            self.experiment.log_asset(ANIME_WEIGHTS_PATH)
            self.experiment.log_asset(USER_WEIGHTS_PATH)

            logger.info("User and Anime weights saved sucesfully....")

        except Exception as e:
            logger.error("Error during saving model and weights....")
            raise CustomException("Failed in saving model and weights",e)
        

if __name__=="__main__":
    model_trainer = ModelTraining(PROCESSED_DIR)
    model_trainer.train_model()
    




        


