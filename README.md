action-recognition
============
A Video Vision Transformer (ViViT) model for detecting incidences of contact between between NFL players in play-by-play video footage

![alt text](https://github.com/markbotros1/action-recognition/blob/main/resources/example.gif)

Summary
-------
- Containerized project using Docker to recreate project env (including OpenCV, PyTorch, hugginface, etc...)
- Uploaded data and Docker image to AWS S3 and ECR, respectively 
- Finetuned pretrained ViViT model (from hugginface) on AWS EC2 GPU-enabled instance
- Model based on [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)

Setup
-----
Assuming you have docker installed:
1. Pull data from: https://www.kaggle.com/competitions/nfl-player-contact-detection/data
2. Store data in project directory
   
        ├── ActionRecognition           <- Project directory
        │    ├── data                   <- Data directory
        │         ├── ...               <- Data files
            
3. Build docker image
```
$cd path/to/ActionRecognition
$docker build -t action-rec:latest .
```
4. Run docker container
```
$docker run -v $(pwd)/data:/usr/src/data -v $(pwd)/models:/usr/src/models -it action-rec:latest sh
```

5. Within container: Run inference
```
$python inference.py
```

Project Organization
--------------------
    ├── README.md          <- Top-level README
    |
    ├── models             <- Trained and serialized models
    │     ├── ...          
    │
    ├── requirements.txt   <- Requirements file for reproducing the analysis environment
    |
    ├── .dockerignore      <- Files to omit when building image
    │
    ├── Dockerfile         <- Instructions for building docker image
    │
    ├── config.yaml        <- Model configurations/hyperparameters
    |
    ├── dataset.py         <- Functions and class for building dataset for model training
    |
    ├── train.py           <- Model training
    |
    ├── inference.py       <- Model inference/testing
