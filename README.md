# trustworthy_sas_system
Source code for "Take No Shortcuts! Stick to the Rubric: A Method for Building Trustworthy Short Answer Scoring Models" (HELMeTO2023)  

## Launching the operating environment
This code is intended to run on docker and docker-compose.  
If you want to run the preprocessing, training, and evaluation code for SAS models, run the following commands under `/sas`.
```
docker-compose up --build
```
After startup, you can use the `docker exec` command to attach to the container.

## Executing source code
The functions of this study can be accessed from `/sas/src/main.py` or `/sas/src/console.py`.

## Launching the web viewer
if you want to run the web viewer for superficial cue visualization, run the following commands under `/web`.  
```
docker-compose up -d --build
```
After starting, you can access the server for development under `localhost:3000`.
