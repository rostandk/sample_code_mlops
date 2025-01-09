CODE SAMPLE by Rostand
Code to perform machine learning model deployment through Git instead of making manual changes directely using MLflow GUI.
- The goal is to introduce engineering best practices to ML model deployment: PR, code review, code quality check, model deployment rollback, automatically start A/B testing  etc.
- The data scientists will deploy the ML models through 
  1. changing the ML model configuration file for the environment he/she wants: dev, pre of production
  2. Creating a pull requests
- Above steps will trigger a Github action with all the CI/CD steps: code quality check, unit tests, etc
- After the PR is approved, this ML model is deployed to the choose environment.

Code in relation the MLOPS project from an idea to running A/B experiment in a couple of weeks
- This code is small part of bigger code aiming at adding MLOPS to our entire stack.
- The goal was to signifincanly improve the number of A/B experiments we ran by automating the entire ML pipeline from featuring engineering to model deployment, model monitoring and performing A/B experimenatations.
- This project has allow us to experiment more frequently and deliver quite a lot of business value.
