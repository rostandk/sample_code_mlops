# Machine Learning Model Deployment with Git and MLflow

This repository demonstrates a workflow for deploying machine learning models through Git instead of making manual changes directly using the MLflow GUI.

## Overview

The goal of this project is to introduce engineering best practices to ML model deployment, including:

- Pull requests (PR) for changes
- Code reviews
- Code quality checks
- Model deployment rollback
- Automatically starting A/B testing

By adopting this workflow, data scientists can deploy ML models in a more robust and automated way.

## Deployment Workflow

1. **Update Configuration**: Data scientists modify the ML model configuration file for the desired environment (e.g., `dev`, `pre-prod`, or `production`).
2. **Create a Pull Request**: Submit a PR for the changes.
3. **Trigger CI/CD**: The PR triggers a GitHub Action that runs the following steps:
   - Code quality checks
   - Unit tests
   - Other CI/CD steps
4. **Approval and Deployment**: Once the PR is approved, the ML model is deployed to the selected environment.

## Project Context

This code is part of a larger MLOps project designed to automate the entire machine learning pipeline. The broader project includes:

- Feature engineering
- Model deployment
- Model monitoring
- Running A/B experiments

### Goals of the Project

- **Automate ML Pipelines**: From idea inception to running A/B experiments in a couple of weeks.
- **Increase Experimentation Frequency**: By automating the ML pipeline, we significantly improved the number of A/B experiments conducted.
- **Deliver Business Value**: Faster experimentation and deployment cycles have allowed us to deliver substantial business impact.

### Code Scope

This repository contains a small portion of the overall MLOps codebase, focusing on the deployment process. It highlights the transition from manual processes to a CI/CD-driven workflow for ML model deployment.

---

_Authored by Rostand_
