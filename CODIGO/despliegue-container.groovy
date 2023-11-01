pipeline {
    agent any

    stages {
        
        stage('Install MLflow') {
            steps {
                sh 'pip install mlflow'
            }
        }

        stage('Run MLflow Server') {
            steps {
                sh 'mlflow server --host 0.0.0.0 --port 5000'
            }
        }
    }
}
