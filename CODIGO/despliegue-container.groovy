pipeline {
    agent any

    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    sh "sudo docker build -t mlflow:latest ."
                    
                }
            }
        }
    }
}
