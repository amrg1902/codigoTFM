pipeline {
    agent any

    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    // Construir la imagen Docker
                    docker.build('mi_imagen_mlflow:latest', '-f Dockerfile /Users/aitormartin-romogonzalez/Documents/GitHub/AAA-TFM/CODIGO/')
                }
            }
        }
    }
}
