pipeline {
    agent any

    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    // Construir la imagen Docker
                    def dockerfilePath = '/Users/aitormartin-romogonzalez/Documents/GitHub/AAA-TFM/CODIGO/'
                    dir(dockerfilePath) {
                        def customImage = docker.build('mi_imagen_mlflow:latest', '.')
                    }
                }
            }
        }
    }
}
