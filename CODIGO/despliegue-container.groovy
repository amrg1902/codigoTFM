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
                    customImage.withRegistry('https://index.docker.io/v1/', 'ce19e7ac-fe9e-47b6-aa1d-65aaf111e2d8') {
                        customImage.push()
                    }
                }
            }
        }
    }
}
