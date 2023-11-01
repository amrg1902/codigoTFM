pipeline {
    agent any

    stages {
        stage('Clonar Repositorio') {
            steps {
                script {
                    // Definir la URL del repositorio de GitHub
                    def repoURL = 'https://github.com/amrg1902/AAA-TFM.git'
                    
                    // Clonar el repositorio en la carpeta de trabajo actual
                    checkout([$class: 'GitSCM', branches: [[name: 'master']], doGenerateSubmoduleConfigurations: false, extensions: [], userRemoteConfigs: [[url: repoURL]]])
                }
            }
        }
        // stage('Build Docker Image') {
        //     steps {
        //         script {
        //             // Construir la imagen Docker
        //             docker.build('mi_imagen_mlflow:latest', '-f Dockerfile .')
        //         }
        //     }
        // }
    }
}
