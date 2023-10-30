pipeline {
    agent any // Utiliza cualquier agente disponible en Jenkins

    stages {
        stage('Clonar Repositorio') {
            steps {
                // Utiliza el paso 'checkout' para clonar el repositorio
                checkout([$class: 'GitSCM', branches: [[name: 'main']], userRemoteConfigs: [[url: 'https://github.com/amrg1902/AAA-TFM/tree/main/CODIGO']]])
            }
        }
    }
}
