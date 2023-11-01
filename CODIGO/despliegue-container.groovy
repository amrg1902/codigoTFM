pipeline {
    agent any

    stages {
        stage('Acceso a Dockerfile en el Workspace') {
            steps {
                script {
                    // Obtener la ruta al directorio de trabajo (workspace)
                    def workspaceDir = env.WORKSPACE

                    // Definir la ruta completa al Dockerfile en el workspace
                    def rutaDockerfile = "${workspaceDir}/Dockerfile"

                    // Verificar si el Dockerfile existe en el workspace
                    if (fileExists(rutaDockerfile)) {
                        echo "El Dockerfile existe en la ruta: ${rutaDockerfile}"
                        
                        // Realizar acciones adicionales con el Dockerfile si es necesario
                    } else {
                        echo "El Dockerfile no se encontró en la ruta: ${rutaDockerfile}"
                    }
                }
            }
        }
        stage('Construir imagen Docker') {
            steps {
                script {
                    def dockerfilePath = "${workspaceDir}/Dockerfile"
                    def dockerImageName = "mi_imagen_docker:latest"
                    sh "docker build -t ${dockerImageName} -f ${dockerfilePath} ."
        }
    }
}

    }
}
